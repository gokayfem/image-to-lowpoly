"""Four-way triangle count + hard-failure checks + disclosed topology (CONTRACT §6).

Usage (venv; launches Blender itself in background for counts B and C, CPU only):
    python3 scripts/validate_variant.py --mesh <mesh.json> \
        [--blend <cat_N.blend>] [--glb <cat_N.glb>] [--out <validation.json>]

Defaults: blend/glb = <mesh dir>/<name>.blend|.glb, out = <mesh dir>/validation.json.
Exit code 0 = no hard failures, 2 = hard failures, 1 = tool error.

Counts:
  A  len(triangles) in mesh.json
  B  reopened .blend, EVERY object of any type (hidden, excluded or unlinked objects are
     made visible / linked in memory first), evaluated depsgraph, sum of
     calc_loop_triangles() over depsgraph.object_instances (so instancing is counted),
     plus loop triangles of mesh datablocks no object uses.  Never len(mesh.polygons).
  C  .glb imported into an empty Blender scene, evaluated loop triangles over every object.
  D  pure-Python GLB container parse (no Blender): sum(indices.count/3) per primitive x
     number of nodes referencing that mesh; non-TRIANGLES modes are errors.

The same file runs inside Blender (probe mode, "-- --probe ...") to collect B, C, the
.blend inventory, self-intersections (mathutils.bvhtree) and centroid ray visibility.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the folder that holds catlib/
from catlib import paths as _paths  # noqa: E402  (resolves the case: --case / LOWPOLY_CASE / cwd)
ROOT = _paths.ROOT

try:
    import bpy  # noqa: F401
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

import numpy as np  # noqa: E402

from catlib import geom, meshio, paths  # noqa: E402

GLTF_TRIANGLES = 4
# ID collections that a variant .blend may contain (anything else non-empty is extra).
UI_ID_TYPES = {"screens", "workspaces", "window_managers", "scenes"}


# =============================================================================================
# Blender probe
# =============================================================================================

def _id_inventory() -> dict:
    import bpy
    inv = {}
    for attr in dir(bpy.data):
        if attr.startswith("_") or attr in ("bl_rna", "rna_type", "filepath", "all_ids"):
            continue
        try:
            c = getattr(bpy.data, attr)
        except Exception:
            continue
        if not hasattr(c, "__iter__") or isinstance(c, (str, bytes)) or not hasattr(c, "rna_type"):
            continue
        try:
            names = [x.name for x in c]
        except Exception:
            continue
        if names:
            inv[attr] = names
    return inv


def _loop_tris(mesh) -> int:
    mesh.calc_loop_triangles()
    return len(mesh.loop_triangles)


def _count_evaluated_scene() -> dict:
    """Sum loop triangles over every evaluated instance in the current scene."""
    import bpy
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    per_obj: dict = {}
    total = 0
    cache: dict = {}
    for inst in dg.object_instances:
        ob = inst.object
        key = ob.name
        if key not in cache:
            n = 0
            if ob.type in {"MESH", "CURVE", "SURFACE", "META", "FONT", "CURVES", "POINTCLOUD",
                           "GREASEPENCIL", "VOLUME"}:
                try:
                    me = ob.to_mesh()
                    n = _loop_tris(me) if me is not None else 0
                    ob.to_mesh_clear()
                except RuntimeError:
                    n = 0
            cache[key] = n
        n = cache[key]
        total += n
        rec = per_obj.setdefault(key, {"type": ob.type, "instances": 0, "loop_triangles_each": n})
        rec["instances"] += 1
        if inst.is_instance:
            rec["is_instance"] = True
    return {"total": total, "per_object": per_obj}


def _reveal_everything() -> list:
    """Make every object/collection visible and linked to the scene (in memory only) so the
    depsgraph evaluates all of it.  Returns notes about what had to be revealed."""
    import bpy
    notes = []
    scene = bpy.context.scene
    for vl in scene.view_layers:
        def walk(lc):
            if lc.exclude:
                notes.append(f"collection {lc.name} excluded from view layer")
                lc.exclude = False
            if lc.hide_viewport:
                lc.hide_viewport = False
            for ch in lc.children:
                walk(ch)
        walk(vl.layer_collection)
    for coll in bpy.data.collections:
        if coll.hide_viewport or coll.hide_render:
            notes.append(f"collection {coll.name} hidden")
            coll.hide_viewport = False
        if not any(coll.name == c.name for c in scene.collection.children_recursive):
            notes.append(f"collection {coll.name} not linked to the scene")
            scene.collection.children.link(coll)
    in_scene = {o.name for o in scene.objects}
    for ob in bpy.data.objects:
        if ob.name not in in_scene:
            notes.append(f"object {ob.name} not linked to the scene")
            scene.collection.objects.link(ob)
        if ob.hide_viewport or ob.hide_get() or ob.hide_render:
            notes.append(f"object {ob.name} hidden")
            ob.hide_viewport = False
            ob.hide_set(False)
    bpy.context.view_layer.update()
    return notes


def probe(blend: Path, glb: Path, obj_name: str) -> dict:
    import bpy
    from mathutils.bvhtree import BVHTree
    from mathutils import Vector

    res: dict = {"blend": str(blend), "glb": str(glb)}
    # ------------------------------------------------------------------ B: the .blend
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    inv = _id_inventory()
    res["blend_inventory"] = inv
    res["blend_objects"] = [{"name": o.name, "type": o.type, "data": o.data.name if o.data else None,
                             "users_collection": [c.name for c in o.users_collection],
                             "modifiers": [m.type for m in o.modifiers],
                             "particle_systems": len(getattr(o, "particle_systems", [])),
                             "instance_type": o.instance_type,
                             "hide_viewport": o.hide_viewport, "hide_render": o.hide_render}
                            for o in bpy.data.objects]
    res["blend_collections"] = [c.name for c in bpy.data.collections]
    used_meshes = {o.data.name for o in bpy.data.objects if o.type == "MESH" and o.data}
    orphan_meshes = {me.name: _loop_tris(me) for me in bpy.data.meshes if me.name not in used_meshes}
    res["reveal_notes"] = _reveal_everything()
    cnt = _count_evaluated_scene()
    res["B_per_object"] = cnt["per_object"]
    res["B_orphan_mesh_triangles"] = orphan_meshes
    res["B"] = cnt["total"] + sum(orphan_meshes.values())

    ob = bpy.data.objects.get(obj_name)
    if ob is not None and ob.type == "MESH":
        me = ob.data
        res["cat_object"] = {
            "found": True,
            "matrix_world_identity": bool(np.allclose(np.array(ob.matrix_world), np.eye(4), atol=1e-9)),
            "polygons": len(me.polygons),
            "polygon_sizes": sorted({len(p.vertices) for p in me.polygons}),
            "smooth_polygons": int(sum(p.use_smooth for p in me.polygons)),
            "shape_keys": len(me.shape_keys.key_blocks) if me.shape_keys else 0,
            "modifiers": [m.type for m in ob.modifiers],
            "materials": [m.name if m else None for m in me.materials],
            "has_region_id": "region_id" in me.attributes and me.attributes["region_id"].domain == "FACE",
            "custom_props": sorted(k for k in ob.keys()),
            "mesh_users": me.users,
        }
        dg = bpy.context.evaluated_depsgraph_get()
        ev = ob.evaluated_get(dg)
        em = ev.to_mesh()
        V = np.array([tuple(v.co) for v in em.vertices], float)
        em.calc_loop_triangles()
        LT = np.array([tuple(t.vertices) for t in em.loop_triangles], np.int64).reshape(-1, 3)
        ev.to_mesh_clear()
        res["cat_object"]["eval_vertices"] = V.tolist()
        res["cat_object"]["eval_loop_triangles"] = LT.tolist()
        polys = [tuple(p.vertices) for p in me.polygons]
        res["cat_object"]["polygons_list"] = [list(p) for p in polys] if len(polys) < 20000 else None
        # self intersections (non-adjacent pairs: no shared welded vertex)
        if len(LT):
            tree = BVHTree.FromPolygons([tuple(v) for v in V], [tuple(t) for t in LT],
                                        all_triangles=True, epsilon=0.0)
            pairs = tree.overlap(tree)
            wid = geom.weld_ids(V)
            WT = wid[LT]
            si = set()
            for i, j in pairs:
                if i >= j:
                    continue
                if set(WT[i].tolist()) & set(WT[j].tolist()):
                    continue
                si.add((int(i), int(j)))
            res["self_intersecting_pairs"] = sorted(si)
            # centroid ray visibility from the 162 sphere directions
            cen = V[LT].mean(1)
            c0 = 0.5 * (V.min(0) + V.max(0))
            R = float(np.linalg.norm(V - c0, axis=1).max()) * 3 + 1.0
            vis = np.zeros(len(LT), bool)
            for dvec in geom.sphere_directions(2):
                dv = Vector(tuple(-dvec))
                for k in range(len(LT)):
                    if vis[k]:
                        continue
                    o = Vector(tuple(cen[k] + dvec * R))
                    hit = tree.ray_cast(o, dv)
                    if hit[2] == k:
                        vis[k] = True
            res["centroid_visible"] = vis.tolist()
    else:
        res["cat_object"] = {"found": False}

    # ------------------------------------------------------------------ C: the .glb
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.import_scene.gltf(filepath=str(glb))
        cnt = _count_evaluated_scene()
        res["C"] = cnt["total"]
        res["C_per_object"] = cnt["per_object"]
        res["C_objects"] = [{"name": o.name, "type": o.type} for o in bpy.data.objects]
        mats = []
        for o in bpy.data.objects:
            if o.type == "MESH":
                Vg = np.array([tuple(o.matrix_world @ v.co) for v in o.data.vertices], float)
                if len(Vg):
                    mats.append(Vg)
        if mats:
            Vg = np.concatenate(mats)
            res["C_bbox_min"] = Vg.min(0).tolist()
            res["C_bbox_max"] = Vg.max(0).tolist()
    except Exception as e:  # noqa: BLE001
        res["C"] = None
        res["C_error"] = repr(e)
    return res


def probe_main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--blend", required=True, type=Path)
    ap.add_argument("--glb", required=True, type=Path)
    ap.add_argument("--object", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    res = probe(a.blend, a.glb, a.object)
    a.out.write_text(json.dumps(res))
    print("PROBE_OK")
    return 0


# =============================================================================================
# D: pure-Python GLB parse
# =============================================================================================

def parse_glb(path: Path) -> dict:
    data = Path(path).read_bytes()
    errors = []
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        return {"D": None, "errors": [f"not a glTF 2 binary (magic {magic!r}, version {version})"]}
    off = 12
    gltf = None
    while off < min(length, len(data)):
        clen, ctype = struct.unpack_from("<I4s", data, off)
        if ctype == b"JSON":
            gltf = json.loads(data[off + 8: off + 8 + clen].decode("utf-8"))
        off += 8 + clen
    if gltf is None:
        return {"D": None, "errors": ["no JSON chunk"]}
    acc = gltf.get("accessors", [])
    nodes = gltf.get("nodes", [])
    refs: dict = {}
    for n in nodes:
        if "mesh" in n:
            refs[n["mesh"]] = refs.get(n["mesh"], 0) + 1
        if "EXT_mesh_gpu_instancing" in n.get("extensions", {}):
            errors.append(f"node {n.get('name')} uses EXT_mesh_gpu_instancing")
        if any(k in n for k in ("translation", "rotation", "scale", "matrix")):
            errors.append(f"node {n.get('name')} has a non-identity transform field "
                          f"{[k for k in ('translation', 'rotation', 'scale', 'matrix') if k in n]}")
    for ext in gltf.get("extensionsUsed", []):
        if "draco" in ext.lower() or "meshopt" in ext.lower():
            errors.append(f"compressed geometry extension {ext}")
    total = 0
    meshes = []
    bmin = np.full(3, np.inf)
    bmax = np.full(3, -np.inf)
    for mi, mesh in enumerate(gltf.get("meshes", [])):
        mt = 0
        for pi, prim in enumerate(mesh.get("primitives", [])):
            mode = prim.get("mode", GLTF_TRIANGLES)
            if mode != GLTF_TRIANGLES:
                errors.append(f"mesh {mi} primitive {pi}: mode {mode} is not TRIANGLES")
                continue
            if "indices" in prim:
                n = acc[prim["indices"]]["count"]
            else:
                n = acc[prim["attributes"]["POSITION"]]["count"]
            if n % 3:
                errors.append(f"mesh {mi} primitive {pi}: index count {n} not divisible by 3")
            mt += n // 3
            pa = acc[prim["attributes"]["POSITION"]]
            if "min" in pa and "max" in pa:
                bmin = np.minimum(bmin, pa["min"])
                bmax = np.maximum(bmax, pa["max"])
        r = refs.get(mi, 0)
        meshes.append({"mesh": mi, "name": mesh.get("name"), "triangles": mt, "node_refs": r})
        total += mt * r
    out = {"D": total, "meshes": meshes, "nodes": len(nodes), "errors": errors,
           "cameras": len(gltf.get("cameras", [])),
           "lights": len(gltf.get("extensions", {}).get("KHR_lights_punctual", {}).get("lights", [])),
           "extensionsUsed": gltf.get("extensionsUsed", [])}
    if np.isfinite(bmin).all():
        out["bbox_min_yup"] = bmin.tolist()
        out["bbox_max_yup"] = bmax.tolist()
        # glTF Y-up (x, y, z) = Blender (x, z, -y)
        out["bbox_min_blender"] = [bmin[0], -bmax[2], bmin[1]]
        out["bbox_max_blender"] = [bmax[0], -bmin[2], bmax[1]]
    return out


# =============================================================================================
# venv side
# =============================================================================================

def run_probe(blend: Path, glb: Path, obj_name: str, work: Path) -> dict:
    out = work / "_probe.json"
    if out.exists():
        out.unlink()
    cmd = [str(paths.BLENDER), "-b", "--factory-startup", "--python", str(Path(__file__).resolve()),
           "--", "--probe", "--blend", str(blend), "--glb", str(glb), "--object", obj_name,
           "--out", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not out.exists():
        raise RuntimeError(f"Blender probe failed (rc {p.returncode}):\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    res = json.loads(out.read_text())
    out.unlink()
    return res


EXPECTED_ID_TYPES = {"objects", "meshes", "materials", "collections"}


def validate(mesh_path: Path, blend: Path | None = None, glb: Path | None = None,
             out: Path | None = None, census_size: int = 512) -> dict:
    t0 = time.time()
    mesh_path = Path(mesh_path).resolve()
    raw = json.loads(mesh_path.read_text())
    hard: list[str] = []
    schema = meshio.validate_dict(raw)
    hard += [f"schema: {e}" for e in schema if not e.startswith("over_budget")]
    m = meshio.CatMesh(raw, mesh_path)
    name = m.name or "cat"
    blend = Path(blend) if blend else mesh_path.parent / f"{name}.blend"
    glb = Path(glb) if glb else mesh_path.parent / f"{name}.glb"
    out = Path(out) if out else mesh_path.parent / "validation.json"
    budget = m.budget
    A = m.n_triangles

    rep: dict = {"mesh": str(mesh_path), "blend": str(blend), "glb": str(glb), "name": name,
                 "budget": budget, "counts": {"A_mesh_json": A}}
    # ---------------------------------------------------------------- B, C via Blender
    pr = None
    if not blend.is_file():
        hard.append(f"missing .blend {blend}")
    if not glb.is_file():
        hard.append(f"missing .glb {glb}")
    if blend.is_file() and glb.is_file():
        try:
            pr = run_probe(blend, glb, name, out.parent)
        except RuntimeError as e:
            hard.append(f"probe: {e}")
    if pr:
        rep["counts"]["B_blend_evaluated"] = pr["B"]
        rep["counts"]["C_glb_reimported"] = pr.get("C")
        rep["blend"] = {k: pr[k] for k in ("blend_inventory", "blend_objects", "blend_collections",
                                          "reveal_notes", "B_per_object", "B_orphan_mesh_triangles")}
        rep["blend"]["cat_object"] = {k: v for k, v in pr["cat_object"].items()
                                      if k not in ("eval_vertices", "eval_loop_triangles", "polygons_list")}
        rep["glb_reimport"] = {k: pr.get(k) for k in ("C_per_object", "C_objects", "C_error",
                                                        "C_bbox_min", "C_bbox_max")}
    # ---------------------------------------------------------------- D
    D = None
    if glb.is_file():
        gp = parse_glb(glb)
        D = gp["D"]
        rep["counts"]["D_glb_parsed"] = D
        rep["glb_parse"] = gp
        hard += [f"glb: {e}" for e in gp["errors"]]
        if gp.get("cameras") or gp.get("lights"):
            hard.append(f"glb contains cameras ({gp.get('cameras')}) / lights ({gp.get('lights')})")
    # ---------------------------------------------------------------- count agreement
    counts = rep["counts"]
    vals = [counts.get(k) for k in ("A_mesh_json", "B_blend_evaluated", "C_glb_reimported", "D_glb_parsed")]
    counts["all_agree"] = all(v is not None for v in vals) and len(set(vals)) == 1
    counts["final_triangles"] = max(v for v in vals if v is not None) if any(v is not None for v in vals) else None
    if not counts["all_agree"]:
        hard.append(f"triangle counts disagree or missing: A={vals[0]} B={vals[1]} C={vals[2]} D={vals[3]}")
    if isinstance(budget, int):
        over = [v for v in vals if v is not None and v > budget]
        if over:
            hard.append(f"over budget: max count {max(over)} > budget {budget}")
        counts["budget"] = budget
        counts["budget_unused"] = budget - counts["final_triangles"] if counts["final_triangles"] is not None else None
    # ---------------------------------------------------------------- .blend contents
    if pr:
        inv = pr["blend_inventory"]
        extra_types = sorted(set(inv) - EXPECTED_ID_TYPES - UI_ID_TYPES)
        for t in extra_types:
            hard.append(f"extra datablocks in .blend: {t} {inv[t]}")
        objs = pr["blend_objects"]
        if len(objs) != 1 or objs[0]["name"] != name or objs[0]["type"] != "MESH":
            hard.append(f"expected exactly one MESH object {name!r}, found "
                        f"{[(o['name'], o['type']) for o in objs]}")
        if len(inv.get("meshes", [])) != 1:
            hard.append(f"expected exactly one mesh datablock, found {inv.get('meshes', [])}")
        exp_coll = "CAT_" + (name[4:] if name.startswith("cat_") else name)
        if pr["blend_collections"] != [exp_coll]:
            hard.append(f"expected exactly one collection {exp_coll!r}, found {pr['blend_collections']}")
        if pr["reveal_notes"]:
            hard.append(f"hidden/unlinked content in .blend: {pr['reveal_notes']}")
        if pr["B_orphan_mesh_triangles"]:
            hard.append(f"orphan mesh datablocks: {pr['B_orphan_mesh_triangles']}")
        mats = inv.get("materials", [])
        bad_mats = [x for x in mats if x not in meshio.MATERIALS]
        if bad_mats:
            hard.append(f"unknown materials in .blend: {bad_mats}")
        co = pr["cat_object"]
        if co.get("found"):
            if co["modifiers"]:
                hard.append(f"modifiers on {name}: {co['modifiers']}")
            if co["shape_keys"]:
                hard.append(f"shape keys on {name}: {co['shape_keys']}")
            if not co["matrix_world_identity"]:
                hard.append(f"{name} transform is not identity")
            if co["polygon_sizes"] != [3]:
                hard.append(f"{name} has non-triangle polygons (sizes {co['polygon_sizes']}; "
                            f"{co['polygons']} polygons vs {pr['B']} evaluated loop triangles)")
            if not co["has_region_id"]:
                hard.append(f"{name} lacks the face attribute region_id")
            for prop in ("regions", "landmarks"):
                if prop not in co["custom_props"]:
                    hard.append(f"{name} lacks custom property {prop!r}")
            if co["smooth_polygons"]:
                rep.setdefault("notes", []).append(f"{co['smooth_polygons']} smooth-shaded polygons")
            # geometry mirror check: .blend evaluated geometry == mesh.json
            EV = np.asarray(pr["cat_object"]["eval_vertices"], float).reshape(-1, 3)
            ET = np.asarray(pr["cat_object"]["eval_loop_triangles"], np.int64).reshape(-1, 3)
            same = (EV.shape == m.vertices.shape and len(ET) == len(m.triangles)
                    and np.abs(EV - m.vertices).max() < 1e-6
                    and np.array_equal(np.sort(np.sort(ET, 1), 0), np.sort(np.sort(m.triangles, 1), 0)))
            rep["blend"]["geometry_matches_mesh_json"] = bool(same)
            if not same:
                hard.append("blend geometry does not match mesh.json (vertices/triangles differ)")
        else:
            hard.append(f"object {name!r} not found in .blend")
        if pr.get("C_objects") is not None and len([o for o in pr["C_objects"] if o["type"] == "MESH"]) != 1:
            hard.append(f"glb re-import yields {pr.get('C_objects')} (expected one mesh object)")
    # ---------------------------------------------------------------- topology (mesh.json)
    if not [e for e in schema if e.startswith("triangles") or e.startswith("vertices")]:
        topo = geom.topology_report(m.vertices, m.triangles)
        if topo["degenerate_triangles"]:
            hard.append(f"degenerate triangles (area < 1 mm^2 or edge < 1 mm): {topo['degenerate_triangles'][:20]}")
        if topo["duplicate_triangles_same_indices"] or topo["duplicate_triangles_coincident_positions"]:
            hard.append(f"duplicate triangles: same indices {topo['duplicate_triangles_same_indices'][:10]}, "
                        f"coincident positions {topo['duplicate_triangles_coincident_positions'][:10]}")
        if topo["min_z"] < -0.002:
            hard.append(f"min z {topo['min_z']:.4f} < -0.002 m")
        if topo["min_extent_ratio"] < 0.12:
            hard.append(f"billboard guard: smallest bbox extent is {100 * topo['min_extent_ratio']:.1f}% "
                        f"of the largest (< 12%)")
        vis = pr.get("centroid_visible") if pr else None
        if vis is not None and len(vis) != len(m.triangles):
            vis = None
        hid = geom.hidden_triangles(m.vertices, m.triangles, 2, census_size, vis)
        lm_dist = {}
        if m.landmarks:
            names = list(m.landmarks)
            dist, _ = geom.point_mesh_distance(np.stack([m.landmarks[n] for n in names]),
                                               m.vertices, m.triangles)
            lm_dist = {n: float(d * 1000) for n, d in zip(names, dist)}
        disclosed = {
            "components": topo["components"],
            "component_triangles": topo["component_triangles"],
            "components_after_weld": topo["components_after_weld"],
            "boundary_edges": topo["boundary_edges"],
            "non_manifold_edges": topo["non_manifold_edges"],
            "closed_manifold": topo["closed_manifold"],
            "inconsistent_winding_edges": topo["inconsistent_winding_edges"],
            "signed_volume_m3": topo["signed_volume_m3"],
            "component_signed_volume_m3": topo["component_signed_volume_m3"],
            "components_with_negative_volume": topo["components_with_negative_volume"],
            "unwelded_coincident_vertex_groups": topo["unwelded_coincident_vertex_groups"],
            "unused_vertices": topo["unused_vertices"],
            "self_intersecting_pairs": len(pr.get("self_intersecting_pairs", [])) if pr else None,
            "self_intersecting_pair_list": (pr.get("self_intersecting_pairs", [])[:200] if pr else None),
            "hidden_triangles": hid["hidden_triangles"],
            "hidden_triangle_ids": hid["hidden_triangle_ids"],
            "hidden_census": {k: hid[k] for k in ("directions", "raster_size", "hidden_by_raster_only")},
            "flat_redundant_vertices": topo["flat_redundant_vertices"],
            "flat_angle_tol_deg": topo["flat_angle_tol_deg"],
            "bbox_min": topo["bbox_min"], "bbox_max": topo["bbox_max"],
            "bbox_extent_m": topo["bbox_extent_m"], "min_extent_ratio": topo["min_extent_ratio"],
            "min_z": topo["min_z"], "min_triangle_area_m2": topo["min_triangle_area_m2"],
            "min_edge_m": topo["min_edge_m"],
            "landmark_surface_dist_mm": lm_dist,
            "landmarks_off_surface_gt_2mm": [n for n, d in lm_dist.items() if d > 2.0],
        }
        # glb bbox sanity (Y-up conversion)
        if glb.is_file() and rep.get("glb_parse", {}).get("bbox_min_blender") is not None:
            gmin = np.array(rep["glb_parse"]["bbox_min_blender"])
            gmax = np.array(rep["glb_parse"]["bbox_max_blender"])
            ok = bool(np.allclose(gmin, topo["bbox_min"], atol=1e-4) and np.allclose(gmax, topo["bbox_max"], atol=1e-4))
            disclosed["glb_bbox_matches_mesh"] = ok
            if not ok:
                hard.append(f"glb bbox {gmin.tolist()}..{gmax.tolist()} != mesh bbox "
                            f"{topo['bbox_min']}..{topo['bbox_max']}")
        rep["disclosed"] = disclosed
        rep["degenerate_triangles"] = topo["degenerate_triangles"]
        rep["duplicate_triangles"] = {"same_indices": topo["duplicate_triangles_same_indices"],
                                      "coincident_positions": topo["duplicate_triangles_coincident_positions"]}
    # ---------------------------------------------------------------- allocation table
    if len(m.tri_region) == A:
        alloc = {r: int(sum(1 for x in m.tri_region if x == r)) for r in meshio.REGIONS}
        rep["region_allocation"] = {r: {"triangles": n, "pct": round(100 * n / max(A, 1), 1)}
                                    for r, n in alloc.items()}
        rep["material_allocation"] = {mt: int(sum(1 for x in m.tri_material if x == mt))
                                      for mt in meshio.MATERIALS}
    rep["hard_failures"] = hard
    rep["valid"] = not hard
    rep["elapsed_s"] = round(time.time() - t0, 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Four-way triangle count and topology validation.")
    ap.add_argument("--mesh", required=True, type=Path)
    ap.add_argument("--blend", type=Path)
    ap.add_argument("--glb", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    try:
        rep = validate(a.mesh, a.blend, a.glb, a.out)
    except Exception as e:  # noqa: BLE001
        print(f"validate_variant: tool error: {e!r}", file=sys.stderr)
        return 1
    c = rep["counts"]
    print(f"{rep['name']}: A={c.get('A_mesh_json')} B={c.get('B_blend_evaluated')} "
          f"C={c.get('C_glb_reimported')} D={c.get('D_glb_parsed')} budget={rep['budget']} "
          f"agree={c.get('all_agree')}")
    for h in rep["hard_failures"]:
        print("HARD FAILURE:", h)
    return 0 if rep["valid"] else 2


if __name__ == "__main__":
    if IN_BLENDER:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
        try:
            rc = probe_main(argv)
        except Exception:
            import traceback
            traceback.print_exc()
            rc = 1
        sys.exit(rc)
    sys.exit(main())
