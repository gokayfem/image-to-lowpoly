"""Build cat_<N>.blend and cat_<N>.glb from a catmesh-1 mesh.json (CONTRACT §6).

Run with Blender (CPU only, no rendering, no GPU lock needed):
    blender -b --factory-startup \
        --python scripts/assemble_variant.py -- --mesh <mesh.json> [--out-dir <dir>]

Output names use mesh.json["name"] (e.g. cat_100, or a baseline name).  The .blend holds
exactly one collection "CAT_<suffix>" (suffix = name without a leading "cat_") with exactly
one mesh object <name>: identity transform, flat shaded, triangles only, one material slot
per used material (plain Principled BSDF base colour from contract/scene.json), integer
face attribute "region_id" (index into the custom property "regions"), landmarks in the
custom property "landmarks" (JSON).  Nothing else is saved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the folder that holds catlib/
from catlib import paths as _paths  # noqa: E402  (resolves the case: --case / LOWPOLY_CASE / cwd)
ROOT = _paths.ROOT

import bpy  # noqa: E402

from catlib import meshio, paths  # noqa: E402


def collection_name(name: str) -> str:
    suffix = name[4:] if name.startswith("cat_") else name
    return f"CAT_{suffix}"


def purge_everything() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.cameras,
                 bpy.data.lights, bpy.data.collections, bpy.data.worlds, bpy.data.images,
                 bpy.data.node_groups, bpy.data.curves, bpy.data.textures):
        for idb in list(coll):
            coll.remove(idb)


def build(mesh_path: Path, out_dir: Path, scene_json: Path) -> dict:
    d = json.loads(mesh_path.read_text())
    errs = [e for e in meshio.validate_dict(d) if not e.startswith("over_budget")]
    if errs:
        raise SystemExit("assemble_variant: invalid mesh.json: " + "; ".join(errs))
    m = meshio.CatMesh(d, mesh_path)
    name = m.name
    scene_cfg = json.loads(scene_json.read_text())
    colours = scene_cfg["materials_linear_rgb"]

    purge_everything()
    scene = bpy.context.scene
    coll = bpy.data.collections.new(collection_name(name))
    scene.collection.children.link(coll)

    me = bpy.data.meshes.new(name)
    me.from_pydata(m.vertices.tolist(), [], m.triangles.tolist())
    me.update(calc_edges=True)
    # Deliberately no me.validate(): it would silently delete duplicate / degenerate faces
    # and hide them from the validator.  The .blend must mirror mesh.json exactly.
    if len(me.polygons) != m.n_triangles:
        raise SystemExit(f"assemble_variant: Blender mesh has {len(me.polygons)} polygons, "
                         f"mesh.json has {m.n_triangles} triangles")
    for p in me.polygons:
        p.use_smooth = False

    used = [mat for mat in meshio.MATERIALS if mat in set(m.tri_material)]
    for mat_name in used:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        rgb = colours[mat_name]
        bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
        bsdf.inputs["Roughness"].default_value = 0.8
        mat.diffuse_color = (rgb[0], rgb[1], rgb[2], 1.0)
        me.materials.append(mat)
    slot = {mn: i for i, mn in enumerate(used)}
    me.polygons.foreach_set("material_index", [slot[x] for x in m.tri_material])

    attr = me.attributes.new("region_id", "INT", "FACE")
    attr.data.foreach_set("value", m.region_ids().tolist())

    obj = bpy.data.objects.new(name, me)
    coll.objects.link(obj)
    obj["regions"] = json.dumps(list(meshio.REGIONS))
    obj["landmarks"] = json.dumps({k: [float(c) for c in v] for k, v in m.landmarks.items()})
    obj["budget"] = int(m.budget)
    obj["source_mesh"] = str(mesh_path)
    obj["catmesh_format"] = meshio.FORMAT

    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    blend = out_dir / f"{name}.blend"
    glb = out_dir / f"{name}.glb"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), compress=False, copy=False)

    for o in bpy.data.objects:
        o.select_set(o == obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(
        filepath=str(glb), export_format="GLB", use_selection=True, export_apply=True,
        export_yup=True, export_cameras=False, export_lights=False, export_extras=True,
        export_draco_mesh_compression_enable=False, export_materials="EXPORT",
        export_animations=False)
    res = {"name": name, "collection": coll.name, "blend": str(blend), "glb": str(glb),
           "triangles_mesh_json": m.n_triangles, "polygons": len(me.polygons),
           "materials": used}
    print("ASSEMBLE_RESULT " + json.dumps(res))
    return res


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mesh", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: the mesh.json folder")
    ap.add_argument("--scene", type=Path, default=paths.SCENE_JSON)
    a = ap.parse_args(argv)
    mesh = a.mesh.resolve()
    build(mesh, (a.out_dir or mesh.parent).resolve(), a.scene)
    return 0


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    sys.exit(main(argv))
