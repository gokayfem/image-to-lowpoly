"""Build the fused proxy cat from the fitted skeleton (CONTRACT §11) - CPU only.

    python3 scripts/build_proxy.py                # mesh, sdf, labels, landmarks, blend, previews
    python3 scripts/build_proxy.py --lock         # write contract/camera.json (locked) +
                                                           #   scene.json["orbit_target"]
    python3 scripts/build_proxy.py --report       # proxy/fit_report.json + renders/ref_overlay.png
                                                           #   (official if proxy/renders/ref_silhouette.png exists)
    (internal) Blender -b --factory-startup --python scripts/build_proxy.py -- --write-blend
                                                           # proxy/proxy.blend, collection CAT_PROXY

Method: every skeleton part is meshed as a dense closed tube, voxelised by generalised
winding number (majority over the three axis directions, robust to self-overlap), turned
into a signed distance field (exact-ish near the surface from dense surface samples, EDT
farther out), the parts are combined by a polynomial smooth-min union (per-part blend
radius), clipped by the ground half-space z >= 0 (flat soles), and polygonised with
marching cubes (scikit-image).  Outputs:

    proxy/proxy_mesh.npz   vertices, triangles (+ tri_region_id, region_names)
    proxy/sdf.npz          origin, spacing, sdf (negative inside)
    proxy/part_labels.npz  per-voxel part label for catlib.proxy.region_of
    proxy/landmarks3d.json 16 landmarks on the fused surface
    proxy/proxy.blend      collection CAT_PROXY with object cat_proxy
    proxy/build_report.json watertightness, z-min, fused-vs-parts silhouette IoU, counts
    proxy/previews/        numpy-rasterised shaded novel views (diagnostic, not official)
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the folder that holds catlib/
from catlib import paths as _paths  # noqa: E402  (resolves the case: --case / LOWPOLY_CASE / cwd)
ROOT = _paths.ROOT
PROXY = ROOT / "proxy"
BLENDER = str(_paths.BLENDER)

# smooth-union blend radius per part (m), applied in this order
UNION_ORDER = (("torso", 0.0), ("neck", 0.014), ("head", 0.010), ("ear_R", 0.003), ("ear_L", 0.003),
               ("leg_FR", 0.013), ("leg_FL", 0.013), ("leg_HR", 0.016), ("leg_HL", 0.016),
               ("tail", 0.008))
SDF_BLUR_VOXELS = 1.0
UNBLURRED_PARTS = ("ear_R", "ear_L")   # thin pinnae: smooth-unioned after the blur       # Gaussian sigma applied to the fused field before marching cubes
PART_REGION = {"torso": "torso", "neck": "neck", "head": "head", "muzzle": "head", "ear_R": "ears",
               "ear_L": "ears", "leg_FR": "leg_FR", "leg_FL": "leg_FL", "leg_HR": "leg_HR",
               "leg_HL": "leg_HL", "tail": "tail"}
EYE_RADIUS = 0.008
RAW_MC = ROOT / ".cache" / "proxy_mc_raw.npz"
DECIMATED = ROOT / ".cache" / "proxy_decimated.npz"
STAGING = ROOT / ".cache" / "proxy_staging"      # all derived files are written here first and
                                                   # then os.replace()d into proxy/ together
STAGED_FILES = ("sdf.npz", "part_labels.npz", "proxy_mesh.npz", "landmarks3d.json",
                "proxy.blend", "build_report.json")
TARGET_TRIS = 60000
NOSE_RADIUS = 0.008
CHIN_RADIUS = 0.012          # chin/jaw underside belongs to "head" even where the neck tube is nearer


# =============================================================== Blender side
def write_blend(target_tris: int):
    """Blender (CPU, no render): raw marching-cubes mesh -> merge -> Decimate(collapse) to
    ~target_tris -> clamp z >= 0 -> proxy/proxy_mesh.npz (geometry) + proxy/proxy.blend."""
    import bpy
    import bmesh
    import numpy as np
    z = np.load(RAW_MC)
    V = np.asarray(z["vertices"], float)
    T = np.asarray(z["triangles"], np.int64)
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    for blocks in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights, bpy.data.materials):
        for b in list(blocks):
            blocks.remove(b)
    me = bpy.data.meshes.new("cat_proxy")
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.ravel())
    me.loops.add(T.size)
    me.loops.foreach_set("vertex_index", T.ravel())
    me.polygons.add(len(T))
    me.polygons.foreach_set("loop_start", np.arange(0, T.size, 3))
    me.polygons.foreach_set("loop_total", np.full(len(T), 3))
    me.update()
    me.validate()
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=1e-7)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("cat_proxy", me)
    col = bpy.data.collections.new("CAT_PROXY")
    bpy.context.scene.collection.children.link(col)
    col.objects.link(ob)
    n0 = len(me.polygons)
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = min(1.0, target_tris / max(n0, 1))
    mod.use_collapse_triangulate = True
    dg = bpy.context.evaluated_depsgraph_get()
    oe = ob.evaluated_get(dg)
    me2 = bpy.data.meshes.new_from_object(oe)
    ob.modifiers.clear()
    old = ob.data
    ob.data = me2
    me2.name = "cat_proxy"
    bpy.data.meshes.remove(old)
    bm = bmesh.new()
    bm.from_mesh(me2)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    for v in bm.verts:
        if v.co.z < 0.0:
            v.co.z = 0.0
    nm = sum(1 for e in bm.edges if not e.is_manifold)
    bm.to_mesh(me2)
    bm.free()
    me2.calc_loop_triangles()
    nv = len(me2.vertices)
    co = np.empty(nv * 3)
    me2.vertices.foreach_get("co", co)
    nt = len(me2.loop_triangles)
    tv = np.empty(nt * 3, np.int64)
    me2.loop_triangles.foreach_get("vertices", tv)
    np.savez_compressed(DECIMATED, vertices=co.reshape(-1, 3),
                        triangles=tv.reshape(-1, 3).astype(np.int32))
    ob["note"] = "INFERRED proxy cat (target/prior), not a deliverable variant"
    bpy.ops.wm.save_as_mainfile(filepath=str(STAGING / "proxy.blend"), compress=True)
    print(f"[build_proxy/blender] raw {n0} -> {nt} triangles, {nv} vertices, non-manifold edges {nm}; "
          f"wrote proxy.blend + proxy_mesh.npz")


# =============================================================== venv side
def _np():
    import numpy as np
    return np


def winding_inside(V, T, origin, h, shape):
    """Voxel centres inside a closed (possibly self-overlapping) triangle mesh.
    Generalised winding along +axis for each of the 3 axes; majority vote."""
    np = _np()
    from catlib import raster
    votes = np.zeros(shape, np.int8)
    for ax in range(3):
        a1, a2 = [k for k in range(3) if k != ax]
        n1, n2, n3 = shape[a1], shape[a2], shape[ax]
        q = (V - origin) / h
        P = np.stack([q[:, a1] + 0.5, q[:, a2] + 0.5], 1)[T]            # (T,3,2) "pixels"
        tri, rows, c0, c1 = raster._spans(P, n1, n2)                      # W = n1 (cols), H = n2 (rows)
        if len(tri) == 0:
            continue
        cnt = c1 - c0 + 1
        ft = np.repeat(tri, cnt)
        fr = np.repeat(rows, cnt)
        start = np.cumsum(cnt) - cnt
        fc = np.repeat(c0, cnt) + (np.arange(int(cnt.sum())) - np.repeat(start, cnt))
        A, B, C = q[T[ft, 0]], q[T[ft, 1]], q[T[ft, 2]]
        nrm = np.cross(B - A, C - A)
        nz = nrm[:, ax]
        ok = np.abs(nz) > 1e-12
        px, py = fc.astype(float), fr.astype(float)
        # plane: nrm . (p - A) = 0 -> solve for the ax coordinate
        d = (nrm[:, a1] * (px - A[:, a1]) + nrm[:, a2] * (py - A[:, a2]))
        za = A[:, ax] - d / np.where(ok, nz, 1.0)
        k = np.ceil(za).astype(np.int64)                                 # first voxel above
        sgn = np.where(nz < 0, 1, -1)                                    # entering if normal points down
        keep = ok & (k < n3)
        k = np.clip(k, 0, n3 - 1)
        acc = np.zeros((n1, n2, n3), np.int32)
        np.add.at(acc, (fc[keep], fr[keep], k[keep]), sgn[keep])
        wind = np.cumsum(acc, axis=2) > 0
        # back to grid axis order
        order = [None, None, None]
        order[a1], order[a2], order[ax] = 0, 1, 2
        votes += np.transpose(wind, order).astype(np.int8)
    return votes >= 2


def dense_samples(part, sk, spacing=0.0006, seed=0):
    np = _np()
    from catlib import geom
    V, T = sk.tube_mesh(part, sk.default_ts(part, 16), 96, dome_rings=16)   # smooth end domes
    A = geom.tri_areas(V, T)
    n = int(A.sum() / spacing ** 2) + 1
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(T), size=n, p=A / A.sum())
    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    S = ((1 - r1)[:, None] * V[T[idx, 0]] + (r1 * (1 - r2))[:, None] * V[T[idx, 1]]
         + (r1 * r2)[:, None] * V[T[idx, 2]])
    return V, T, np.concatenate([V, S])


def part_sdf(part, sk, origin, h, shape, pad=0.02):
    """SDF of one part restricted to its bbox + pad: returns (slices, sdf_sub float32)."""
    np = _np()
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree
    V, T, S = dense_samples(part, sk)
    lo = np.floor((V.min(0) - pad - origin) / h).astype(int)
    hi = np.ceil((V.max(0) + pad - origin) / h).astype(int) + 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.array(shape))
    sub_origin = origin + lo * h
    sub_shape = tuple(int(x) for x in hi - lo)
    inside = winding_inside(V, T, sub_origin, h, sub_shape)
    d_out = distance_transform_edt(~inside) * h
    d_in = distance_transform_edt(inside) * h
    sdf = np.where(inside, -(d_in - 0.5 * h), d_out - 0.5 * h).astype(np.float32)
    band = np.abs(sdf) < 3.5 * h
    ii = np.argwhere(band)
    pts = sub_origin + ii * h
    dist, _ = cKDTree(S).query(pts, workers=-1)
    sdf[band] = np.where(inside[band], -dist, dist).astype(np.float32)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return sl, sdf


def smin(a, b, k):
    np = _np()
    if k <= 0:
        return np.minimum(a, b)
    hh = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - hh) + a * hh - k * hh * (1 - hh)


def build(spacing: float, sdf_only: bool = False):
    np = _np()
    from skimage.measure import marching_cubes
    from catlib import geom, raster
    from catlib import skeleton as sk
    sys.path.insert(0, str(_paths.SCRIPTS))
    import fit_camera_proxy as F
    t0 = time.time()
    skel = sk.load_skeleton(PROXY / "skeleton.json")
    fit = json.loads((PROXY / "fit_params.json").read_text())
    P = fit["params"]
    cam = F.camera_from(P)
    # grid: bbox of all parts + 5 cm; z=0 mid-voxel so the ground clip lands exactly on 0
    allV = np.concatenate([sk.tube_mesh(p, sk.default_ts(p, 4), 24)[0] for p in skel.parts.values()])
    lo = allV.min(0) - 0.05
    hi = allV.max(0) + 0.05
    lo[2] = -0.05
    h = spacing
    k0 = math.ceil(-lo[2] / h)
    origin = np.array([lo[0], lo[1], -(k0 + 0.5) * h])
    shape = tuple(int(x) for x in np.ceil((hi - origin) / h).astype(int) + 1)
    print(f"[build_proxy] grid {shape} spacing {h*1000:.1f} mm origin {origin.round(4).tolist()}", flush=True)
    BIG = np.float32(0.2)
    total = np.full(shape, BIG, np.float32)
    best = np.full(shape, BIG, np.float32)
    labels = np.full(shape, -1, np.int8)
    names = []
    late = []
    for name, k in UNION_ORDER:
        if name not in skel.parts:
            continue
        sl, s = part_sdf(skel[name], sk, origin, h, shape)
        if name == "torso":
            torso_sdf = (sl, s)
        if name in UNBLURRED_PARTS:
            late.append((sl, s, k))           # unioned after the blur (keeps thin tips)
        else:
            cur = total[sl]
            total[sl] = s if not names else smin(cur, s, k).astype(np.float32)
        bsub = best[sl]
        better = s < bsub
        bsub[better] = s[better]
        lsub = labels[sl]
        lsub[better] = len(names)
        names.append(name)
        print(f"   part {name:7s} sub {s.shape} blend {k*1000:.0f} mm  ({time.time()-t0:.1f}s)", flush=True)
    zc = origin[2] + np.arange(shape[2]) * h
    from scipy.ndimage import gaussian_filter
    total = gaussian_filter(total, SDF_BLUR_VOXELS).astype(np.float32)   # removes sampling grain
    for sl, s, k in late:
        total[sl] = smin(total[sl], s, k).astype(np.float32)
    total = np.maximum(total, -zc[None, None, :].astype(np.float32))
    # far-field sanity: the union keeps BIG outside every part box
    verts, faces, _, _ = marching_cubes(total, level=0.0, spacing=(h, h, h))
    verts = verts + origin
    faces = faces.astype(np.int64)
    vol = (verts[faces[:, 0]] * np.cross(verts[faces[:, 1]], verts[faces[:, 2]])).sum()
    if vol < 0:
        faces = faces[:, ::-1].copy()
    # keep the largest connected component
    lab = geom._components(len(verts), faces)
    cnt = np.bincount(lab)
    removed = int(len(faces) - cnt.max())
    if removed > 0.002 * len(faces):
        raise SystemExit(f"fused surface splits into components {sorted(cnt.tolist())[::-1][:5]}: "
                         "a part is detached (refusing to drop it silently)")
    faces = faces[lab == np.argmax(cnt)]
    used = np.unique(faces)
    remap = -np.ones(len(verts), np.int64)
    remap[used] = np.arange(len(used))
    verts = verts[used]
    faces = remap[faces]
    verts[:, 2] = np.maximum(verts[:, 2], 0.0)        # MC interpolation round-off at the sole plane
    print(f"[build_proxy] marching cubes: {len(faces)} triangles, {len(verts)} vertices, "
          f"removed {removed} tris in small components ({time.time()-t0:.1f}s)", flush=True)
    import shutil
    shutil.rmtree(STAGING, ignore_errors=True)
    STAGING.mkdir(parents=True)
    np.savez_compressed(STAGING / "sdf.npz", origin=origin, spacing=np.float64(h), sdf=total,
                        note=np.array("signed distance (m), negative inside; trilinear via catlib.proxy"))
    if sdf_only:                                       # regenerate proxy/sdf.npz only (no Blender)
        import os
        os.replace(STAGING / "sdf.npz", PROXY / "sdf.npz")
        print(f"[build_proxy] wrote {PROXY / 'sdf.npz'} only ({time.time()-t0:.1f}s)", flush=True)
        return
    RAW_MC.parent.mkdir(exist_ok=True)
    np.savez(RAW_MC, vertices=verts, triangles=faces)
    raw_tris = int(len(faces))
    r = subprocess.run([BLENDER, "-b", "--factory-startup", "--python", str(Path(__file__).resolve()),
                        "--", "--write-blend", str(TARGET_TRIS)], capture_output=True, text=True)
    print("\n".join(l for l in r.stdout.splitlines() if "build_proxy" in l or "Error" in l), flush=True)
    if r.returncode != 0 or "build_proxy/blender" not in r.stdout:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit("Blender decimation / proxy.blend write failed")
    zz = np.load(DECIMATED)
    verts, faces = np.asarray(zz["vertices"], float), np.asarray(zz["triangles"], np.int64)
    # landmarks on the fused surface
    lm_model = {k: np.asarray(v, float) for k, v in fit["landmarks_model"].items()}
    lm3d, lm_info = {}, {}
    o = cam.location
    # tail_base: where the tail centreline leaves the torso part (torso SDF crosses 0), i.e. the
    # tail/rump junction seen in the photo
    tail = skel["tail"]
    tt = np.linspace(0, 0.5, 101)
    cc = tail.center(tt)
    tsl, tsd = torso_sdf
    ii = np.rint((cc - origin) / h).astype(int) - np.array([x.start for x in tsl])
    ii = np.clip(ii, 0, np.array(tsd.shape) - 1)
    out_t = tsd[ii[:, 0], ii[:, 1], ii[:, 2]] > 0
    j = int(np.argmax(out_t)) if out_t.any() else 0
    lm_model["tail_base"] = cc[j]
    tail_base_t = float(tt[j])
    for n, p in lm_model.items():
        dist, cp = geom.point_mesh_distance(p[None], verts, faces)
        method = "closest_point"
        q = cp[0]
        if n.startswith("ear_tip_"):
            # the ear's apex: fused-surface vertex farthest along the ear axis (CONTRACT §12)
            ear = skel["ear_" + n[-1]]
            axis = ear.tangent(1.0)
            near = np.flatnonzero(np.linalg.norm(verts - ear.center(1.0), axis=1) < 0.02)
            j = near[np.argmax(verts[near] @ axis)]
            q, method = verts[j], "ear_apex_vertex"
        elif not n.startswith("paw_"):
            d = p - o
            L = np.linalg.norm(d)
            t = geom.ray_hits(o, d / L, verts, faces)
            if np.isfinite(t).any():
                tf = t[np.isfinite(t)]
                lim = 0.07 if n.startswith("tail_") else 0.04
                first = o + d / L * tf.min()
                tn = tf[np.argmin(np.abs(tf - L))]                  # ray/surface hit nearest p
                if np.linalg.norm(first - p) < lim and tf.min() <= L + 0.005:
                    q, method = first, "camera_ray_first_hit"
                elif abs(tn - L) < 0.03:
                    q, method = o + d / L * tn, "camera_ray_hit_nearest_point(occluded)"
        lm3d[n] = q
        lm_info[n] = {"model_point": p.tolist(), "method": method,
                      **({"tail_t_of_junction": tail_base_t} if n == "tail_base" else {}),
                      "moved_mm": float(np.linalg.norm(q - p) * 1000)}
    # part labels + special spheres
    sp = np.array([lm3d["eye_R"], lm3d["eye_L"], lm3d["nose"], lm3d["chin"]])
    np.savez_compressed(STAGING / "part_labels.npz", origin=origin, spacing=np.float64(h), labels=labels,
                        part_names=np.array(names), part_regions=np.array([PART_REGION[n] for n in names]),
                        special_points=sp, special_radius=np.array([EYE_RADIUS, EYE_RADIUS, NOSE_RADIUS, CHIN_RADIUS]),
                        special_regions=np.array(["eyes", "eyes", "nose", "head"]))
    from catlib import proxy as px
    from catlib import meshio
    px.clear_cache()
    cen = verts[faces].mean(1)
    reg = px.region_of(cen, path=STAGING / "part_labels.npz")
    rid = np.array([meshio.REGIONS.index(r) for r in reg], np.int32)
    np.savez_compressed(STAGING / "proxy_mesh.npz", vertices=verts.astype(np.float64),
                        triangles=faces.astype(np.int32), tri_region_id=rid,
                        region_names=np.array(meshio.REGIONS))
    (STAGING / "landmarks3d.json").write_text(json.dumps({
        "note": "16 landmarks on the fused proxy surface. tail_base: tail centreline point where it "
                "leaves the torso part (tail/rump junction), then as below. ear_tip_*: apex vertex of the fused ear "
                "(farthest along the ear axis). paw_*: closest surface point to the fitted sole "
                "centre. Others: first camera-ray hit through the fitted model point (the visible "
                "surface of that feature, keeps its reference-view projection) when the hit is within "
                "40 mm (70 mm for tail_*: the tail root is seen through the rump flank) and not behind the point; otherwise the ray/surface hit nearest "
                "the point if within 30 mm (feature hidden behind nearer fur in the reference view, "
                "e.g. the tail root behind the rump flank; flagged occluded by the metric); otherwise "
                "the closest surface point.",
        "landmarks": {k: v.tolist() for k, v in lm3d.items()}, "details": lm_info}, indent=1))
    # verification
    topo = geom.topology_report(verts, faces)
    Vp, Tp, _, _ = F.model_mesh(skel, hi=True)
    m_parts = raster.silhouette(Vp, Tp, cam)
    m_fused = raster.silhouette(verts, faces, cam)
    iou = float((m_parts & m_fused).sum() / (m_parts | m_fused).sum())
    ref = F.Ref(512).full
    iou_ref_fused = float((ref & m_fused).sum() / (ref | m_fused).sum())
    iou_ref_parts = float((ref & m_parts).sum() / (ref | m_parts).sum())
    sdf_chk = px.sdf_lookup(verts[:: max(1, len(verts) // 5000)], path=STAGING / "sdf.npz")
    rep = {"spacing_m": h, "grid_shape": list(shape), "marching_cubes_triangles": raw_tris,
           "decimation": "Blender Decimate COLLAPSE (CPU, background) to ~%d" % TARGET_TRIS,
           "triangles": int(len(faces)),
           "vertices": int(len(verts)), "removed_small_component_triangles": removed,
           "watertight_closed_manifold": bool(topo["closed_manifold"]),
           "boundary_edges": topo["boundary_edges"], "non_manifold_edges": topo["non_manifold_edges"],
           "components": topo["components"], "inconsistent_winding_edges": topo["inconsistent_winding_edges"],
           "degenerate_triangles": len(topo["degenerate_triangles"]),
           "signed_volume_m3": topo["signed_volume_m3"], "min_z": topo["min_z"],
           "bbox_min": topo["bbox_min"], "bbox_max": topo["bbox_max"],
           "bbox_center": ((np.array(topo["bbox_min"]) + topo["bbox_max"]) / 2).tolist(),
           "fused_vs_unfused_parts_silhouette_iou_1536": iou,
           "raster_iou_vs_reference_fused": iou_ref_fused, "raster_iou_vs_reference_parts": iou_ref_parts,
           "sdf_at_mesh_vertices_abs_mean_mm": float(np.abs(sdf_chk).mean() * 1000),
           "sdf_at_mesh_vertices_abs_max_mm": float(np.abs(sdf_chk).max() * 1000),
           "union_order_blend_m": UNION_ORDER, "sdf_blur_sigma_voxels": SDF_BLUR_VOXELS,
           "unblurred_parts": UNBLURRED_PARTS, "landmarks": lm_info,
           "region_triangle_counts": {r: int((rid == i).sum()) for i, r in enumerate(meshio.REGIONS)},
           "seconds": round(time.time() - t0, 1)}
    (STAGING / "build_report.json").write_text(json.dumps(rep, indent=1))
    import os
    for name in STAGED_FILES:                              # atomic per file, all in one go
        os.replace(STAGING / name, PROXY / name)
    px.clear_cache()
    print(json.dumps({k: rep[k] for k in ("triangles", "watertight_closed_manifold", "components",
                                          "min_z", "fused_vs_unfused_parts_silhouette_iou_1536",
                                          "raster_iou_vs_reference_fused", "bbox_min", "bbox_max",
                                          "region_triangle_counts", "seconds")}, indent=None))
    return rep


def previews(size=512):
    """Shaded numpy z-buffer renders through the fitted camera + the 8 novel views."""
    np = _np()
    from PIL import Image, ImageDraw
    from catlib import camera as camlib, geom, raster
    sys.path.insert(0, str(_paths.SCRIPTS))
    import fit_camera_proxy as F
    z = np.load(PROXY / "proxy_mesh.npz")
    V, T = z["vertices"], z["triangles"].astype(np.int64)
    P = json.loads((PROXY / "fit_params.json").read_text())["params"]
    ref = F.camera_from(P)
    rep = json.loads((PROXY / "build_report.json").read_text())
    scene = camlib.load_scene()
    scene = dict(scene, orbit_target=rep["bbox_center"])
    cams = {"reference": ref.scaled(size)}
    cams.update(camlib.novel_view_cameras(ref, scene, size=(size, size)))
    N = geom.tri_normals(V, T)
    out = PROXY / "previews"
    out.mkdir(exist_ok=True)
    tiles = []
    for name, c in cams.items():
        ids, _ = raster.id_buffer(V, T, c)
        img = np.full((size, size, 3), 200, np.uint8)
        m = ids >= 0
        n = N[ids[m]]
        cen = V[T[ids[m]]].mean(1)
        view = c.location - cen
        view /= np.linalg.norm(view, axis=1, keepdims=True)
        key = view + np.array([0.0, 0.0, 0.8])
        key /= np.linalg.norm(key, axis=1, keepdims=True)
        lam = np.clip((n * key).sum(1), 0, 1)
        shade = (40 + 190 * lam).astype(np.uint8)
        img[m] = shade[:, None]
        im = Image.fromarray(img)
        ImageDraw.Draw(im).text((8, 8), name, fill=(200, 0, 0))
        im.save(out / f"view_{name}.png")
        tiles.append(im)
    W = 3
    sheet = Image.new("RGB", (W * size, math.ceil(len(tiles) / W) * size), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % W) * size, (i // W) * size))
    sheet.save(out / "contact_sheet.png")
    print(f"[build_proxy] previews -> {out}")


def lock():
    np = _np()
    from catlib import camera as camlib
    sys.path.insert(0, str(_paths.SCRIPTS))
    import fit_camera_proxy as F
    fit = json.loads((PROXY / "fit_params.json").read_text())
    P = fit["params"]
    rep = json.loads((PROXY / "build_report.json").read_text())
    cam = F.camera_from(P)
    loc = cam.location
    look = np.array([P["cam_lx"], P["cam_ly"], P["cam_lz"]])
    cpath = ROOT / "contract" / "camera.json"
    d = json.loads(cpath.read_text())
    d.pop("note", None)
    d["status"] = "locked"
    d["location"] = [round(float(x), 6) for x in loc]
    d["look_at"] = [round(float(x), 6) for x in look]
    d["roll_deg"] = 0.0
    mw = camlib.look_at_matrix(d["location"], d["look_at"], 0.0)
    d["matrix_world"] = [[round(float(x), 9) for x in row] for row in mw]
    d["fit_summary"] = {
        "method": "joint CMA-ES fit of camera (look_at + azimuth/elevation/distance) and the "
                  "parametric proxy cat to reference/derived/mask.png + landmarks.json "
                  "(scripts/fit_camera_proxy.py); gauge: withers 0.25 m; 85 mm, roll 0 fixed",
        "azimuth_deg": round(P["cam_az"], 3), "elevation_deg": round(P["cam_el"], 3),
        "distance_to_look_at_m": round(P["cam_dist"], 4),
        "camera_height_m": round(float(loc[2]), 4),
        "proxy_raster_iou": rep["raster_iou_vs_reference_fused"],
        "details": "proxy/fit_report.json"}
    cpath.write_text(json.dumps(d, indent=2))
    chk = camlib.Camera.from_json(cpath).check_stored_matrix()
    if not chk["ok"]:
        raise SystemExit(f"camera.json matrix check failed: {chk}")
    spath = ROOT / "contract" / "scene.json"
    s = json.loads(spath.read_text())                         # load-modify-save, keep other keys
    s["orbit_target"] = [round(float(x), 6) for x in rep["bbox_center"]]
    spath.write_text(json.dumps(s, indent=2))
    print(f"[build_proxy] locked camera: location {d['location']} look_at {d['look_at']}  check {chk}")
    print(f"[build_proxy] scene.json orbit_target = {s['orbit_target']}")


def report():
    np = _np()
    from catlib import camera as camlib, meshio, metrics, paths
    from catlib.overlay import make_overlay
    sys.path.insert(0, str(_paths.SCRIPTS))
    import fit_camera_proxy as F
    cam = camlib.Camera.from_json()
    z = np.load(PROXY / "proxy_mesh.npz")
    lm = json.loads((PROXY / "landmarks3d.json").read_text())["landmarks"]
    mesh = meshio.CatMesh({"vertices": z["vertices"].tolist(), "triangles": z["triangles"].tolist(),
                           "landmarks": lm, "name": "cat_proxy"})
    sil = PROXY / "renders" / "ref_silhouette.png"
    rendered = metrics.load_mask(sil) if sil.is_file() else None
    m, M = metrics.compute_metrics(mesh, cam, paths.derived_paths(), rendered_mask=rendered,
                                   proxy_path=None)
    m.pop("secondary", None)
    regions = metrics.load_regions(paths.REGIONS_JSON)
    (PROXY / "renders").mkdir(exist_ok=True)
    make_overlay(PROXY / "renders" / "ref_overlay.png", M, m, regions,
                 ref_mask=metrics.load_mask(paths.MASK_PNG), title="proxy (inferred) through locked camera")
    fit = json.loads((PROXY / "fit_params.json").read_text())
    build = json.loads((PROXY / "build_report.json").read_text())
    trace = json.loads((PROXY / "fit_trace.json").read_text())
    out = {
        "what": "Inferred high-resolution proxy cat scored through the locked camera with the same "
                "catlib.metrics functions used for the budget variants. NOT a deliverable mesh.",
        "camera_status": cam.status, "camera": cam.meta["json"].get("fit_summary"),
        "camera_location": cam.meta["json"]["location"], "camera_look_at": cam.meta["json"]["look_at"],
        "metrics": m,
        "proxy_mesh": {k: build[k] for k in ("triangles", "vertices", "watertight_closed_manifold",
                                              "components", "min_z", "bbox_min", "bbox_max",
                                              "bbox_center", "fused_vs_unfused_parts_silhouette_iou_1536",
                                              "region_triangle_counts", "spacing_m")},
        "fitted_parameters": fit["params"],
        "parameter_spec_init_scale_lo_hi": fit["spec"],
        "priors": fit["priors_doc"],
        "final_prior_penalties": fit["final_internal_scores_512"]["priors"],
        "objective_trace_summary": [{"stage": t["stage"], "title": t["title"], "n_params": t["n_params"],
                                     "evals": t["evals"], "seconds": t["seconds"],
                                     "start_objective": t.get("start_objective"),
                                     "best_objective": t["iters"][-1]["best"] if t["iters"] else None}
                                    for t in trace],
        "assumptions": [
            "85 mm focal length on a 36 mm sensor (decision, not measured); roll 0; principal point centred",
            "scale gauge: withers (torso top above the shoulder joints) = 0.25 m",
            "body sagittal plane y = 0, nose +X, paw soles flat on z = 0",
            "left/right legs share thickness parameters exactly; left/right segment lengths tied by prior",
        ],
        "inferred_not_observed": [
            "everything on the cat's left side (left flank, far side of head, inner surfaces of the left ear)",
            "torso width / cross-section (only height and outline are observed): width set by the 55-75 % "
            "of height prior and the superellipse exponent",
            "belly underside and chest depth seen from below",
            "head depth (back-of-skull to muzzle) and skull/cheek widths beyond the visible face outline",
            "stance width (paw y) of all four legs: 8-11 cm prior; lateral positions of hips/shoulders",
            "left-leg thicknesses (copied from the right legs); far hind leg (HL) upper part hidden by the belly",
            "ear thickness (3-4.5 mm semi-axis, bounded), ear cup shape (pinnae are solid flattened cones)",
            "tail lateral (y) deviation: kept near the sagittal plane by prior",
            "top-of-back shape between the observed outline points (smoothness prior)",
            "smooth-union fillets between parts (blend radii in build_report.json)",
        ],
        "sources": {"skeleton": "proxy/skeleton.json", "params": "proxy/fit_params.json",
                    "trace": "proxy/fit_trace.json", "build": "proxy/build_report.json",
                    "overlay": "proxy/renders/ref_overlay.png", "stage_overlays": "proxy/fit_stages/"},
    }
    tmp = PROXY / ".fit_report.json.tmp"
    tmp.write_text(json.dumps(out, indent=1))
    import os
    os.replace(tmp, PROXY / "fit_report.json")
    keys = ("official", "iou", "ref_uncovered_frac", "render_outside_frac", "boundary_mean_px",
            "boundary_hd95_px", "region_iou", "legs_gap_iou", "landmark_sharp_mean_px",
            "landmark_sharp_max_px", "landmark_fuzzy_mean_px", "raster_vs_blender_iou", "landmark_flags")
    print(json.dumps({k: m.get(k) for k in keys}, indent=1))
    print({n: round(r["error_px"], 1) for n, r in m["landmarks"].items()})


def main():
    if "--write-blend" in sys.argv:
        write_blend(int(sys.argv[sys.argv.index("--write-blend") + 1]))
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spacing", type=float, default=0.002)
    ap.add_argument("--lock", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--previews-only", action="store_true")
    ap.add_argument("--sdf-only", action="store_true",
                    help="rebuild proxy/sdf.npz from skeleton.json + fit_params.json and touch nothing else "
                         "(CPU only; for a case that ships proxy_mesh.npz but not the 15 MB SDF)")
    a = ap.parse_args()
    if a.lock:
        lock()
        return
    if a.report:
        report()
        return
    if a.sdf_only:
        build(a.spacing, sdf_only=True)
        return
    if not a.previews_only:
        build(a.spacing)
    previews()


if __name__ == "__main__":
    main()
