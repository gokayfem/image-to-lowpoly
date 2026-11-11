"""Projection self-test inside Blender (renders, so it takes the GPU lock first).

    Blender -b --factory-startup --python tests/blender_projection_check.py -- \
        --blend .cache/render_test/cat_test.blend --out .cache/render_test/projection.json

1. For a handful of world points, compare CONTRACT section 3's formula (render_variant.project)
   with bpy_extras.object_utils.world_to_camera_view on the camera render_variant builds.
2. Render tiny emissive markers (pixel-centre point sampling, like the silhouette pass) at
   those points and compare the centroid of each blob with the formula.
3. Silhouette cross-check: a pure-numpy pixel-centre rasteriser of the cat triangles vs the
   rendered silhouette (IoU).
4. Novel views: for all eight scene.json views, the Blender camera the renderer actually builds
   (Presenter.view_list + set_view, read back through world_to_camera_view) vs
   catlib.camera.novel_view_cameras, on projected points (< 0.5 px at view resolution).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/
import _setup  # noqa: E402  (skill scripts on sys.path; LOWPOLY_CASE defaults to examples/cat)
REPO = _setup.REPO
ROOT = _setup.CASE
spec = importlib.util.spec_from_file_location("render_variant", _setup.SCRIPTS / "render_variant.py")
rv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv)

import bmesh  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from bpy_extras.object_utils import world_to_camera_view  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402


def raster_mask(V, T, M, f, W, H):
    """Pixel-centre coverage of projected triangles (independent numpy implementation)."""
    uvd = rv.project(M, f, W, H, V)
    mask = np.zeros((H, W), bool)
    for t in T:
        p = uvd[t, :2]
        if (uvd[t, 2] <= 0).any():
            continue
        u0, v0 = np.floor(p.min(0) - 0.5).astype(int)
        u1, v1 = np.ceil(p.max(0) + 0.5).astype(int)
        u0, v0, u1, v1 = max(u0, 0), max(v0, 0), min(u1, W), min(v1, H)
        if u1 <= u0 or v1 <= v0:
            continue
        uu, vv = np.meshgrid(np.arange(u0, u1) + 0.5, np.arange(v0, v1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = p
        e0 = (bx - ax) * (vv - ay) - (by - ay) * (uu - ax)
        e1 = (cx - bx) * (vv - by) - (cy - by) * (uu - bx)
        e2 = (ax - cx) * (vv - cy) - (ay - cy) * (uu - cx)
        inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
        mask[v0:v1, u0:u1] |= inside
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(rv.script_args())
    rv._load_gpulock().acquire_gpu_lock(label="projection_check")
    args = types.SimpleNamespace(blend=a.blend, camera=str(ROOT / "contract" / "camera.json"),
                                 scene=str(ROOT / "contract" / "scene.json"), samples=None, wire_ss=None, wire_px=None)
    p = rv.Presenter(args)
    p.build()
    W, H = p.ref_res
    f = float(p.cam_cfg["focal_length_mm"]) / float(p.cam_cfg["sensor_width_mm"]) * W
    p.set_view(p.ref_M, None, (W, H))
    bpy.context.view_layer.update()
    scene = p.scene

    # 1. formula vs world_to_camera_view, points spread over the frame
    # points around the camera's look_at, kept only if they project >= 40 px inside the frame
    la = np.asarray(p.cam_cfg["look_at"], float)
    cand = la + np.array([[0, 0, 0], [0.25, 0.05, 0.2], [-0.3, -0.1, -0.15], [0.2, -0.15, -0.15],
                          [-0.25, 0.12, 0.22], [0.1, 0.3, 0.1], [-0.15, -0.3, 0.05], [0.12, 0.1, -0.2],
                          [-0.1, 0.05, 0.12], [0.05, -0.05, 0.25]], float)
    cand[:, 2] = np.clip(cand[:, 2], 0.0, None)
    cf = rv.project(p.ref_M, f, W, H, cand)
    keep = (cf[:, 0] > 40) & (cf[:, 0] < W - 40) & (cf[:, 1] > 40) & (cf[:, 1] < H - 40) & (cf[:, 2] > 0)
    pts = cand[keep][:7]
    if len(pts) < 5:
        raise RuntimeError("fewer than 5 test points project inside the frame")
    form = rv.project(p.ref_M, f, W, H, pts)
    w2cv = []
    for q in pts:
        c = world_to_camera_view(scene, p.cam, Vector(q))
        w2cv.append([c.x * W, (1 - c.y) * H])
    w2cv = np.array(w2cv)
    d1 = np.linalg.norm(form[:, :2] - w2cv, axis=1)

    # 2. render emissive markers at the points (cat hidden), find blob centroids
    p.set_mode("silhouette")
    p.cat.hide_render = True
    bm = bmesh.new()
    for q in pts:
        r = 0.004 * form[0, 2] / 2.5   # ~6 px radius
        bmesh.ops.create_icosphere(bm, subdivisions=3, radius=0.004, matrix=Matrix.Translation(Vector(q)))
    me = bpy.data.meshes.new("__markers")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(p.m_white)
    mk = bpy.data.objects.new("__markers", me)
    p.coll.objects.link(mk)
    img = p.render_exr("markers")[:, :, 0] > 0.5
    cents = []
    for (u, v, _d) in form:
        u0, v0 = int(u) - 15, int(v) - 15
        win = img[max(v0, 0):v0 + 31, max(u0, 0):u0 + 31]
        ys, xs = np.nonzero(win)
        cents.append([xs.mean() + max(u0, 0) + 0.5, ys.mean() + max(v0, 0) + 0.5] if len(xs) else [np.nan, np.nan])
    cents = np.array(cents)
    d2 = np.linalg.norm(cents - form[:, :2], axis=1)

    # 3. silhouette vs numpy rasteriser
    mk.hide_render = True
    p.cat.hide_render = False
    p.set_mode("id")
    ids = p.render_ids(None, "ids_xcheck")
    ras = raster_mask(p.V, p.T, p.ref_M, f, W, H)
    ren = ids > 0
    iou = float((ras & ren).sum() / max(1, (ras | ren).sum()))
    # 4. renderer novel-view cameras vs catlib.camera.novel_view_cameras
    from catlib.camera import Camera, novel_view_cameras
    lib = novel_view_cameras(Camera.from_json(args.camera), json.loads(Path(args.scene).read_text()))
    vres = tuple(p.scn_cfg["view_resolution"])
    rng = np.random.default_rng(0)
    vp = np.asarray(p.orbit_target) + rng.uniform(-0.35, 0.35, (200, 3))
    vp[:, 2] = np.clip(vp[:, 2], 0.0, None)
    views = {}
    for name, M, az in p.view_list():
        p.set_view(M, az, vres)
        bpy.context.view_layer.update()
        bl = np.array([[c.x * vres[0], (1 - c.y) * vres[1]] for c in
                       (world_to_camera_view(scene, p.cam, Vector(q)) for q in vp)])
        cu = lib[name].project_uv(vp)
        views[name] = {"max_px": float(np.abs(bl - cu).max()),
                       "max_matrix_diff": float(np.abs(np.array(p.cam.matrix_world) - lib[name].matrix_world).max())}
    nv_max = max(v["max_px"] for v in views.values())
    res = {
        "novel_views_renderer_vs_catlib": views, "novel_views_max_px": nv_max,
        "orbit_radius_m": p.orbit_radius, "orbit_target": list(map(float, p.orbit_target)),
        "points": pts.tolist(), "formula_uv": form[:, :2].round(4).tolist(),
        "world_to_camera_view_uv": w2cv.round(4).tolist(),
        "formula_vs_w2cv_px": {"max": float(d1.max()), "mean": float(d1.mean())},
        "marker_centroid_uv": cents.round(3).tolist(),
        "formula_vs_marker_px": {"max": float(np.nanmax(d2)), "mean": float(np.nanmean(d2))},
        "raster_vs_render_iou": iou, "raster_px": int(ras.sum()), "render_px": int(ren.sum()),
        "xor_px": int((ras ^ ren).sum()),
        "pass": bool(d1.max() < 0.5 and np.nanmax(d2) < 0.5 and iou >= 0.995 and nv_max < 0.5
                     and len(views) == 8),
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("[projection_check]", json.dumps({k: res[k] for k in ("formula_vs_w2cv_px", "formula_vs_marker_px",
                                                               "raster_vs_render_iou", "xor_px", "novel_views_max_px", "pass")}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
