"""Blender-side reference checks for the tooling self-test (CPU only, no rendering).

    Blender -b --factory-startup --python tests/blender_checks.py -- --mesh <mesh.json> --out <dir>

1. Projection: catlib.camera.project vs bpy_extras.object_utils.world_to_camera_view on
   2000 random points, for the reference camera and every fixed novel view.
2. Silhouette: BVH ray cast through every pixel centre (inside the projected bbox) at
   1536 px using Blender's own camera frame (Camera.view_frame), saved as raycast_mask.npy
   so the venv side can compare it with catlib.raster.silhouette (pixel-centre semantics).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/
import _setup  # noqa: E402  (skill scripts on sys.path; LOWPOLY_CASE defaults to examples/cat)
REPO = _setup.REPO
ROOT = _setup.CASE

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from bpy_extras.object_utils import world_to_camera_view  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402

from catlib import meshio  # noqa: E402
from catlib.camera import Camera, novel_view_cameras  # noqa: E402


def make_cam(scene, c: Camera):
    cd = bpy.data.cameras.new("c")
    cd.lens = c.focal_length_mm
    cd.sensor_width = c.sensor_width_mm
    cd.sensor_fit = "HORIZONTAL"
    ob = bpy.data.objects.new("c", cd)
    scene.collection.objects.link(ob)
    ob.matrix_world = Matrix(c.matrix_world.tolist())
    scene.camera = ob
    scene.render.resolution_x, scene.render.resolution_y = c.width, c.height
    scene.render.resolution_percentage = 100
    bpy.context.view_layer.update()
    return ob


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    ref = Camera.from_json()
    res = {"projection": {}}
    rng = np.random.default_rng(0)
    cams = {"reference": ref, **novel_view_cameras(ref)}
    for name, c in cams.items():
        ob = make_cam(scene, c)
        P = rng.uniform([-0.45, -0.2, 0.0], [0.45, 0.2, 0.5], (2000, 3))
        u, v, d = c.project(P)
        bu, bv, bd = [], [], []
        for p in P:
            q = world_to_camera_view(scene, ob, Vector(p))
            bu.append(q.x * c.width)
            bv.append((1.0 - q.y) * c.height)
            bd.append(q.z)
        err = np.hypot(u - np.array(bu), v - np.array(bv))
        res["projection"][name] = {"max_err_px": float(err.max()), "mean_err_px": float(err.mean()),
                                   "max_depth_err_m": float(np.abs(d - np.array(bd)).max()),
                                   "size": [c.width, c.height]}
        bpy.data.objects.remove(ob)

    # ---- ray-cast silhouette at full resolution through Blender's camera frame
    m = meshio.load(a.mesh)
    ob = make_cam(scene, ref)
    frame = [ob.matrix_world @ v for v in ob.data.view_frame(scene=scene)]
    # view_frame order: top-right, bottom-right, bottom-left, top-left (camera space)
    tr, br, bl, tl = [np.array(v) for v in frame]
    origin = np.array(ob.matrix_world.translation)
    W, H = ref.width, ref.height
    tree = BVHTree.FromPolygons([tuple(v) for v in m.vertices], [tuple(t) for t in m.triangles],
                                all_triangles=True)
    uv = ref.project_uv(m.vertices)
    c0, c1 = max(int(uv[:, 0].min()) - 3, 0), min(int(uv[:, 0].max()) + 4, W)
    r0, r1 = max(int(uv[:, 1].min()) - 3, 0), min(int(uv[:, 1].max()) + 4, H)
    mask = np.zeros((H, W), bool)
    for j in range(r0, r1):
        fy = (j + 0.5) / H
        left = tl + (bl - tl) * fy
        right = tr + (br - tr) * fy
        for i in range(c0, c1):
            p = left + (right - left) * ((i + 0.5) / W)
            dvec = Vector(p - origin).normalized()
            if tree.ray_cast(Vector(origin), dvec)[0] is not None:
                mask[j, i] = True
    np.save(a.out / "raycast_mask.npy", mask)
    res["raycast_pixels"] = int(mask.sum())
    (a.out / "blender_checks.json").write_text(json.dumps(res, indent=1))
    print("BLENDER_CHECKS_OK")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
