"""Throw-away CONTRACT-section-6-style test cat for the render tooling (CPU only, no lock).

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        --python tests/make_test_blend.py -- [--out .cache/render_test/cat_test.blend]

One collection CAT_TEST with one all-triangle mesh object cat_test (identity transform, flat
shaded, several MATERIALS slots): box body, four triangular-prism legs with real gaps, a box
head with two ear pyramids and a nose, a two-segment tail. Nose towards +X, ~0.5 m long,
paw soles on z = 0. Prints the triangle count.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/
import _setup  # noqa: E402  (skill scripts on sys.path; LOWPOLY_CASE defaults to examples/cat)
REPO = _setup.REPO
ROOT = _setup.CASE
MATS = ["fur_orange", "fur_cream", "fur_stripe", "ear_inner", "nose", "eye"]


def add_box(bm, lo, hi, mat):
    c = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2]
    s = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]
    M = Matrix.Translation(c) @ Matrix.Diagonal((*s, 1.0))
    r = bmesh.ops.create_cube(bm, size=1.0, matrix=M)
    return r["verts"], mat


def add_prism(bm, base, top, radius, mat, segments=3, rot=0.0):
    """Capped prism with `segments` sides from point base to point top."""
    base, top = Vector(base), Vector(top)
    d = top - base
    q = d.to_track_quat("Z", "Y")
    M = Matrix.Translation((base + top) / 2) @ q.to_matrix().to_4x4() @ Matrix.Rotation(rot, 4, "Z")
    r = bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments, radius1=radius,
                              radius2=radius, depth=d.length, matrix=M)
    return r["verts"], mat


def add_pyramid(bm, base_pts, apex, mat):
    vs = [bm.verts.new(p) for p in base_pts] + [bm.verts.new(apex)]
    n = len(base_pts)
    faces = [bm.faces.new([vs[i], vs[(i + 1) % n], vs[n]]) for i in range(n)]
    faces.append(bm.faces.new(list(reversed(vs[:n]))))
    return vs, mat


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / ".cache" / "render_test" / "cat_test.blend"))
    a = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bm = bmesh.new()
    parts = []
    parts.append(add_box(bm, (-0.17, -0.065, 0.12), (0.12, 0.065, 0.25), "fur_orange"))      # body
    for x in (0.09, -0.14):                                                                 # legs
        for y in (0.04, -0.04):
            parts.append(add_prism(bm, (x, y, 0.0), (x, y, 0.135), 0.024, "fur_cream", 3, 0.3))
    parts.append(add_box(bm, (0.10, -0.055, 0.22), (0.22, 0.055, 0.33), "fur_orange"))      # head
    for y in (0.03, -0.03):                                                                 # ears
        parts.append(add_pyramid(bm, [(0.13, y - 0.022, 0.329), (0.19, y - 0.022, 0.329), (0.16, y + 0.022, 0.329)],
                                 (0.16, y, 0.39), "ear_inner"))
    parts.append(add_pyramid(bm, [(0.219, -0.015, 0.25), (0.219, 0.015, 0.25), (0.219, 0.0, 0.275)],
                             (0.25, 0.0, 0.258), "nose"))                                   # nose
    for y in (0.025, -0.025):                                                               # eyes
        parts.append(add_pyramid(bm, [(0.219, y - 0.01, 0.29), (0.219, y + 0.01, 0.29), (0.219, y, 0.305)],
                                 (0.232, y, 0.295), "eye"))
    parts.append(add_prism(bm, (-0.165, 0.0, 0.22), (-0.24, 0.0, 0.30), 0.016, "fur_stripe", 4))  # tail
    parts.append(add_prism(bm, (-0.24, 0.0, 0.30), (-0.26, 0.0, 0.42), 0.014, "fur_stripe", 4))
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.verts.ensure_lookup_table()
    me = bpy.data.meshes.new("cat_test")
    # material index per face via the part each face's verts came from
    owner = {}
    for vs, mat in parts:
        for v in vs:
            owner[v] = MATS.index(mat)
    face_mat = [owner[f.verts[0]] for f in bm.faces]
    bm.to_mesh(me)
    bm.free()
    for m in MATS:
        mat = bpy.data.materials.new(m)
        me.materials.append(mat)
    me.polygons.foreach_set("material_index", face_mat)
    me.polygons.foreach_set("use_smooth", [False] * len(me.polygons))
    me.update()
    assert all(len(p.vertices) == 3 for p in me.polygons)
    ob = bpy.data.objects.new("cat_test", me)
    col = bpy.data.collections.new("CAT_TEST")
    bpy.context.scene.collection.children.link(col)
    col.objects.link(ob)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    zs = [v.co.z for v in me.vertices]
    print(f"[make_test_blend] wrote {out}: {len(me.polygons)} triangles, z [{min(zs):.3f}, {max(zs):.3f}]")


if __name__ == "__main__":
    main()
