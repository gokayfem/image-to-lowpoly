"""Blender (background, CPU only, NO rendering -> no GPU lock) part of variants/1000/fuse.py.

    Blender -b --factory-startup --python variants/1000/fuse_blender.py -- in_parts.json out_fused.json [merge_m]

Boolean-unions the closed part lofts into one surface (EXACT solver, sequentially into the
torso), merges vertices closer than merge_m (default 1.5 mm), dissolves degenerates,
triangulates, and writes {"vertices", "triangles"}.
"""
import json
import sys

import bmesh
import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]
merge = float(argv[2]) if len(argv) > 2 else 0.0015
data = json.load(open(src))

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)


def make(name, V, T):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in V], [], [tuple(t) for t in T])
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


objs = {k: make(k, p["V"], p["T"]) for k, p in data["parts"].items()}
base = objs[data["order"][0]]
bpy.context.view_layer.objects.active = base
for k in data["order"][1:]:
    m = base.modifiers.new("u_" + k, "BOOLEAN")
    m.operation = "UNION"
    m.solver = "EXACT"
    m.object = objs[k]
    bpy.ops.object.modifier_apply(modifier=m.name)
    print("union", k, len(base.data.polygons))

bm = bmesh.new()
bm.from_mesh(base.data)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge)
bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=0.001)
bmesh.ops.triangulate(bm, faces=bm.faces)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge)
bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=0.001)
bmesh.ops.triangulate(bm, faces=bm.faces)
loose = [v for v in bm.verts if not v.link_faces]
bmesh.ops.delete(bm, geom=loose, context="VERTS")
bm.verts.index_update()
V = [list(v.co) for v in bm.verts]
T = [[v.index for v in f.verts] for f in bm.faces]
nonman = sum(1 for e in bm.edges if len(e.link_faces) != 2)
json.dump({"vertices": V, "triangles": T, "non_manifold_edges": nonman}, open(dst, "w"))
print("FUSED", len(V), len(T), "nonmanifold", nonman)
