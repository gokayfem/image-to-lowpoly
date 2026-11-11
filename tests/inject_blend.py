"""Tamper with an assembled variant .blend to prove the validator catches it (test only).

    Blender -b --factory-startup --python tests/inject_blend.py -- --blend X.blend --mode MODE

MODE: hidden_object | quad_trap | modifier | orphan_mesh
"""
import argparse
import sys

import bmesh
import bpy


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--mode", required=True)
    a = ap.parse_args(argv)
    bpy.ops.wm.open_mainfile(filepath=a.blend)
    cat = [o for o in bpy.data.objects if o.type == "MESH"][0]
    if a.mode == "hidden_object":
        me = bpy.data.meshes.new("helper")
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=0.05)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new("helper", me)
        coll = bpy.data.collections.new("hidden_stuff")
        bpy.context.scene.collection.children.link(coll)
        coll.objects.link(ob)
        ob.hide_viewport = True
        ob.hide_render = True
        ob.hide_set(True)
    elif a.mode == "quad_trap":
        bm = bmesh.new()
        bm.from_mesh(cat.data)
        bmesh.ops.join_triangles(bm, faces=bm.faces[:], angle_face_threshold=3.1416,
                                 angle_shape_threshold=3.1416)
        bm.to_mesh(cat.data)
        bm.free()
    elif a.mode == "modifier":
        m = cat.modifiers.new("sub", "SUBSURF")
        m.levels = 1
    elif a.mode == "orphan_mesh":
        me = cat.data.copy()
        me.name = "hidden_copy"
        me.use_fake_user = True
    else:
        raise SystemExit(f"unknown mode {a.mode}")
    bpy.ops.wm.save_as_mainfile(filepath=a.blend)
    print("INJECT_OK", a.mode, len(cat.data.polygons))


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:])
