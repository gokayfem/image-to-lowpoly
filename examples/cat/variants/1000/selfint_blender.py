"""Blender background (CPU only, no render): self-intersecting non-adjacent triangle pairs
of a mesh.json, same test as scripts/validate_variant.py (BVH overlap, shared-vertex
pairs skipped).  Blender -b --factory-startup --python selfint_blender.py -- mesh.json out.json"""
import json, sys
from mathutils.bvhtree import BVHTree
a = sys.argv[sys.argv.index("--") + 1:]
d = json.load(open(a[0]))
V = [tuple(v) for v in d["vertices"]]; T = [tuple(t) for t in d["triangles"]]
tree = BVHTree.FromPolygons(V, T, all_triangles=True, epsilon=0.0)
out = sorted({(min(i, j), max(i, j)) for i, j in tree.overlap(tree) if i != j and not (set(T[i]) & set(T[j]))})
json.dump(out, open(a[1], "w"))
