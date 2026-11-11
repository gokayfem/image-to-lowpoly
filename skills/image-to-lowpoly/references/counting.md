# Counting triangles

"500 triangles" means 500 triangles in the geometry someone opens or exports, after every modifier and after
triangulation, summed over every object in the file. Blender's polygon count is not that number: a quad is
two triangles, an n-gon is n-2, a Subdivision, Mirror, Array or Solidify modifier multiplies them, and a
hidden second object adds its own.

## The mesh format

A mesh is `mesh.json` (`catmesh-1`, `catlib/meshio.py`): `vertices` in world metres, `triangles` only
(counter-clockwise seen from outside), one `tri_region` and one `tri_material` per triangle, landmark
positions and their attachments. There are no quads, so nothing hides behind triangulation, and
`len(triangles)` is the count.

`scripts/assemble_variant.py` turns it into a `.blend` with exactly one collection, one mesh object with
identity transform, flat shading, one material slot per used material and no modifiers, and a `.glb` of that
one object (no Draco, no cameras, no lights).

## Four independent counts

All four must agree and be at most the budget. `scripts/validate_variant.py` takes them and
`scripts/evaluate.py` refuses a mesh where they do not.

| | How | Catches |
|---|---|---|
| A | `len(triangles)` in `mesh.json` | the design count |
| B | reopen the `.blend` in Blender; sum `calc_loop_triangles()` over the **evaluated** mesh of **every** object (any type, any collection, hidden or not) | modifiers, extra or hidden objects, quads |
| C | import the `.glb` into an empty scene; sum loop triangles over every imported object | export-side changes |
| D | parse the GLB container in plain Python: `indices.count / 3` per primitive, times the number of nodes using that mesh; any primitive not in TRIANGLES mode is an error | instancing, strips, fans, a lying importer |

In the cat run every final had A = B = C = D = N (100, 200, 500, 1000).

## Hard failures

The mesh is invalid until fixed if any of these hold:

- any count disagrees with another, or exceeds the budget;
- anything else in the `.blend` (a second object, a camera, a light, a curve, a modifier, a shape key, a
  particle or hair system, an instance, a second mesh datablock, a wireframe helper);
- a degenerate triangle (area under 1 mm², or an edge shorter than 1 mm);
- a duplicate triangle (same vertex set, or coincident positions within 1 µm);
- geometry below the ground (min z under -2 mm);
- a billboard: the smallest bounding-box extent under 12 % of the largest.

The validator was tested against deliberate cheats before any mesh was scored (over budget, a duplicate
triangle, a degenerate triangle, a hidden extra object, quads hiding triangles, a Subdivision modifier, a
hidden duplicate mesh, a billboard); each one fires (`tests/test_tooling.py`, needs Blender).

## Disclosed, not fatal

Reported for every checkpoint and to be kept low and explained: more than one connected component; open
(boundary) edges; non-manifold edges; inconsistent winding; self-intersecting triangle pairs; triangles never
visible from any of 162 directions (wasted budget); interior vertices whose whole fan is flat (padding).

## Presentation is not geometry

The ground, lights, world, cameras and the wireframe overlay are built in memory by `render_variant.py` and
never saved into a variant file or exported. A wire render is an annotation, not geometry.
