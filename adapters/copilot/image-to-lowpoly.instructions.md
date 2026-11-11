---
applyTo: "**/*.py"
---

Copy this file into your repository's `.github/instructions/`. For agent mode, also install the skill with
`gh skill install gokayfem/image-to-lowpoly image-to-lowpoly`.

- Build low-poly meshes of a photo with the `image-to-lowpoly` skill: `preflight_reference.py` →
  `fit_camera_proxy.py` → `build_proxy.py --lock` → `make_decimation_baseline.py` → per budget: build script →
  `refine_vertices.py` → `evaluate.py --views` → blind review → `finalize_variant.py` → `aggregate_metrics.py`.
- The triangle count is the evaluated, exported geometry, counted four ways; never Blender's polygon count.
- No textures, cards, billboards or padding stand in for shape.
