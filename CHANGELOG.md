# Changelog

## [0.1.0] — 2026-11-11

First version.

- An agent skill that turns one reference photo into editable low-poly meshes under hard triangle budgets:
  fixed, hashed targets (mask, landmarks, image windows), one camera fitted jointly with a parametric proxy
  and then locked, a build script per budget that welds every part into one closed surface, a shared seeded
  vertex optimiser that never changes the count, a four-way triangle count (mesh file, evaluated `.blend`,
  re-imported GLB, parsed GLB), metrics from a Blender render through the locked camera, eight fixed views
  and turntables in clay and wireframe, a decimation control per budget, a blind reviewer's packet, and a
  build timelapse drawn over the photo.
- Scripts take the case directory from `--case DIR` or `LOWPOLY_CASE`; `build_proxy.py --sdf-only`
  rebuilds the proxy's signed distance field without Blender.
- Worked case `examples/cat/`: one orange tabby photo (MiniMax H3 Image on fal, generated once), budgets 100,
  200, 500 and 1000, each final exactly on budget, with each budget's build scripts, notes, allocation,
  metrics against both controls, both blind reviews and the run report. Mesh iteration was stopped by the
  user; the heads are unfinished (see the README).
- `tests/smoke.py` (Python only); the original Blender test suites adapted to this layout but not re-run here.
