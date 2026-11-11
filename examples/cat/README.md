# Worked case: one cat, four budgets

The run the skill was built from, laid out as a case directory, so every script works on it with
`--case examples/cat`. One reference photo of an orange tabby (MiniMax H3 Image on fal, generated once, never
changed), budgets 100, 200, 500 and 1000 triangles. Mesh iteration was stopped by the user; the finals are the
best validated checkpoints at that moment. [`REPORT.md`](REPORT.md) is the full run report.

| Path | What |
|---|---|
| `reference/cat.png`, `PROVENANCE.json`, `prompt.txt` | the photo (SHA-256 `b2e4800a…ad16`), how it was generated |
| `reference/derived/` | mask, landmarks, image windows, bbox, and the manual annotations they came from |
| `contract/camera.json`, `scene.json`, `FROZEN.json` | the locked camera, the render scene, the hashes frozen before scoring |
| `proxy/` | the fitted parametric cat (`skeleton.json`, `fit_params.json`), its fused 60,000-triangle surface (`proxy_mesh.npz`), part labels, 3D landmarks, fit and build reports |
| `head/` | the shared symmetric head kit at five levels of detail (22-224 triangles), depth-aligned |
| `variants/<N>/final/` | `mesh.json`, `cat_<N>.glb`, `metrics.json`, `validation.json`, `CHOSEN.json` |
| `variants/<N>/NOTES.md`, `history.json` | each worker's notes: allocations tried, every official checkpoint, what failed |
| `variants/<N>/*.py`, `100/reproduce_v3.sh` | the build scripts of the accepted checkpoints |
| `metrics/metrics.json`, `metrics.csv` | finals, initial builds, both decimation controls, the proxy, four-way counts, topology |
| `metrics/allocation.csv` | triangles per region per budget |
| `critic/round_1_CRITIQUE.md`, `round_2_CRITIQUE.md` | the two blind reviews (round 2 is of the finals) |

| Budget | Final checkpoint | Rebuild (construction is deterministic; refinement is seeded) |
|---|---|---|
| 100 | `v3_welded` | `sh variants/100/reproduce_v3.sh` (build, four refine runs with unfold passes, evaluate) |
| 200 | `v5u_untangled` | `variants/200/build.py --alloc W`, `refine_vertices.py --iters 20000 --flips --seed 0`, `variants/200/untangle.py` |
| 500 | `v3_nose` | `variants/500/build.py --config D`, `refine_vertices.py --iters 40000 --flips --seed 1 --freeze-verts <freeze_verts.json from the build>` |
| 1000 | `v7_body` | `variants/1000/fuse.py --design f3 --paw-align` (Blender boolean), three refine runs and `fix_selfint.py --alpha 0.2`; see the end of `variants/1000/NOTES.md` |

Exact commands are in each `NOTES.md`; they were written for the original project, where the shared scripts
lived in `scripts/` and the analysis Python was `.venv/bin/python`. Here, run the budget scripts from this
directory and the shared ones as `python3 ../../skills/image-to-lowpoly/scripts/<name>.py --case .`.

Before refining, rebuild the proxy's signed distance field (15 MB, not shipped):

```bash
python3 ../../skills/image-to-lowpoly/scripts/build_proxy.py --case . --sdf-only
```

It reproduces the original `sdf.npz` exactly (checked: maximum difference 0). The construction stages of
100, 200 and 500 were rebuilt from this layout and match the original initial meshes byte for byte; the
1000 construction (Blender boolean) and the refinement runs were not re-run here.

Notes on what is here:

- The 200 worker's `NOTES.md` ends with "Pick: v8_kitL2" (a kit-head checkpoint). The accepted final is
  `v5u_untangled`, chosen when the user stopped iteration; `CHOSEN.json` records why.
- Absolute paths of the original machine were removed from the JSON files (made relative) and from one
  string in each `.glb`'s metadata; mesh data and counts are unchanged, and each `mesh.json` still matches the
  hash recorded in its `metrics.json`.
- Not shipped: `.blend` files, renders, turntables, intermediate checkpoints, the release film, the proxy's
  `.blend` and SDF.
- The meshes carry flat per-face materials (fur, cream, inner ear, eye, pupil, nose). They are not shown in
  the gallery and play no part in any score.
