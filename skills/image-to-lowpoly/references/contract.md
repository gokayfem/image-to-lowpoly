# Case contract: one subject, one camera, one evaluation procedure

This is the contract of the worked cat case, kept nearly as it was written for that run, so the rules can be
read with the evidence behind them. It is cat-shaped in its names: the mesh object is `cat_<N>`, the format
is `catmesh-1`, the landmark list, regions (`leg_FR` ... `tail`) and materials (`fur_orange` ...) are a
cat's, and the proxy template is a quadruped. For another subject keep every rule and change only those
lists (in `catlib/meshio.py`: `REGIONS`, `MATERIALS`, landmark names) and the proxy template.

Paths below are relative to the **case directory** (`--case DIR` or `LOWPOLY_CASE`). `scripts/` and
`catlib/` mean the skill's `scripts/` folder. Where the run used a project-local virtualenv, read
`python3`.

## 1. Environment

| Thing | Value |
|---|---|
| Blender | 5.2+ (`$BLENDER`, `blender` on PATH, or the macOS app); its bundled Python needs only numpy |
| Analysis Python | 3.10+ with numpy, pillow, scipy, opencv-python-headless, scikit-image, cma. Nothing uses the GPU. |
| ffmpeg | on PATH (turntables, timelapse) |
| Case root | the directory holding `reference/`, `contract/`, `proxy/`, `variants/`. Scripts resolve it from `--case` / `LOWPOLY_CASE` / the current directory, never from their own location. |

The reference image and its provenance file are read-only. Never regenerate, overwrite, resize in place, or
replace them. Derived files go in `reference/derived/`.

GPU rule: every Blender invocation that *renders* (EEVEE or Cycles, any resolution) must
hold the exclusive lock `catlib.gpulock.gpu_lock()` (an `fcntl.flock` on
`.cache/gpu.lock`) for the whole lifetime of that Blender process. The lock is taken in
exactly one place: inside the rendering Blender script itself, first thing, released at
process exit. Callers (e.g. `scripts/evaluate.py`) must **not** take it as well, or the
child would deadlock against its parent. CPU-only work (mesh
construction, numpy rasterisation, metrics, Blender used only for mesh assembly / export
/ counting) runs without the lock and may run concurrently.

## 2. World, units, cat frame

- Blender world coordinates, metres, right-handed, **+Z up**, ground plane **z = 0**.
- The cat is authored directly in world coordinates (object transform = identity).
  Canonical cat frame = world frame: body axis along X, **nose towards +X**, the
  **cat's own left towards +Y**, the cat's own right towards -Y. The torso's sagittal
  plane is y = 0 and the centroid of the four paw contacts is near x = 0, y = 0.
- Scale gauge (a single image has no absolute scale): the withers (top of the shoulders)
  of the fitted proxy cat is 0.25 m above the ground. Focal length is fixed by decision
  at 85 mm on a 36 mm-wide sensor (the generation prompt asked for an 85 mm look; this is
  an assumption, not a measurement). Camera roll is 0.
- "Left/right" in every file name, landmark name and region name means the **cat's own**
  left/right. In the reference the camera sees the cat's **right** flank; the head is
  turned towards the camera, so the cat's right ear/eye are on the image-left side of
  the face. Verify against the pixels, do not trust this sentence blindly.
- All cat geometry has z >= -0.002 m; paw soles touch z = 0.

## 3. Camera (`contract/camera.json`)

One pinhole camera, no distortion, square pixels, principal point at the image centre.

```
resolution      [1536, 1536]      (same as the reference)
focal_length_mm 85.0, sensor_width_mm 36.0, sensor_fit HORIZONTAL  -> f_px = 85/36*1536
location        [x, y, z]         world metres            (authoritative)
look_at         [x, y, z]         world metres            (authoritative)
roll_deg        0
matrix_world    4x4 row-major     derived from the above, stored for convenience
status          "provisional" | "locked"
```

Derivation: `fwd = normalize(look_at - location)`, `right = normalize(fwd x (0,0,1))`,
`up = right x fwd`; Blender camera axes are X=right, Y=up, Z=-fwd, so
`matrix_world = [right | up | -fwd | location]` (columns).

Pixel convention (used by masks, landmarks, the numpy rasteriser and all metrics):
`u` grows to the right, `v` grows **downwards**, origin at the top-left corner of the
top-left pixel, pixel centres at integer + 0.5. For a world point with camera coordinates
`(xc, yc, zc)` (`zc < 0` in front of the camera):

```
u = W/2 + f_px * xc / (-zc)        v = H/2 - f_px * yc / (-zc)        depth = -zc
```

`catlib.camera` is the single implementation of this; it must agree with Blender's own
projection to < 0.5 px. While `status` is `provisional` the numbers are a placeholder for
tool development. The camera stage fits the camera jointly with the proxy cat, writes the
final values and sets `status: "locked"`. After that nobody changes it, and every metric
ever reported in the final tables is computed with the locked camera.

## 4. Reference-derived data (`reference/derived/`)

- `mask.png` — 1536x1536, 8-bit single channel, 255 = cat, 0 = not cat. Foreground is
  the fur silhouette (body, head, ears, legs, paws, tail). The floor contact shadow is
  excluded. Individual whiskers / stray hairs outside the fur outline are excluded.
  Binarise with `>= 128`.
- `landmarks.json` — `{"image_size":[1536,1536], "landmarks": {name: {"uv":[u,v],
  "class":"sharp"|"fuzzy", "note": "..."}}}` in the pixel convention above. They are
  manual annotations made by inspecting zoomed crops, and are documented as such.
- Sharp landmarks (12): `ear_tip_R, ear_tip_L, eye_R, eye_L, nose, paw_FR, paw_FL,
  paw_HR, paw_HL, tail_base, tail_mid, tail_tip`.
  - `paw_*`: centre of the paw's ground footprint (F = front, H = hind).
  - `tail_base`: centre of the tail cross-section where it leaves the rump.
    `tail_mid`: point on the tail centreline at half its arc length. `tail_tip`: the tip.
  - `eye_*`: pupil centre. `nose`: centre of the nose pad.
- Fuzzy landmarks (4): `shoulder_R, hip_R, hock_HR, chin` — soft-tissue / joint
  estimates on the camera-facing side; reported separately, never mixed into the headline
  landmark number.
- Also saved: `mask_overlay.png`, `landmarks_overlay.png`, `reference_bbox.json`
  (`{"u0","v0","u1","v1","diag_px","area_px"}` of the mask) for visual checking and for
  metric normalisation.
- `regions.json` — fixed image-space windows `{name: [u0, v0, u1, v1]}` for `head`
  (ears, skull, muzzle, neck top), `tail` (whole tail up to the rump), `legs` (everything
  below the belly line, including the negative space between and around the four legs)
  and `torso`. They are drawn once from the reference and never moved. They exist so the
  metric can say *where* a budget gains or loses, and so leg gaps cannot hide inside a
  global score.

## 5. Mesh interchange: `mesh.json` (format `catmesh-1`)

The single source of truth for one checkpoint of one variant.

```
{
  "format": "catmesh-1",
  "budget": 100,                      # target budget this mesh is judged against
  "name": "cat_100",
  "vertices":  [[x,y,z], ...],        # world metres
  "triangles": [[i,j,k], ...],        # triangles ONLY, counter-clockwise seen from outside
  "tri_region":   ["torso", ...],     # one label per triangle, from REGIONS
  "tri_material": ["fur_orange", ...],# one label per triangle, from MATERIALS
  "landmarks": {"nose": [x,y,z], ...},# all 16 landmark names, world metres, ON the surface
  "notes": "free text"
}
```

- REGIONS = `head` (skull, cheeks, muzzle, chin), `ears`, `eyes`, `nose`, `neck`,
  `torso` (chest, belly, back, rump), `leg_FR`, `leg_FL`, `leg_HR`, `leg_HL` (each leg
  including its paw), `tail`. Every triangle has exactly one. This drives the
  "triangle allocation by anatomical region" table.
- MATERIALS = `fur_orange`, `fur_cream`, `fur_stripe`, `ear_inner`, `eye`, `pupil`,
  `nose`. Flat per-face colours only. No textures, no UV-projected photo, no alpha.
- Only triangles. `len(triangles)` is the triangle count and must be `<= budget`; the
  aim is `== budget`. No quads/n-gons, so nothing is hidden behind triangulation.
- A landmark is a point on the mesh surface (distance <= 2 mm) where that anatomical
  feature is on *this* mesh. At low budgets an eye landmark is simply the spot on the head
  surface where the eye would be.

## 6. Final files per variant and what "counted" means

`scripts/assemble_variant.py` turns a `mesh.json` into:

- `cat_<N>.blend` — contains exactly one collection `CAT_<N>` with exactly one mesh
  object `cat_<N>` (identity transform, flat shaded, one material slot per used material,
  integer face attribute `region_id` + the region names stored as a custom property).
  No modifiers, no shape keys, no curves, no particle/hair systems, no instances, no
  second mesh datablock, no camera, no lights, no ground, no wireframe helper.
- `cat_<N>.glb` — glTF binary export of that one object (Y-up, apply modifiers, no
  Draco, no cameras/lights).

Triangle count is established four independent ways and all four must agree and be
`<= budget`:

- A: `len(triangles)` in `mesh.json`.
- B: reopen the `.blend`; sum of `calc_loop_triangles()` over the **evaluated** mesh of
  **every** object in the file (any type, any collection, hidden or not).
- C: import the `.glb` into an empty Blender scene; sum of loop triangles over every
  imported object.
- D: pure-Python parse of the GLB container (no Blender): sum of `indices.count / 3`
  over every primitive of every mesh, multiplied by the number of nodes referencing that
  mesh; any primitive whose mode is not TRIANGLES is an error.

Hard failures (variant is invalid until fixed): any count disagrees; any count > budget;
anything extra in the `.blend`; degenerate triangles (area < 1e-6 m^2, i.e. 1 mm^2, or
shortest edge < 1 mm); duplicate triangles (same vertex set, or coincident positions
within 1e-6 m); min z < -0.002; smallest bounding-box extent < 12 % of the largest
(billboard guard).

Disclosed, not fatal, but to be minimised and explained: more than one connected
component; boundary (open) edges; non-manifold edges; inconsistent winding;
self-intersecting triangle pairs; triangles never visible from any of 162 directions on
a sphere (`hidden_triangles`, wasted budget); interior vertices whose whole fan is
coplanar (`flat_redundant_vertices`, padding suspicion).

## 7. Official metrics (identical for every budget and baseline)

All official numbers come from a **Blender render** of the assembled `.blend` through the
locked camera at 1536x1536, never from the optimiser's internal rasteriser.

- Rendered mask `renders/ref_silhouette.png`: cat = white, everything else = black, no
  ground, no lights needed; binarise with `>= 128`.
- `iou` = |R ∩ M| / |R ∪ M| with R = reference mask, M = rendered mask.
  Also `ref_uncovered_frac` = |R \ M| / |R| and `render_outside_frac` = |M \ R| / |M|.
- Boundary pixels = mask pixels with at least one 4-neighbour outside the mask.
  `boundary_mean_px` = mean of (mean distance from each R-boundary pixel to the nearest
  M-boundary pixel) and the reverse. `boundary_hd95_px` = max of the two directed 95th
  percentiles. `boundary_max_px` = symmetric Hausdorff. Each also reported `_norm` =
  px / `diag_px` of the reference mask bbox.
- `region_iou.<name>`: the same IoU restricted to each window of `regions.json`, plus
  `legs_gap_iou` = IoU of the *background* inside the `legs` window (negative space:
  are the gaps between the legs open where the reference has them open).
- Landmarks: project the variant's 3D landmarks with the locked camera.
  `landmark_sharp_mean_px` / `_max_px` / `_mean_norm` (÷ `diag_px`) over the 12 sharp
  landmarks is the headline landmark number; `landmark_fuzzy_mean_px` is separate. A
  per-landmark table is kept. A landmark is flagged `off_surface` (> 2 mm from the
  mesh) or `occluded` (first camera-ray hit more than 25 mm in front of it; 60 mm for
  `paw_*`, whose sole-centre definition sits behind the paw's own front surface); flagged
  landmarks are still scored and the flags are reported.
- Cross-check: IoU between `catlib.raster` silhouette and the Blender silhouette of the
  same mesh must be >= 0.99 (aim >= 0.995), otherwise the evaluation is reported as failed;
  the actual value is always recorded.
- Secondary, explicitly *not ground truth* (the proxy is inferred geometry): mean
  silhouette IoU against the proxy over the fixed novel views (`novel_view_iou_vs_proxy`)
  and symmetric Chamfer distance to the proxy surface in mm (`chamfer_mm_vs_proxy`).

Photo/colour similarity is not a metric here. Higher budgets are not assumed better.

No single number is trusted alone: a result is only called an improvement when IoU,
symmetric boundary distance and landmarks do not contradict each other *and* the overlay
looks right (leg gaps, which paws are occluded, ear separation, muzzle, belly line, tail).

Provenance against stale outputs: every `metrics.json` records the SHA-256 of
`reference/cat.png`, `mask.png`, `landmarks.json`, `regions.json`, `camera.json`,
`scene.json`, the `mesh.json` it scored, and one `tooling_hash` over `catlib/*.py` and
the shared `scripts/*.py`. The final tables only combine rows whose reference, camera,
scene and tooling hashes are identical; if shared code, the camera or an annotation
changes, every affected budget is re-evaluated before any comparison.

## 8. Renders (`scripts/render_variant.py`, same scene for every mesh)

Presentation objects (ground/shadow catcher, lights, world, cameras, wireframe overlay)
are built in memory by the render script from `contract/scene.json` and are **never**
saved into a variant `.blend` or exported.

Per checkpoint, in `renders/`:

| File | Content |
|---|---|
| `ref_silhouette.png` | official binary mask render, locked camera, 1536² |
| `ref_clay.png` | neutral grey clay, **flat shaded**, locked camera, 1536² |
| `ref_wire.png` | same clay + every triangle edge drawn dark (hidden lines removed) |
| `ref_beauty.png` | flat per-face material colours, flat shaded |
| `ref_overlay.png` | reference photo + rendered silhouette contour + reference (green) vs projected (red) landmarks |
| `view_<name>_clay.png`, `view_<name>_wire.png` | fixed novel views, 1024² |
| `turntable_clay_wire.mp4`, `turntable_beauty.mp4` | 72 frames, 5° steps, 768², 24 fps, H.264 yuv420p |

Fixed novel views orbit `orbit_target` at the reference camera's distance with the same
focal length. Azimuth 0° = camera on +X (in front of the cat), 90° = on +Y (cat's left),
180° = behind, -90° = cat's right. Views: `front` (az 0, el 5), `side_R` (-90, 5),
`back` (180, 5), `side_L` (90, 5), `threeq_front_L` (45, 15), `threeq_back_L` (135, 15),
`threeq_back_R` (-135, 15), `top` (straight down, image-up = +X). The left-side views are
entirely inferred geometry.

Lighting is identical in every image of every budget: the light rig listed in
`scene.json` is defined relative to the reference view and rotates with the view azimuth
for novel views and turntables (equivalent to turning the cat on a turntable under fixed
studio lights). Engine: Cycles, 32 samples, no denoiser, Standard view transform; the
silhouette and triangle-ID passes are 1 sample at the pixel centre (no anti-aliasing).

## 9. Layout and ownership

```
contract/            orchestrator (camera.json values: camera stage; scene.json values: render stage)
catlib/              shared Python package (numpy only inside Blender-importable modules)
scripts/             shared CLIs
reference/derived/   preflight stage
proxy/               camera + proxy stage (inferred high-res cat; NOT a deliverable mesh)
baselines/           decimation baselines
variants/<N>/        the budget-<N> worker, and nobody else
  build.py           constructs this budget's topology -> mesh.json
  checkpoints/<tag>/ mesh.json, cat_<N>.blend, cat_<N>.glb, renders/, metrics.json, validation.json
  final/             copy of the chosen best validated checkpoint
  history.json       one row per evaluated checkpoint (appended by scripts/evaluate.py)
  NOTES.md           allocation table, what was tried, what failed
outputs/             integration: comparison sheets, comparison.blend, metrics.json/csv, turntables
```

One evaluation entry point for everybody:

```
python3 scripts/evaluate.py --mesh variants/100/checkpoints/v0_initial/mesh.json \
    [--views] [--turntable] [--no-render]
```

It assembles, validates (4-way count + topology), renders under the GPU lock, computes
the official metrics, writes `metrics.json` / `validation.json` next to the mesh and
appends to `history.json`. Exit code is non-zero on any hard failure.

## 10. Lessons carried over from the earlier room experiment

These come from the Blockout 2 Real run (gokayfem/blockout-to-real).

- The reference, mask, landmarks, windows and camera are fixed targets. Nothing is
  regenerated or re-annotated to make a candidate fit; a genuine annotation correction is
  disclosed and triggers re-evaluation of every budget.
- An edge/outline score alone once passed visibly wrong geometry. See §7: combined
  metrics plus overlays, with explicit attention to negative space between the legs.
- More primitives made the earlier result worse. Spend triangles on anatomical contours,
  transitions and stance, not on extra lumps, and prove each change helped. Keep the best
  validated checkpoint per budget; the newest or the largest need not win.
- The independent critic sees the reference and neutral-named renders only, never scores
  or success claims, and its findings are converted into concrete budget-neutral
  geometric edits. Billboard, texture-mask, fur-card or uncounted-geometry suggestions are
  rejected.
- Materials and lighting never count as shape progress: no bump/normal/displacement
  maps, no stripe geometry, no hidden fill lights (every light is listed in
  `scene.json`), and geometry is ranked on clay / silhouette / landmarks only.
- Novel views are checked early, not at the end: all four paws, belly, head thickness and
  tail attachment must survive the orbit. Unseen shape is inferred, not verified.
- The earlier `inspect_blend.py` reports raw `len(mesh.polygons)`; that is **not** a
  triangle count and must not be reused as one. §6 is the only valid count.
- Monocular depth/normal estimates are optional, uncertain priors and are not used unless
  the orchestrator decides a specific ambiguity justifies one.

## 11. Inferred proxy cat and the locked camera (`proxy/`)

A single photo does not show the far side, the belly from below, the head's depth or the
body's width. Those are **inferred** once, in one place, so that all four budgets
approximate the same solid cat rather than four different guesses. The proxy is a
high-resolution *target and prior*, not a deliverable and never part of any variant file.

- `proxy/skeleton.json` (`catskel-1`) — the parametric cat as generalised cylinders:
  parts `torso, neck, head, ear_R, ear_L, leg_FR, leg_FL, leg_HR, leg_HL, tail`
  (optionally `muzzle`), each an ordered list of cross-section rings
  `{"c":[x,y,z], "u":[..], "v":[..], "ru":r, "rv":r, "p":2.0}` (centre, two orthonormal
  in-plane axes, semi-axes, superellipse exponent) plus cap types. A ring point at angle
  θ is `c + ru·sgn(cosθ)|cosθ|^(2/p)·u + rv·sgn(sinθ)|sinθ|^(2/p)·v`.
- `catlib/skeleton.py` — the only reader: `load_skeleton()`, `part.center(t)`,
  `part.frame(t)`, `part.radii(t)`, `part.ring(t, n, phase=0.0) -> (n,3)` for normalised
  arc length `t in [0,1]`, `part.length`, and `tube_mesh(part, ts, n, cap_start, cap_end)`.
  Budget workers resample this at whatever ring/side counts their allocation affords.
- `proxy/proxy_mesh.npz` — `vertices (V,3) float64`, `triangles (T,3) int32`: all parts
  fused into one watertight surface (roughly 20k-80k triangles).
- `proxy/sdf.npz` — `origin (3,)`, `spacing` (<= 3 mm), `sdf (nx,ny,nz) float32` signed
  distance in metres, negative inside, covering the proxy bbox + 5 cm; read through
  `catlib.proxy.sdf_lookup(points)` (trilinear) and `catlib.proxy.closest_surface(points)`.
- `proxy/landmarks3d.json` — the 16 landmarks as 3D points on the proxy surface.
- `proxy/part_labels.npz` or an equivalent `catlib.proxy.region_of(points)` so any mesh
  can be given `tri_region` labels consistently.
- `proxy/fit_report.json`, `proxy/renders/` — how well the proxy itself matches the
  reference through the locked camera (same official metrics), the fitted parameters, the
  priors used, and a list of what is inferred rather than observed.

The camera is fitted **jointly** with this proxy (free: location, look_at; fixed: 85 mm,
roll 0, principal point centred; gauge: withers at 0.25 m), then `contract/camera.json`
gets `"status": "locked"` and `scene.json["orbit_target"]` is set to the proxy's bbox
centre. Both are frozen from then on.

## 12. Landmark attachment and shared vertex refinement

- A variant's landmark is attached to the mesh feature that *is* that anatomy and moves
  with it: `ear_tip_*` = the ear's apex vertex; `tail_tip` = the tail's tip vertex (or
  tip-cap centre); `paw_*` = centre of that paw's sole; `nose` = the muzzle-tip vertex or
  nose-face centroid; `eye_*` = centroid of the eye faces if eyes are modelled, otherwise
  a fixed barycentric point on the head face where the eye would sit; `tail_base`,
  `tail_mid` = points on the tail's first ring / half-arc-length ring; fuzzy landmarks =
  fixed barycentric points on the relevant near-side faces. Landmark error may only be
  reduced by moving geometry, never by sliding the attachment to a more convenient spot
  afterwards. `mesh.json` may carry `"landmark_attach": {name: {"tri": i, "bary":
  [a,b,c]} | {"vertex": i}}`; if present, tools recompute `landmarks` from it.
- `scripts/refine_vertices.py` (shared, CPU-only) moves vertices of a fixed-connectivity
  mesh to reduce a weighted objective: reference-view silhouette (IoU + boundary term at
  a chosen working resolution) + landmark reprojection + agreement with the proxy
  (surface distance via the SDF, and low-resolution novel-view silhouettes) +
  regularisers (soles on z = 0, no flipped or degenerate triangles, triangle quality,
  optional mirror-symmetry groups). It never changes the triangle count. Its internal
  score is a search heuristic only; the official metrics of §7 decide what is kept.
- `catlib/topo.py` (shared, CPU-only) offers small tested mesh edits that keep
  `tri_region`, `tri_material` and landmark attachments consistent: edge flip, edge
  split, edge collapse, vertex weld, ring bridging (equal and unequal counts), fan/tip
  caps. `refine_vertices.py --flips` can also try silhouette-aware edge flips. These are
  tools; choosing where to use them is the budget worker's job.
- Topology decisions — how many rings and sides each part gets, where parts are welded,
  edge flips/splits/collapses, what is dropped at low budgets — belong to the budget
  worker and are the experiment. The refinement tool is the same for all budgets so that
  differences between budgets come from triangle allocation, not from tooling quality.

## 13. Build timelapse (`scripts/timelapse/`)

In this repository the timelapse tools live in `scripts/timelapse/` and write to
`outputs/timelapse/`; the film compositor of the cat run is not shipped.

Added on user request: a timelapse of the meshes being built, drawn on top of the cat
photo, and a release film in the style of the earlier "Blockout 2 Real" film
(1920x1080, 24 fps, PIL compositor, beat-timed, dark panel with cyan accents).

- Every frame shows real pipeline output: the reference photo, the fixed mask/landmarks,
  the proxy, real intermediate geometry and the final validated meshes. The compositor
  only lays things out, wipes between them and draws text. Numbers on screen come from
  `outputs/metrics.json`; a number computed by the optimiser's internal rasteriser is
  labelled as such or not shown.
- `scripts/refine_vertices.py --snapshot-dir DIR --snapshot-every K` (additive, default
  off) writes the vertex array every K accepted-or-rejected proposals:
  `snap_<proposal>.json` = `{"proposal": i, "vertices": [...], "objective": x}`.
- `variants/<N>/timelapse/stages.json` — written by the budget-N worker:

  ```
  {"budget": N, "final_checkpoint": "<tag>",
   "stages": [
     {"label": "torso rings", "kind": "construct", "mesh": "construct/01_torso.json"},
     ...
     {"label": "refine", "kind": "refine", "base_mesh": "refine/base.json",
      "snapshots": "refine/snaps/"},
     {"label": "final", "kind": "final", "mesh": "../final_mesh.json"}]}
  ```

  A timelapse mesh needs only `vertices`, `triangles`, `tri_region` (construction stages
  may be partial: a few parts, fewer triangles). `construct` stages come from the same
  `build.py` that made the real mesh (emit the parts in build order); `refine` stages are
  real optimiser snapshots of the run that produced the chosen checkpoint (re-run with
  the same seed and arguments plus the snapshot flags, and verify the last snapshot
  reproduces the checkpoint); `edit` stages are the real meshes before/after a topology
  edit. The last stage is byte-identical in vertices and triangles to the final mesh.
- The film may interpolate vertex positions linearly between consecutive real snapshots
  of the same connectivity; between different connectivities it cuts or cross-fades. It
  never invents geometry.
- Overlays on the photo use the locked camera and `catlib.camera` projection, so the
  wireframe sits exactly where the official silhouette render does.
- Outputs: `release_video/out/cat_triangle_budgets_release.mp4` (+ `_silent.mp4`),
  `release_video/out/timelapse_<N>.mp4` per budget and `timelapse_all.mp4`.

## 14. Shared head kit (`head/`, added after the first critic review)

Round 1 showed that four separately improvised heads were the weakest part at every
budget, and that the shared optimiser folds a face when it chases landmarks without
symmetry. So the head is designed once: one bilaterally symmetric cat head at five levels
of detail (`head/head_L0.json` 22 triangles … `head_L4.json` 224), fitted once to the
photo through the locked camera with symmetry preserved (one pose and one set of
proportions shared by all levels), carrying its own regions, materials, `neck_loop`,
mirror pairs and feature-attached landmarks.

- A budget worker welds one level onto its body (`catlib.headkit.attach_head`, or a 1:1
  weld of its neck ring onto the kit loop), keeps every kit vertex frozen during
  refinement, and keeps the kit's landmark attachments. Only the body moves.
- The head's triangles count toward the budget like any others; which level a budget can
  afford, and what the body gives up for it, is part of that budget's allocation table.
- The kit head is inferred where the photo cannot decide (depth of the skull, ear
  thickness, the far side). A frozen kit head may fit the head window slightly less
  tightly than an optimiser-fitted but folded one; both versions are evaluated officially
  and the choice between them is reported with its reason.
