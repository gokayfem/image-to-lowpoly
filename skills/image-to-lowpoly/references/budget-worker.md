# Budget worker: one subject, N triangles

The brief each budget worker received in the cat run, kept as written. Read "cat" as your subject, the paths
as relative to the case directory, and `scripts/` as the skill's `scripts/` folder. Where it says
`.venv/bin/python`, use your analysis Python. Round 2 of the run added one rule the brief did not yet have:
**build one closed surface; weld limbs, tail and ears into the body instead of pushing closed parts into it**
(see `lessons.md`). Round 3 added the shared head kit (`contract.md` section 14).

## The problem you are solving

You have exactly **N triangles** to make the most recognisable, best-proportioned solid
cat that matches `reference/cat.png` through the locked camera. The triangle allocation
is the experiment: where each triangle goes, what is merged, what is dropped.

- Target in the reference view: `reference/derived/mask.png` + `landmarks.json` through
  `contract/camera.json` (locked). This is the only observed evidence.
- Target everywhere else: the inferred proxy (`proxy/`), read through
  `catlib.skeleton` (resample any part at any ring/side count) and `catlib.proxy`
  (`sdf_lookup`, `closest_surface`, `region_of`). The proxy is a prior, not truth; where
  it disagrees with the photo in the reference view, the photo wins.

You own `variants/<N>/` and nothing else. Do not edit `contract/`, `catlib/`, `scripts/`,
`proxy/`, `baselines/`, `reference/`, or another budget's directory. If a shared tool
looks wrong, report it; work around it only inside your own directory. No git commits.

## Procedure

1. **Study.** Open `reference/cat.png`, `reference/derived/landmarks_overlay.png`,
   `proxy/renders/ref_overlay.png`, the proxy clay views in `proxy/renders/`, and (when
   present) the naive controls for your budget in `baselines/decimate/<N>/` and
   `baselines/decimate_refined/<N>/` (their `metrics.json` and renders). List the
   `proxy/skeleton.json` parts and what each costs at different ring × side counts.
2. **Allocate.** Write the triangle allocation table (REGION → triangles, with the
   rings × sides behind each number and a one-line reason) in `variants/<N>/NOTES.md`.
   Build at least **two materially different allocations** (not the same layout ±2
   triangles), give each the same short refinement, compare them on the unofficial
   `catlib.metrics` numbers *and* by looking at them, and pick one with the evidence
   written down. Losing candidates stay in `NOTES.md` with their numbers.
3. **Build.** `variants/<N>/build.py` (CLI, `.venv/bin/python`, deterministic) constructs
   the chosen topology and writes a `catmesh-1` `mesh.json` with `tri_region`,
   `tri_material`, `landmark_attach` (CONTRACT §12) and exactly N triangles (assert it).
   Shared helpers: `catlib.skeleton` (`part.ring`, `tube_mesh`), `catlib.topo`
   (`bridge_rings`, `cap_fan`, `cap_tip`, `edge_flip`, `edge_split`, `edge_collapse`,
   `weld_vertices`, `orient_outward`), `catlib.meshio` (`attach_point`, `save`).
4. **Initial checkpoint.** Evaluate the unrefined construction officially:
   `.venv/bin/python scripts/evaluate.py --mesh variants/<N>/checkpoints/v0_initial/mesh.json --views`.
   This is the "initial" row of the final tables. Open `renders/ref_overlay.png`,
   `ref_clay.png`, `ref_wire.png` and the novel views.
5. **Refine, measure, look, edit, repeat.** Move vertices with the shared tool
   (`.venv/bin/python scripts/refine_vertices.py --mesh in.json --out out.json ...`, see
   `--help`; add `--flips` to let it try silhouette-aware edge flips), then evaluate
   officially, then look, then make **targeted topology edits** where the overlay shows
   the budget is misallocated (move a ring to where the outline bends, give a silhouette
   edge another side, collapse triangles the camera and the novel views never benefit
   from, split where a landmark feature is missing), rebuild/refine/evaluate again. At
   least two measured refinement checkpoints after `v0_initial` (`v1_*`, `v2_*`, ...),
   each with an official evaluation and a line in `NOTES.md` saying what changed and
   whether it helped. Keep the best validated checkpoint; the newest need not win.
6. **Choose.** Write `variants/<N>/BEST.json`: `{"checkpoint": "<tag>", "reason": "..."}`.
   Do not create `variants/<N>/final/`; the orchestrator finalises.

## Hard rules

- Aim for exactly N triangles; never more. If you finish below N, say why.
- One mesh object, triangles only, no modifiers. A single closed manifold is preferred.
  Allowed with disclosure: several closed components that interpenetrate (e.g. an ear or
  a leg pushed into the body), or an open tube whose rim is entirely buried inside
  another closed part. Every triangle buried inside another part is wasted budget:
  `hidden_triangles` should be 0 or explained. Visible holes are not acceptable.
- No double-sided coincident faces (an ear needs real thickness), no degenerate or
  duplicate triangles, no coplanar padding to reach the count.
- Real volume from every view: no billboard, no flat silhouette extrusion.
- Landmarks are attached to the feature they name (§12) and only move with geometry.
- Materials are flat per-face colours with anatomical meaning (cream chest / muzzle /
  paws, pink inner ear and nose, amber eyes with dark pupils where the budget models
  them). They never substitute for geometry.
- No other cat asset, no image-to-3D service, no new reference image.

## Priorities when triangles are scarce

Silhouette in the reference view → head and ears → four distinct legs, stance and the
negative space between them → tail curve → recognisable anatomy (chest, belly line,
haunch, hock bend, muzzle) → small surface details last. More small lumps is not better
geometry; contours and transitions are.

## What to check with your own eyes every iteration

- `ref_overlay.png`: leg gaps open where the photo has them open, which paws are in
  front, ear tips and ear separation, muzzle and chin, back line, belly line, chest
  front, tail curve and tip.
- Novel views (`view_*_clay.png`, `view_*_wire.png`): four paws on the ground with a
  believable stance width, head with depth, tail attached to the rump, no part collapsed
  to a sheet, and **no spikes or folds on the hidden side** created by the optimiser
  chasing the reference silhouette. A better score with a worse shape is a regression.

## Machine etiquette

- Rendering happens only through `scripts/evaluate.py`; it serialises itself on the GPU
  lock and may block while another worker renders. Never wrap it in the lock, never call
  Blender to render yourself, do not request turntables.
- Three other budget workers share the 10 CPU cores: run one optimisation process at a
  time, and keep official evaluations to the ones you need.
- Evaluation code is frozen. Every checkpoint must be scored by it unmodified.

## Report (concise, under 450 words)

Allocation table for the chosen design and the rejected one(s) with numbers; the
checkpoint table (tag, triangles, official IoU, boundary mean / HD95 px, sharp-landmark
mean px, legs_gap_iou, region IoUs, novel-view IoU and Chamfer vs proxy) for every
official evaluation; which checkpoint is best and why; topology disclosures (components,
boundary edges, self-intersections, hidden triangles); what still visibly mismatches the
photo and what you would do with 20 % more triangles; exact commands to rebuild; anything
that failed.
