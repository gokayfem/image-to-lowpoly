# Budget 200: notes (worker: budget-200)

All "official" numbers come from `scripts/evaluate.py` (Blender silhouette, locked camera).
Candidate A_r and B_r rows are UNOFFICIAL (`--no-render`, catlib.raster; the raster-vs-Blender
cross-check is 1.0000 on every official run, so they are comparable). Triangle counts are the
validator's A=B=C=D four-way counts (mesh.json, evaluated .blend, re-imported GLB, parsed GLB).

## Construction (build.py)

Separate generalised-cylinder components resampled from `proxy/skeleton.json`, each built
explicitly (rings, bands, cones, fans), then `topo.orient_outward`:

- torso+neck: one tube from the rump tip cap to the chest, then up the neck. Its last ring is
  pulled into the head and left open (rim buried inside the head).
- head: closed. Back cap, 2 rings, and a muzzle cone ending at a nose-tip vertex placed at the
  proxy nose. Facing the camera.
- ears: k-sided pyramids with real thickness, open base buried in the head, apex at the
  proxy ear tip.
- legs: open top rim buried in the torso, bend rings, a ground sole polygon at z = 0 with a
  forward toe vertex, closed by a fan.
- tail: open start rim buried in the rump, rings, and a tip vertex at the proxy tail tip.

Every open rim is checked to be buried: `check_rims.py` requires generalised winding number
>= 0.5 against all other components. Every checkpoint passes; the minimum winding is about 0.95.
Rim vertices are frozen during refinement (`rim_verts.py` -> `--freeze-verts`).

## Allocation candidates (same short refinement: 8000 proposals, seed 0, no flips, rims frozen)

| | A: 6-sided torso, 4-sided legs | B: 5-sided torso, 3-sided legs with 2 bend rings |
|---|---|---|
| torso | 6 sides, rump tip + 4 rings (t 0, .30, .62, chest) = 42 | 5 sides, rump tip + 5 rings (t 0, .2, .45, .7, chest) = 45 |
| neck | 2 bands (neck t .45, buried rim) = 24 | 2 bands = 20 |
| head | 7 sides, fan back (5) + 1 band (14) + muzzle cone (7) = 26 | 8 sides, tip back (8) + band (16) + cone (8) = 32 |
| ears | 2 x 4 (4-sided pyramids) = 8 | 2 x 3 (3-sided) = 6 |
| legs | 4 sides, top/mid/sole: 2 bands + 2-tri sole = 18 x 4 = 72 | 3 sides, top/2 bends/sole: 3 bands + 1 = 19 x 4 = 76 |
| tail | 4 sides x 4 rings + tip = 28 | 3 sides x 4 rings + tip = 21 |
| total | 200 | 200 |
| unoff. IoU / bnd mean / HD95 | 0.9485 / 4.41 / 17.9 px | **0.9643 / 3.00 / 10.2 px** |
| sharp lm / legs_gap | 6.77 px / 0.915 | 7.40 px / **0.949** |
| novel IoU vs proxy / chamfer | 0.920 / 7.33 mm | **0.925** / 7.33 mm |

Decision: **B**. B gives one extra bend per leg (elbow plus wrist on the front legs, stifle
plus hock on the hind legs), and the second bend matters more to the outline and the leg gaps
than a 4th side does. Its 8-sided head beats the 7-sided one: unoff. head IoU 0.954 vs 0.936.
B's legs are triangular in cross-section; they read as legs from the front and back (looked at
in the CPU preview `candidates/AB_preview.png` and in the official v0/v1 views). A's only
advantage was the sharp landmark mean. Losing candidate A is kept in `candidates/`.

Before the candidates, a first build had no chest ring (torso ended at t = 0.92 and went
straight into a buried neck rim). The chest/neck front was then missing (unoff. IoU A 0.80,
B 0.84). Both allocations got a chest ring at the torso end-cap plus a full neck ring before
comparison.

## Chosen allocation (v3_tailbase = build.py --alloc D)

| region | tris | rings x sides / reason |
|---|---|---|
| torso | 35 | 5 sides; rump tip (5) + rings t 0, .30, .65, chest (t 1 + 0.6 cap depth, x0.85) = 3 bands (30); vertex on the back line, rings at rump, hip/belly, chest |
| neck | 20 | 5 sides; chest -> neck mid (t .45) -> rim buried in head; gives chest front and nape |
| head | 30 | 8 sides; flat fan back (6, hidden from camera) + 1 band (16) + muzzle cone to nose vertex (8); cheek width, chin, muzzle toward camera |
| ears | 8 | 2 x 4-sided pyramid, base x1.25, thickness = proxy ear thickness, base buried |
| leg_FR, leg_FL | 21 each | 3 sides; buried top -> elbow (t .37) -> wrist (t .68) -> 4-corner sole with toe (3->4 zipper, 7) + 2 sole tris |
| leg_HR, leg_HL | 19 each | 3 sides; buried top -> stifle (t .285) -> hock (t .65) -> triangular sole with toe + 1 sole tri |
| tail | 27 | 3 sides; buried start, rump exit ring (t .08), t .32, .58, .84 + tip (3) |
| **total** | **200** | 124 vertices, 9 components |

## Checkpoints (official unless noted)

| tag | tris | IoU | bnd mean / HD95 px | sharp lm px | fuzzy lm px | legs_gap | region IoU head/tail/legs/torso | novel IoU vs proxy | chamfer mm | comps | bnd edges | self-isect pairs | hidden tris |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decimate (control) | 200 | 0.9105 | 7.99 / 20.6 | 9.44 | 16.9 | 0.855 | 0.935/0.861/0.816/0.983 | 0.9336 | 3.27 | 1 | 0 | 0 | 0 |
| decimate_refined (control) | 200 | 0.9668 | 2.77 / 8.5 | 5.37 | 24.1 | 0.956 | 0.952/0.940/0.941/0.990 | 0.9497 | 3.29 | 1 | 0 | 0 | 0 |
| v0_initial (B, unrefined) | 200 | 0.8838 | 10.18 / 36.0 | 9.08 | 24.6 | 0.883 | 0.805/0.745/0.830/0.956 | 0.8430 | 11.28 | 9 | 26 | 108 | 4 |
| v1_refine | 200 | 0.9619 | 3.17 / 11.2 | 6.77 | 47.0 | 0.954 | 0.928/0.938/0.938/0.991 | 0.9324 | 6.83 | 9 | 26 | 133 | 1 |
| v2_alloc_C | 200 | 0.9634 | 3.05 / 10.8 | 8.01 | 24.6 | 0.953 | 0.941/0.924/0.937/0.992 | 0.9323 | 6.43 | 9 | 28 | 133 | 1 |
| **v3_tailbase** | 200 | **0.9640** | **3.00** / 11.0 | **6.76** | **22.6** | 0.954 | 0.944/0.928/0.939/0.991 | 0.9300 | 6.42 | 9 | 28 | 141 | 1 |
| v4_proxyreg | 200 | 0.9640 | 3.01 / 10.4 | 7.24 | 23.3 | 0.954 | 0.948/0.928/0.938/0.991 | 0.9336 | 6.37 | 9 | 28 | 145 | 1 |

- **v1_refine**: v0 + `refine_vertices.py --iters 20000 --flips --seed 0`, rims frozen (37 s).
  Big gain everywhere. In the overlay, the ears' outer edges are too thin, the paw bottoms are
  short, tail_base is 38 px off (flagged occluded), and the chin is 41 px off. The novel views
  show volume and four paws; a small fold sits on the cat's-left side of the neck.
- **v2_alloc_C** (topology edit): the per-region diff image showed misses at the ears, the paw
  soles and the right-hand neck, while the torso was already at IoU 0.991. Changes: one torso
  ring dropped (-10), head back tip changed to a fan (-2); spent on 4-sided ears with a 1.25x
  base (+2), a tail ring at the rump exit (+6) and 4-corner front soles (+4). Same refinement.
  Result: head IoU 0.928 -> 0.941, chin 41 -> 6 px, boundary better, ears symmetric and wider
  in the front view. Sharp landmarks got worse (8.0 px) because tail_base was attached to a
  ring vertex, which sits about 60 px off the tail centreline.
- **v3_tailbase** (attachment fix at build time, then the same refinement): tail_base is
  attached to the closest surface point on the faces around the rump-exit ring, never slid
  after refinement. Sharp landmarks 6.76 px, tail_base 25 px (still flagged occluded: the
  tail root is seen through the rump flank, as with the proxy). Best IoU and boundary mean of
  my checkpoints.
- **v4_proxyreg**: v3 + 12000 proposals with `--w-sdf 0.5 --w-nv 1.5 --seed 1`, to tame the
  hidden-side neck spike. Mixed result: HD95, head IoU and novel-view IoU slightly better;
  sharp landmarks worse (7.24 px, paw_FR 9.7 -> 12.7). The spike moved but did not go away.
  Not kept.

## Honest comparison with the controls

- Plain decimation (0.9105) is beaten clearly: leg gaps 0.855 -> 0.954, tail region 0.861 ->
  0.928, legs region 0.816 -> 0.939, sharp landmarks 9.4 -> 6.8 px.
- **The refined decimation control is still slightly better than v3 in the reference view**
  (IoU 0.9668 vs 0.9640, HD95 8.5 vs 11.0, sharp landmarks 5.4 vs 6.8 px), and clearly better
  against the proxy (novel-view IoU 0.950 vs 0.930, Chamfer 3.3 vs 6.4 mm). Reasons:
  - It is one adaptive manifold with no buried geometry.
  - My part-based design spends roughly 25-30 triangles on bands whose first ring is buried
    (leg tops, neck end, tail root). These are only partly visible, so `hidden_triangles` = 1
    but they add little outline.
  - Part of my Chamfer is the buried interpenetrating surfaces themselves, measured against
    the proxy surface.
  - What v3 offers instead: an editable, anatomically labelled part structure, ears with
    thickness, explicit sole polygons on z = 0 with toes, and landmarks bound to the features
    (ear apexes, nose vertex, tail tip, soles).
  - I do not claim v3 beats the refined control numerically.

## What 20 % more triangles would buy (about 40)

Weld the legs and tail into the torso rings instead of burying open tubes. This is the main
lever: it recovers the buried-band waste and removes the self-intersections. Then:

- a 4th side on the front legs;
- a jaw/cheek ring on the head, since the face is the main remaining head-region miss;
- one more tail ring at the tightest curve.

## Commands

    .venv/bin/python variants/200/build.py --alloc D --out variants/200/checkpoints/v3_tailbase_init/mesh.json
    .venv/bin/python variants/200/rim_verts.py variants/200/checkpoints/v3_tailbase_init/mesh.json variants/200/checkpoints/v3_tailbase_init/rims.json
    .venv/bin/python scripts/refine_vertices.py --mesh variants/200/checkpoints/v3_tailbase_init/mesh.json --out variants/200/checkpoints/v3_tailbase/mesh.json --iters 20000 --flips --freeze-verts variants/200/checkpoints/v3_tailbase_init/rims.json --seed 0
    .venv/bin/python variants/200/check_rims.py variants/200/checkpoints/v3_tailbase/mesh.json
    .venv/bin/python scripts/evaluate.py --mesh variants/200/checkpoints/v3_tailbase/mesh.json --views

`--alloc B` reproduces v0_initial exactly (byte-identical vertices and triangles, checked).
Helpers: `preview.py` (CPU-only painter's quick-look, not a metric).

---

# Round 2: welded single manifold (orchestrator request)

**Best checkpoint is now `v5u_untangled`** (BEST.json). Round 1 (v0-v4, nine interpenetrating
parts) is superseded.

## Construction (`build.py --alloc W` -> `welded.py`)

One closed genus-0 surface, so T = 2V - 4: exactly 200 triangles means exactly 102 vertices.
Cut-and-bridge, no booleans:

- The torso -> chest -> neck -> skull -> face tube is generated with the limb and tail socket
  quads left out.
- Each 4-vertex socket loop is bridged straight to that limb's first ring. Unequal ring
  counts (6 -> 8 at the neck/skull) use the `topo._align_ring` alignment plus a zipper band.
- Ears are 4-sided pyramids standing on a head quad and sharing its vertices.
- Two `topo.edge_split`s add the last 2 vertices: a chin point at the proxy chin and a
  brisket point.
- Chest, neck and skull rings are custom tilted ellipses (chest normal tilted 35 deg up, neck
  ring leaning toward the camera, skull ring tilted 0.55 toward +z). The first welded try
  (W, plain skeleton rings) folded the neck into the chest and head: 23 intersecting pairs
  before refinement, 58 after. The tilted rings give a gradual bend: 2 pairs before refining.
- The tail grows out of the upper half of the rump end cap; the lower half is a 2-triangle
  fan. W's tail on the top of the rump put tail_base 53 px off; on the cap it is 13 px.

| region | vertices | tris | structure |
|---|---|---|---|
| torso | 30 | 48 | 6 sides x 5 rings (rump cap ring, hip, waist, shoulder, tilted chest); sockets for 4 legs + tail; 2-tri lower rump cap |
| neck | 6 | 26 | tilted neck ring; chest->neck 12, neck->skull 14 (6->8 zipper) |
| head | 17 + chin 1 | 26 | 8-sided skull + face rings (phase pi/8: flat top, ear quads), muzzle cone to nose vertex, chin split |
| ears | 2 | 8 | 4-sided pyramids on head quads, apex at proxy ear tips |
| each leg | 8 | 18 | 4 sides: socket -> hock/wrist ring (8) -> diamond sole on z = 0 with toe (8) + 2-tri sole |
| tail | 13 | 28 | 4 sides: socket -> 3 rings (t .12 .42 .72) -> tip |
| brisket split | 1 | +2 | chest underside |

## `untangle.py`

The shared refiner has no self-intersection term. `untangle.py` (deterministic) finds
intersecting non-adjacent triangle pairs. Its pair list is identical to the validator's on
v5_welded (12) and on the no-flip run (11). It then moves single vertices toward their 1-ring
centroid or the proxy surface, picking the smallest move that reduces the pair count.
On v5_welded: 3 vertices moved, at most 7.7 mm, 12 -> 0 pairs, metrics unchanged.

## Round-2 checkpoints (official) vs the control

| tag | IoU | bnd / HD95 px | sharp lm | fuzzy | legs_gap | head/tail/legs/torso | novel IoU | Chamfer mm | comps | bnd edges | self-isect | hidden |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decimate_refined (control) | 0.9668 | 2.77 / 8.5 | 5.37 | 24.1 | 0.956 | .952/.940/.941/.990 | 0.9497 | 3.29 | 1 | 0 | 0 | 0 |
| v3_tailbase (round 1 best) | 0.9640 | 3.00 / 11.0 | 6.76 | 22.6 | 0.954 | .944/.928/.939/.991 | 0.9300 | 6.42 | 9 | 28 | 141 | 1 |
| v5_welded | 0.9717 | 2.35 / 8.0 | 5.28 | 18.3 | 0.956 | .974/.943/.941/.994 | 0.9349 | 4.23 | 1 | 0 | 12 | 0 |
| **v5u_untangled** | 0.9716 | **2.35 / 8.0** | **5.24** | **18.3** | 0.956 | **.974**/.943/.941/.994 | 0.9345 | 4.26 | 1 | 0 | **0** | 0 |
| v6_nvreg | 0.9696 | 2.53 / 8.5 | 6.52 | 27.8 | 0.953 | .974/.941/.937/.994 | 0.9460 | 3.98 | 1 | 0 | 0 | 0 |

- **v5_welded**: `--alloc W` build (= `v5_welded_init`), then
  `refine_vertices.py --iters 20000 --flips --seed 0`, with no freeze list.
- **v5u_untangled**: `untangle.py` on v5_welded.
- **v6_nvreg**: v5u + `refine --iters 15000 --flips --seed 1 --w-nv 3.0 --w-sdf 0.3`, then
  untangle (2 vertices, at most 13.7 mm). It improves proxy agreement but worsens every
  reference-view number, landmarks most (6.52 px, above the control). Not kept.

Unofficial side runs (in `candidates/`), all starting from the W/v5 build:

| run | IoU | sharp lm px | novel IoU | Chamfer mm | self-isect |
|---|---|---|---|---|---|
| W2 with `--w-sdf 0.5 --w-nv 1.5` | 0.9588 | 7.58 | 0.942 | 4.02 | - |
| continuation `--w-nv 1.5` | 0.9713 | 5.92 | 0.9435 | 4.08 | 14 |
| continuation, 30000 proposals | 0.9716 | 6.03 | 0.943 | 4.02 | 13 |
| no `--flips` (then untangle needed a 43 mm move, rejected) | 0.9718 | 4.43 | 0.933 | - | 11 |

No setting reached novel-view IoU 0.95 without losing reference-view fit.

## What I see (8 novel views, ref clay/wire, overlay of v5u)

- One continuous surface:
  - the haunch flows into the hind thigh;
  - the shoulder into the front leg;
  - the chest into the neck and skull;
  - the tail grows out of the rump.
- No plugs, no collar slab, and no hidden-side neck spike in the front, side_L or threeq
  views.
- Legs are 4-sided above the hock/wrist, so they no longer read as blades from the front or
  back.
- Weaknesses:
  - the torso is still a 6-sided slab, narrow and flat-sided from the top; this is the main
    novel-view loss (top 0.929, threeq 0.92-0.93);
  - a small forward toe spike on paw_FL in the reference view;
  - chin 30 px off (fuzzy landmark);
  - Chamfer 4.26 mm.

## With 20 % more triangles

- An 8-sided torso: +10 vertices = +20 triangles. This is the dominant novel-view and
  Chamfer gap.
- An elbow/stifle ring on the hind legs.

## Commands (v5u_untangled)

    .venv/bin/python variants/200/build.py --alloc W --out variants/200/checkpoints/v5_welded_init/mesh.json --stages-dir variants/200/timelapse/construct
    .venv/bin/python scripts/refine_vertices.py --mesh variants/200/checkpoints/v5_welded_init/mesh.json --out variants/200/checkpoints/v5_welded/mesh.json --iters 20000 --flips --seed 0
    .venv/bin/python variants/200/untangle.py --mesh variants/200/checkpoints/v5_welded/mesh.json --out variants/200/checkpoints/v5u_untangled/mesh.json
    .venv/bin/python scripts/evaluate.py --mesh variants/200/checkpoints/v5u_untangled/mesh.json --views

## Timelapse

`variants/200/timelapse/stages.json`:

- 9 construct stages from the same `build.py --alloc W --stages-dir`.
- One refine stage: re-run of the exact v5_welded command with
  `--snapshot-every 400`, 51 snapshots in `refine/snaps/`. The output is identical to
  v5_welded; the last snapshot differs by at most 5e-8 m (file rounding).
- 115 edge flips were accepted during that run. Snapshots carry vertices only, so the
  connectivity changes inside the refine stage.
- Then the untangle edit and the final stage.

---

# Round 3: shared head kit (orchestrator request). **Pick: `v8_kitL2`** (BEST.json)

## Correction to rounds 1-2

The `band_uneq` zipper in `build.py` could wrap one ring around a single vertex of the
other. That left **1 non-manifold edge** (at a front sole) in round-1 v2/v3/v4, which I did
not report earlier.

- Fixed: the zipper now never steps onto an already-used vertex pair before its closing step.
- Builds B and W still reproduce v0_initial and v5_welded_init byte-identically.
- Rebuilding D now gives a corrected mesh; the stored v3 files keep the defect.

## Construction (`build.py --alloc K1|K2` -> `kitbody.py`)

The round-2 welded body has its open chest ring R3 welded to the kit's 6-vertex neck loop
with `catlib.headkit.attach_head` (a zipper band). The kit replaces the neck ring, skull and
face rings, nose, ears and chin. The vertex budget is fixed at 102 (T = 2V - 4).

| | K1 / v7_kitL1 | K2 / v8_kitL2 |
|---|---|---|
| head | L1 kit, 20 vertices, 32 tris | L2 kit, 29 vertices, 50 tris |
| torso | R0..R3 8-sided (phase pi/8), Rc 6-sided, + brisket split | 6-sided as v5, y x1.08 |
| legs | 4-sided; front soles 4-cornered, hind soles triangles | same |
| tail | 4-sided phase 0 (lateral vertex pair), radii x1.15 | same |
| triangles by region | torso 57, neck 15, head 26, ears 6, legs 16/16/18/18, tail 28 | torso 42, neck 12, head 40, nose 2, eyes 2, ears 6, legs 16/16/18/18, tail 28 |

Refinement: body only, kit frozen (`--freeze-verts` = the `attach_head` list),
`--iters 20000 --flips --seed 0`. Then `untangle.py --freeze-verts` (the kit is never moved):

- v7: 3 vertices, at most 45.9 mm.
- v8: 1 vertex, 20.7 mm.

## Official numbers

| tag | IoU | bnd / HD95 px | sharp lm | fuzzy | legs_gap | head/tail/legs/torso | novel IoU | Chamfer | comps / bnd / nm / self-isect / hidden |
|---|---|---|---|---|---|---|---|---|---|
| control | 0.9668 | 2.77 / 8.5 | 5.37 | 24.1 | 0.956 | .952/.940/.941/.990 | 0.9497 | 3.29 | 1/0/0/0/0 |
| v5u_untangled | **0.9716** | **2.35 / 8.0** | **5.24** | 18.3 | 0.956 | **.974**/.943/.941/.994 | 0.9345 | 4.26 | 1/0/0/0/0 |
| v7_kitL1 | 0.9557 | 3.80 / 16.3 | 11.09 | 11.5 | 0.937 | .935/.942/.913/.990 | 0.8951 | 4.72 | 1/0/0/0/0 |
| **v8_kitL2** | 0.9677 | 2.68 / 9.1 | 6.36 | 53.2 | **0.958** | .962/.918/.943/.994 | 0.8874 | 5.35 | 1/0/0/0/0 |

## Why the novel-view and Chamfer numbers dropped with the kit

The kit head was fitted to the photo; the proxy head was inferred. They differ in depth and
yaw. A raster silhouette diff of K2 against the proxy (front, side_R, top) shows the kit's
ears and face outside the proxy silhouette, and the proxy head outside the kit. The drop is
already there before refinement (K2 initial novel IoU 0.73-0.85 per view, vs W initial 0.84).
The body itself is unchanged in kind.

## What I see

`candidates/round3_compare.png`: head crop plus 8 views for v5u, v7 and v8.

- v8's head reads as a cat: brow plane, blunt muzzle wedge with a nose facet, eye
  triangles, thick ears. v5u's head is a long pointed cone.
- v8's far-front toe spike is gone.
- v7's 8-sided torso is not visibly rounder from the top and costs reference fit everywhere
  (worse legs, landmarks 11 px).
- Still open:
  - the tail is still thin from behind and from the top in all three; tail_scale 1.15 and
    the phase-0 lateral vertices did not survive the silhouette-driven refiner;
  - the torso is still narrow from above;
  - v8's shoulder_R fuzzy landmark drifted 132 px (the attachment followed a refined
    flank triangle);
  - a small toe point on paw_HL.

## Rebuild (v8)

    .venv/bin/python variants/200/build.py --alloc K2 --out variants/200/checkpoints/v8_kitL2_init/mesh.json --freeze-out variants/200/checkpoints/v8_kitL2_init/freeze.json --stages-dir variants/200/timelapse/construct_K2
    .venv/bin/python scripts/refine_vertices.py --mesh variants/200/checkpoints/v8_kitL2_init/mesh.json --out variants/200/checkpoints/v8_kitL2_refined/mesh.json --iters 20000 --flips --seed 0 --freeze-verts variants/200/checkpoints/v8_kitL2_init/freeze.json
    .venv/bin/python variants/200/untangle.py --mesh variants/200/checkpoints/v8_kitL2_refined/mesh.json --out variants/200/checkpoints/v8_kitL2/mesh.json --freeze-verts variants/200/checkpoints/v8_kitL2_init/freeze.json
    .venv/bin/python scripts/evaluate.py --mesh variants/200/checkpoints/v8_kitL2/mesh.json --views

## Timelapse

`timelapse/stages.json` now points at v8:

- 7 construct stages, including the head-kit weld.
- A refine stage: re-run with `--snapshot-every 400`, 51 snapshots. The output is identical
  to v8_kitL2_refined; the last snapshot differs by at most 5e-8 m.
- The untangle edit, then the final stage.
