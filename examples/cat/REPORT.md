> Copy of the run report from the original project. Links in it point at that project's layout (`variants/`, `outputs/`, `baselines/`, `release_video/`); in this repository only the files listed in `examples/cat/README.md` are shipped. Renders, `.blend` files and videos are not.

# One cat, four triangle budgets — report

Four editable low-poly meshes of one fixed cat photo, built under hard budgets of 100, 200,
500 and 1000 triangles. Every budget landed exactly on its count. This report says what was
done, what the numbers are, what an independent reviewer saw, and what is wrong with the
results.

Run on 2026-11-11, about three hours wall-clock on a 24 GB Apple M4. Orchestrator: Claude
Fable 5.2 (Extra High). Every implementation worker and both visual critics: Claude Opus 5.5
Medium through the project's `stage-worker` and `visual-critic` definitions. No model was
substituted.

**Mesh iteration was stopped by the user** ("current forms are enough") while a third round
was still running. The delivered meshes are the best validated forms at that moment, not the
end of what the method could reach. Section 8 lists what was in flight.

## 1. Result at a glance

![Clay and wireframe at the reference view](outputs/comparison_clay_wire.png)

| Budget | Triangles (4-way agreed) | Silhouette IoU | Boundary mean / HD95 | Sharp landmarks | Leg-gap IoU | Chosen checkpoint |
|---|---|---|---|---|---|---|
| 100 | 100 | 0.943 | 4.9 / 18.0 px | 8.6 px | 0.913 | `v3_welded` |
| 200 | 200 | 0.972 | 2.4 / 8.0 px | 5.2 px | 0.956 | `v5u_untangled` |
| 500 | 500 | 0.983 | 1.4 / 5.0 px | 2.8 px | 0.977 | `v3_nose` |
| 1000 | 1000 | 0.984 | 1.3 / 5.0 px | 1.6 px | 0.974 | `v7_body` |

All numbers are official: a Blender silhouette render of the assembled `.blend` through the
one locked camera at 1536², compared with the fixed mask and landmarks. Boundary distances
are in pixels of the 1536² reference (mask bounding-box diagonal 1510 px). Full tables:
[outputs/metrics.json](outputs/metrics.json), [outputs/metrics.csv](outputs/metrics.csv),
[outputs/comparison_metrics_table.png](outputs/comparison_metrics_table.png).

Three things to read before trusting that table:

- **Budget target achieved at every budget.** No budget is under or over.
- **Extra triangles stop paying early.** 100 → 200 → 500 each buy a clear gain; 500 → 1000
  buys almost nothing in the reference view (IoU +0.0014, boundary −0.1 px) and the second
  critic judged the 200 the best likeness per triangle.
- **The heads are the weak part at every budget, and the bodies are too narrow.** Both are
  described in section 6 and were not fixed in the delivered meshes.

## 2. Fixed input and targets

- Reference: [reference/cat.png](reference/cat.png), 1536², generated once with MiniMax H3
  Image on fal (`minimax/h3-image/text-to-image`) before this run and never regenerated or
  edited. Its SHA-256 matches [reference/manifest.json](reference/manifest.json) and is
  recorded in every `metrics.json`.
- Foreground mask (524,780 px, floor shadow and whiskers excluded), 12 sharp and 4 fuzzy
  landmarks, and four fixed image windows (head, tail, legs, torso):
  [reference/derived/](reference/derived/). The landmarks are manual annotations made from
  zoomed, gridded crops; overlays are in the same folder.
- One camera for everything: [contract/camera.json](contract/camera.json), fitted jointly
  with a proxy cat and then locked. 85 mm lens (fixed by decision, not measured), 1.36 m
  from the cat, 0.21 m high, azimuth −42°, elevation 0.3°. Scale convention: withers 0.25 m.
- Mask, landmarks, windows, camera, scene, proxy and evaluation code were frozen and hashed
  ([contract/FROZEN.json](contract/FROZEN.json)) before any budget mesh was scored. The four
  finals carry identical hashes.

The full rule set every worker followed is [contract/CONTRACT.md](contract/CONTRACT.md).

## 3. Technique

1. **Inferred proxy cat.** A parametric cat (generalised cylinders for torso, neck, head,
   ears, four independently posed legs, tail) was fitted with CMA-ES on a CPU rasteriser to
   the mask and landmarks, jointly with the camera, then fused into one 60,000-triangle
   surface ([proxy/](proxy/)). It is the shared prior for everything the photo cannot show.
   Through the locked camera the proxy itself scores IoU 0.950, sharp landmarks 4.7 px.
2. **Budget-specific construction.** Each budget had its own worker and directory
   (`variants/<N>/`). Each compared at least two materially different allocations, built its
   mesh from the proxy skeleton with its own `build` script, and scored the unrefined
   construction officially (the "initial" row).
3. **Shared vertex refinement.** One CPU-only, seeded optimiser
   ([scripts/refine_vertices.py](scripts/refine_vertices.py)) moves vertices of a
   fixed-connectivity mesh against the reference silhouette, boundary distance, landmark
   reprojection, the proxy's signed distance field and low-resolution novel views, with
   optional silhouette-aware edge flips. It never changes the triangle count. It was the same
   tool for every budget so differences between budgets come from triangle allocation.
4. **Welding into one surface.** The first-round meshes were kits of closed parts pushed
   into each other. In round 2 every budget was rebuilt as one closed manifold: limbs, tail
   and ears extruded from faces of the body at 100; sockets bridged ring-to-ring at 200;
   holes cut and welded at 500; an exact boolean union of the lofts followed by cheapest-
   first edge collapses to exactly 1000 at 1000.
5. **Render, measure, critique, revise.** One entry point
   ([scripts/evaluate.py](scripts/evaluate.py)) assembles the `.blend` and `.glb`, counts
   triangles four ways, renders through a GPU lock (Cycles on Metal, serialised), and
   computes the metrics. Two rounds per budget were completed, with a blind visual critic
   after round 1 and again on the finals.

Naive controls were built for every budget: Blender Decimate of the proxy to exactly N
triangles, and the same mesh after the shared refiner ([baselines/](baselines/)).

## 4. Triangle counting

Each mesh is triangles only, one object, no modifiers. The count is established four
independent ways and all must agree: the mesh file, the evaluated `.blend` (loop triangles
over every object in the file), the GLB re-imported into an empty scene, and a pure-Python
parse of the GLB container.

| Budget | mesh.json | evaluated .blend | re-imported GLB | parsed GLB |
|---|---|---|---|---|
| 100 | 100 | 100 | 100 | 100 |
| 200 | 200 | 200 | 200 | 200 |
| 500 | 500 | 500 | 500 | 500 |
| 1000 | 1000 | 1000 | 1000 | 1000 |

The validator was tested against deliberate cheats before use (over budget, duplicate
triangle, degenerate triangle, hidden extra object, quads hiding triangles, subdivision
modifier, hidden duplicate mesh, billboard) and each one fires. The comparison file
[outputs/comparison.blend](outputs/comparison.blend) was reopened and recounted per
collection: 100, 200, 500, 1000.

## 5. Per budget

### Topology of the finals

| Budget | Components | Open edges | Self-intersecting pairs | Never-visible triangles | Smallest triangle |
|---|---|---|---|---|---|
| 100 | 1, closed manifold | 0 | 3 | 1 | 3.0 mm² |
| 200 | 1, closed manifold | 0 | 0 | 0 | 27.9 mm² |
| 500 | 1, closed manifold | 0 | **95** | 0 | 14.7 mm² |
| 1000 | 1, closed manifold | 0 | 4 | 1 | 9.4 mm² |

The 500 mesh's 95 self-intersecting pairs sit at its welded joints (head/neck, tail/rump)
and were created by the refiner, which has no intersection constraint. A later 500 body
with a guard reached 14 pairs but was not the accepted form.

### Triangle allocation by region

| Region | 100 | 200 | 500 | 1000 |
|---|---|---|---|---|
| Head (skull, cheeks, muzzle) | 19 | 22 | 69 | 151 |
| Ears | 6 | 8 | 24 | 48 |
| Eyes | 0 | 0 | 2 | 6 |
| Nose | 0 | 0 | 1 | 6 |
| Neck | 10 | 27 | 32 | 75 |
| Torso | 16 | 43 | 98 | 223 |
| Front legs (right / left) | 7 / 7 | 18 / 18 | 52 / 52 | 85 / 85 |
| Hind legs (right / left) | 13 / 7 | 18 / 18 | 52 / 52 | 100 / 98 |
| Tail | 15 | 28 | 66 | 123 |

Legs take 34-42 % of every budget. At 100 the far hind leg lost its hock bend so the ears
could have thickness.

### Initial → final, and against the controls

| Budget | Row | IoU | Boundary mean | Sharp landmarks | Leg-gap IoU | Novel-view IoU vs proxy | Chamfer vs proxy |
|---|---|---|---|---|---|---|---|
| 100 | initial (unrefined build) | 0.769 | 18.9 px | 12.2 px | 0.641 | 0.772 | 10.6 mm |
| 100 | **final** | **0.943** | **4.9 px** | **8.6 px** | 0.913 | 0.897 | 6.8 mm |
| 100 | decimation | 0.884 | 10.7 px | 25.0 px | 0.845 | 0.879 | 5.2 mm |
| 100 | decimation + refiner | 0.935 | 5.7 px | 12.3 px | **0.921** | **0.904** | **5.2 mm** |
| 200 | initial | 0.884 | 10.2 px | 9.1 px | 0.883 | 0.843 | 11.3 mm |
| 200 | **final** | **0.972** | **2.4 px** | **5.2 px** | 0.956 | 0.935 | 4.3 mm |
| 200 | decimation | 0.911 | 8.0 px | 9.4 px | 0.855 | 0.934 | 3.3 mm |
| 200 | decimation + refiner | 0.967 | 2.8 px | 5.4 px | 0.956 | **0.950** | **3.3 mm** |
| 500 | initial | 0.916 | 7.7 px | 22.8 px | 0.894 | 0.919 | 4.9 mm |
| 500 | **final** | 0.983 | 1.4 px | 2.8 px | 0.977 | 0.961 | 3.5 mm |
| 500 | decimation | 0.942 | 5.0 px | 6.4 px | 0.927 | 0.975 | 2.2 mm |
| 500 | decimation + refiner | 0.983 | 1.4 px | **2.0 px** | 0.974 | **0.974** | **2.5 mm** |
| 1000 | initial | 0.939 | 5.2 px | 13.5 px | 0.930 | 0.934 | 6.3 mm |
| 1000 | **final** | 0.984 | 1.3 px | 1.6 px | 0.974 | 0.974 | 2.4 mm |
| 1000 | decimation | 0.948 | 4.5 px | 5.1 px | 0.940 | 0.987 | 2.0 mm |
| 1000 | decimation + refiner | 0.984 | 1.3 px | 1.6 px | **0.979** | **0.981** | **2.2 mm** |

"vs proxy" columns measure agreement with inferred geometry, not with truth.

What this says, honestly:

- **Deliberate allocation clearly wins at 100 and 200.** Decimation at 100 loses the ears
  entirely and leaves a box head; the designed mesh has separate ears and better landmarks.
  At 200 the designed mesh is ahead on IoU, boundary and landmarks.
- **At 500 and 1000 the designed mesh only ties the refined-decimation control** in the
  reference view (differences in the fourth decimal), and the control is ahead on leg gaps at
  1000 and on landmarks at 500.
- **The control agrees better with the proxy from other angles at every budget.** It is a
  decimation of the proxy, so that is partly by construction, but it is still a real result.
- Most of each budget's gain came from the shared refiner (initial → final), which the
  control also receives. The refiner, not the hand-built topology, is the larger effect.

### What improved at each budget, and what is left

- **100** — Round 1 (eight interpenetrating open parts) scored 0.923. Rebuilding as one
  surface with limbs extruded from the body raised it to 0.943 and opened the leg gap
  (0.870 → 0.913). Left: a flat three-sided blade for a tail, a faceted box head, no hock on
  the far hind leg, and its fuzzy landmarks (shoulder, hip) drifted to 113 px.
- **200** — Round 1 (nine parts, buried rims) scored 0.964; the welded tube with bridged
  limb sockets scored 0.972 with zero intersections and zero buried triangles. Left: a
  pointed fox-like muzzle, a narrow six-sided torso, a thin tail, a toe spike.
- **500** — One manifold from the first round; replacing a single nose point by a nose-pad
  triangle and freezing the face interior removed a refiner-made cavity and spike. Left: a
  shelf across the face, a crease at the neck, short ears, 95 intersecting pairs.
- **1000** — Round 1 was ten interpenetrating parts with 161 buried triangles (16 % of the
  budget) and lost to the control on every number. The fused version removed the buried
  triangles and the steps and now edges the control on landmarks and boundary. Left: a head
  with little structure, a tail-root spike, small dents on the hidden flank.

## 6. Independent visual critique

Two blind reviews by fresh critics that saw only the reference and neutral-named renders,
never scores: [round 1](outputs/critic/round_1/CRITIQUE.md) (first-round meshes) and
[round 2](outputs/critic/round_2/CRITIQUE.md) (the finals).

Round 1's three top changes were: make each body one continuous surface, rebuild the heads,
and give the models taller pointed ears and round tails. The first was carried out at every
budget. The second and third were under way when work stopped.

Round 2, on the delivered meshes, said:

- Shared by all four: **the bodies are thin slabs** from the front, back and top, and
  **every head is tilted up and turned away** instead of looking into the camera with a
  protruding muzzle.
- 100: generic origami cat; flat axe-shaped tail, box head, wedge hind legs.
- 200: recognisably a cat; faceless upturned head on a visible collar step, spiky paws.
- 500: good paws; a visor-like ledge across the face, a crease splitting head from body.
- 1000: best outline; a featureless blob of a head, a spike at the tail root, dents on the
  hidden flank.
- Across budgets: outline fit, tail roundness, paws and leg smoothness improve with
  triangles; torso width, head direction and the missing muzzle do not improve at all; the
  500 face is worse than the 200 face and the 1000 head has less structure than the 500
  head. **Best likeness per triangle: the 200.**

None of the round-2 items was acted on, because iteration had already been stopped.

## 7. Inferred geometry

Everything not visible in the one photo is inferred and unverified: the entire left side,
the torso's width, the belly from below, the head's depth, stance width, the left legs'
upper parts, ear thickness and the tail's lateral position. All four budgets take these from
the same proxy so they approximate the same solid.

The proxy's torso was fitted at 0.54-0.60 of its height in width (about 9-10 cm wide), the
low end of the prior it was given. The second critic read the resulting bodies as
implausibly thin. That is a limitation of the inferred prior that every budget inherited;
the photo cannot confirm or refute it.

## 8. Failures, limitations and unfinished work

- **Heads.** Four separately improvised heads were the weakest part at every budget, and
  the refiner folds a face when it chases landmarks without symmetry. A shared, bilaterally
  symmetric head kit at five levels of detail was built ([head/](head/)), and a depth
  mismatch against the proxy (about 4-5 cm along the viewing rays, a single-view
  scale/depth ambiguity) was found and corrected by scaling the kit about the camera centre.
  Kit-head checkpoints exist but are **not** the delivered meshes:
  - 1000 `v12_kit_aligned` (officially scored): a head that reads as a cat facing the
    camera, but IoU 0.979 vs 0.984, landmarks 2.8 vs 1.6 px, head-window IoU 0.955 vs 0.988.
  - 500 `v6`-`v8_kit`: better-looking face, pre-correction kit, IoU 0.971-0.976.
  - 200 `v8_kitL2`: fixes the fox muzzle, pre-correction kit, IoU 0.968.
  - 100 `v4_kit`: the 22-triangle kit head read flatter than the mesh's own, IoU 0.938.
- **Tails** are flat from the front and back at 100 and 200; the refiner flattens them.
- **Self-intersections:** 95 pairs at 500 (see section 5).
- **Proxy mismatches:** its rear-thigh outline sits 20-25 px outside the mask and its torso
  is narrow.
- **Camera assumptions:** focal length and scale are conventions, not measurements.
- **One subject.** This was done for one cat from one photo. Nothing here shows the method
  transfers to another animal or object without a new proxy template.
- **First-round design error:** every budget worker initially built interpenetrating parts.
  That cost a full round at 100, 200 and 1000.
- **Process notes:** the 1000 worker's last run overwrote the build snapshots of the
  accepted mesh, so the film's 1000 timelapse was regenerated by re-running the recorded
  refinement (reproduced to 5e-8 m). The 200 worker found and fixed a non-manifold edge in
  three early checkpoints, none of them delivered.

## 9. Deliverables

| What | Where |
|---|---|
| Four meshes, `.blend` + `.glb` + `mesh.json` | `variants/100/final/`, `variants/200/final/`, `variants/500/final/`, `variants/1000/final/` |
| Per-budget reference-view clay, wire, silhouette, colour, overlay; eight novel views; two turntables | `variants/<N>/final/renders/` |
| Why each checkpoint was chosen | `variants/<N>/final/CHOSEN.json`, `variants/<N>/NOTES.md` |
| Comparison file with `CAT_100` … `CAT_1000` collections | [outputs/comparison.blend](outputs/comparison.blend) |
| Reference plus four | [outputs/comparison_reference_plus_four.png](outputs/comparison_reference_plus_four.png) |
| Clay / wire / silhouette-difference sheet with counts | [outputs/comparison_clay_wire.png](outputs/comparison_clay_wire.png) |
| Novel views, all budgets | [outputs/comparison_multiview_clay.png](outputs/comparison_multiview_clay.png), [outputs/comparison_multiview_wire.png](outputs/comparison_multiview_wire.png) |
| Synchronised turntables | [outputs/turntable_all_clay_wire.mp4](outputs/turntable_all_clay_wire.mp4), `outputs/turntables/` |
| Metrics, initial vs final, controls, topology, allocation | [outputs/metrics.json](outputs/metrics.json), [outputs/metrics.csv](outputs/metrics.csv) |
| Critic reviews | [outputs/critic/round_1/CRITIQUE.md](outputs/critic/round_1/CRITIQUE.md), [outputs/critic/round_2/CRITIQUE.md](outputs/critic/round_2/CRITIQUE.md) |
| Decimation controls | `baselines/decimate/<N>/`, `baselines/decimate_refined/<N>/` |
| Release film and build timelapses | `release_video/out/` (see [release_video/README.md](release_video/README.md)) |
| How to rerun | [README.md](README.md), [scripts/reproduce.sh](scripts/reproduce.sh) |

The release film was added at the user's request during the run. It is a simple visual cut
(no statistics on screen), with the build timelapse drawn over the photo from real
construction stages and optimiser snapshots, and a soundtrack generated with ElevenLabs
Music through fal.
