# Budget 100 — notes

Owner: budget-100 worker. Every number marked "official" comes from `scripts/evaluate.py`
(Blender render, locked camera). "Unofficial" means the shared refiner's catlib.raster numbers.

## Method

`build.py` resamples the proxy skeleton (`catlib.skeleton`) into polygonal ring chains.
Triangle winding is outward by construction, using one theta convention around n = u x v.
Open rims are pushed inside their host by `bury()`, which checks the generalised winding number
against the spine triangles. The open rims are the leg tops, the tail root, the ear bases and,
in design B, the neck end. These rim vertices are written to `<mesh>.freeze.json` and stay frozen
in `scripts/refine_vertices.py`, so the refiner cannot pull a rim out into a visible hole.
Without the freeze, the first refinement exposed the open leg tops as visible back-faces at the
haunch and shoulder. `preview.py` and `diag.py` are CPU-only diagnostics: a numpy shaded preview
in which back-faces are blue, and a count of visible back-faces in the reference view.

## Candidate allocations (same 4000-iteration refinement, rims frozen; unofficial)

| region | A: continuous spine (chosen) | B: separate head |
|---|---|---|
| torso | 26 = rump fan 2 + 3 bands x 8 (4-sided box rings, phase 45 deg; last band = chest -> neck ring) | 18 = rump fan 2 + 2 bands x 8 (neck ring ends open inside head) |
| head | 12 = neck->head band 8 + muzzle tip 4 (box ring corners = temples / cheeks, tip = nose) | 20 = back tip 5 + 5-gon band 10 + muzzle tip 5 |
| ears | 6 = 2 x open 3-sided pyramid (base buried) | 6 |
| legs | FR 7, FL 7 (3-sided, 1 segment + sole tri), HR 13, HL 13 (hock ring) | FR 7, FL 7, HR 13, HL 7 (straight) |
| tail | 16 = 4-gon root (buried) -> 3-gon (7) + 3-gon band (6) + tip (3) | 22 = 4-gon root + 3 bands + tip |
| total | 100 | 100 |
| IoU / bnd mean / sharp lm / nv IoU / chamfer | 0.914 / 7.52 px / 6.72 px / 0.870 / 7.73 mm | 0.903 / 8.57 px / 9.61 px / 0.859 / 8.87 mm |

I chose A because it wins on every number. It also looks better: the box head with a muzzle-tip
pyramid reads as a cat face turned toward the camera. B's 5-gon head reads as a faceted ball,
and its straight HL leg spends its saved triangles on a longer tail that gains nothing measurable.

Rejected along the way (visual and diagnostic evidence, no official evaluation):
- **Diamond torso (phase 0).** Leg tops at |y| = 0.03 stick out of a diamond's flanks.
  The box section buries them.
- **Neck ring at neck t = 0.25.** The band folded where the head turns toward the camera,
  leaving a degenerate right side of the head.
- **No neck ring, torso ending at t = 0.9.** The front legs were not covered and the chest band folded.
- **Freezing the sole triangles as well.** With A-built soles the sharp landmark mean was 12.2 px.
  With landmark-centred soles (A2) it was 6.06 px, but IoU fell to 0.907 and a large torso fold
  appeared (3796 px of visible back-face). Kept in `work/trial_soles_frozen/`.

## Checkpoints (official)

| tag | what changed | IoU | bnd mean / HD95 px | sharp lm px | fuzzy lm px | legs_gap | head/tail/legs/torso IoU | nv IoU | chamfer mm |
|---|---|---|---|---|---|---|---|---|---|
| v0_initial | design A, unrefined | 0.7693 | 18.86 / 45.5 | 12.21 | 19.5 | 0.641 | .869/.748/.481/.939 | 0.772 | 10.59 |
| v1_refine | refiner 12000 iterations, rims frozen | 0.9226 | 6.87 / 23.0 | 6.93 | 59.3 | 0.868 | .909/.895/.826/.988 | 0.876 | 7.90 |
| v2_hrflip | A2 (soles centred on the proxy paw landmarks) + refine + 2 edge flips on HR's twisted hock->sole band + 6000 more iterations | 0.9233 | 6.80 / 23.0 | 7.16 | 65.8 | 0.870 | .909/.893/.828/.990 | 0.890 | 7.05 |

The v1 HR leg had a folded lower band (about 2400 px of visible back-face at 768 px). The
refiner spun the sole triangle and the hock-to-sole quads folded. Flipping both quad diagonals
removed the fold, and re-refining did not bring it back (diag: no visible back-faces). Against
v1, v2 has slightly better boundary, novel-view IoU and chamfer, and a slightly worse sharp
landmark mean (+0.23 px).

Tried, not kept: v2 refined further with w_bnd 12, w_lm 2 and resolutions 768,1536 (15000
iterations). Unofficial IoU rose from 0.9233 to 0.9269 and boundary fell from 6.80 to 6.46, but
the sharp landmark mean worsened from 7.16 to 9.35 and a torso back-face came back. The metrics
disagree, so it does not count as an improvement and was not given an official evaluation.

## Versus the naive controls (official)

| | IoU | bnd mean | HD95 | sharp lm | legs_gap | head | tail | legs | nv | chamfer | components |
|---|---|---|---|---|---|---|---|---|---|---|---|
| decimate | 0.884 | 10.69 | 38.0 | 24.99 | 0.845 | .836 | .851 | .792 | 0.879 | 5.24 | 1 |
| decimate_refined | **0.935** | **5.72** | **22.0** | 12.32 | **0.921** | .881 | **.918** | **.892** | **0.904** | **5.19** | 1, watertight |
| ours v2_hrflip | 0.923 | 6.80 | 23.0 | **7.16** | 0.870 | **.909** | .893 | .828 | 0.890 | 7.05 | 8, interpenetrating |

The refined decimation beats this design on silhouette, legs and tail. This design beats it on
landmarks (7.2 px against 12.3 px) and the head window. It also has real ears: the decimated cat
has no ear separation, and its head is one wedge with misplaced ear-tip landmarks. The likely
cause of the leg loss is that my 3-sided leg tubes keep their buried tops frozen, so the thigh
and haunch must come from the box torso. The decimation instead lets the hind thighs blend into
the body silhouette. This is an honest loss on legs and tail numbers.

## Topology disclosures (v2_hrflip)

- 8 components: the spine, 4 legs, the tail and 2 ears. They interpenetrate.
- 22 boundary edges, all on buried open rims: 4 leg tops x 3, tail root 4, ear bases 2 x 3.
  The diagnostic shows no visible back-faces in the reference view.
- 63 self-intersecting pairs, all where parts pass through each other. 0 non-manifold edges,
  0 hidden triangles, 0 flat-redundant vertices.

## Remaining mismatches

- The ears are short stubs because most of each pyramid is buried in the flat box skull.
  The photo has large ears.
- The chin (fuzzy landmark) sits about 65 px low. The neck-to-head band carries the jaw line down
  into the chest.
- The haunch, the inner thigh and the HR hock are underfit, giving legs window IoU 0.83.
- The tail root is a wide wedge.
- Seen from above, the torso is a thin box.

## With 20 % more triangles

I would make HR and HL proper thighs (a 4-sided top ring welded into the torso instead of a
buried open tube), give the head a second ring so the skull top can carry the ear bases, and add
one tail band.

## Round 2: welded box-model (`build_welded.py`, checkpoint `v3_welded`)

The orchestrator asked for one closed surface with every part extruded from it. I verified the
proposed count and changed the allocation. `build_welded.py` asserts exactly 100 triangles and a
closed oriented manifold.

| region | triangles | construction |
|---|---|---|
| spine | 40 | rump tip fan 4 + 4 bands × 8 (rings: rump, waist, chest, neck, head; 4-sided box) + muzzle tip fan 4 |
| legs | +30 | FR +6, FL +6 (flank triangle → sole ring + 1-tri sole), HR +12 (hock ring), HL +6 (straight far leg) |
| tail | +14 | top rump-fan triangle → ring → ring → tip |
| ears | +12 | per ear: 2 skull edge splits cut a small corner triangle (+4), replaced by a pyramid to the ear tip (+2) |
| chin | +2 | edge split on the neck ring's front edge |
| chest | +2 | edge split on the chest ring's bottom edge |
| total | 100 | |

The head is lower than in v2: the head ring is dropped 8 mm and its vertical scale is 1.0 instead
of 1.35, so the ears stand as tall pyramids on the skull corners. HL has no hock because 12 per
ear plus a hock would have meant 106 triangles.

What failed on the way:
1. **Legs on the bottom-quad triangles.** The left and right triangles of one belly quad share
   its diagonal, so the two legs of a pair were welded along it and their walls crossed. Moving
   the legs to the lower flank triangles keeps the belly quad between them, which leaves an open
   gap.
2. **The FR leg on the rear-heavy flank triangle twisted.** I flipped that quad's diagonal so FR
   uses the front-heavy triangle.
3. **Twisted prisms from ring matching.** Matching rings by the face normal is ambiguous for a
   vertical flank face. `extrude` now picks the vertex assignment, out of 6, whose walls face
   outward best.
4. **The refiner folds thin leg walls.** `unfold.py` greedily flips edges until no back-face is
   visible in the reference view or the 8 novel views. Each unfold is followed by a fresh
   refinement because a flip can cost silhouette area.
5. **A stronger landmark weight (w_lm 25)** reached 2.7 px but folded the mesh badly. Repairing
   the folds brought IoU down to 0.888, so I rejected it.

The refinement chain (`reproduce_v3.sh`, deterministic) has four runs, with an `unfold` pass
after each of the first three:

| run | iterations | seed | w_lm | flips |
|---|---|---|---|---|
| r1 | 15000 | default | default | yes |
| r2 | 8000 | 2 | default | no |
| r3 | 8000 | 3 | 10 | no |
| r4 | 8000 | 6 | 10 | no |

| tag | IoU | bnd mean / HD95 | sharp lm | fuzzy lm | legs_gap | head/tail/legs/torso | nv IoU | chamfer |
|---|---|---|---|---|---|---|---|---|
| v3_welded | **0.9432** | **4.92 / 18.0** | 8.57 | 112.9 | 0.913 | **.939**/.862/.883/.992 | 0.897 | 6.77 |
| v2_hrflip | 0.9233 | 6.80 / 23.0 | **7.16** | 65.8 | 0.870 | .909/.893/.828/.990 | 0.890 | 7.05 |
| decimate_refined | 0.9352 | 5.72 / 22.0 | 12.32 | 39.5 | **0.921** | .881/**.918**/**.892**/.981 | **0.904** | **5.19** |

**Topology (v3).**
- 1 component, 0 boundary edges, 0 non-manifold edges.
- 3 self-intersecting pairs.
- 1 hidden triangle.
- All 4 counts = 100.

**Choice: v3_welded.** It beats both v2 and the control on IoU, boundary mean and HD95, and has
the best head window. Against v2 it improves legs and leg gaps (0.913 vs 0.870) and replaces 8
interpenetrating components with one closed surface. It has real, separated ears that are
visible from the front. It loses to the control on legs_gap (0.913 vs 0.921), tail and chamfer.
It loses to v2 on the sharp landmark mean (8.57 vs 7.16), but is still ahead of the control
(12.3).

**Visible problems.**
- The tail is a flat 3-sided blade.
- The fuzzy shoulder and hip points drift, because they sit on large flank triangles.
- Seen from above, the body is still narrow.

The v2 head fix (`v4_*`) was not built, because v3 is better in both looks and numbers.

The timelapse (CONTRACT §13) is in `timelapse/stages.json`:
- 9 construction stages.
- 4 real refine runs with snapshots every 250 proposals.
- 3 edge-flip edits.
- The final stage.

The last snapshot matches the checkpoint (max difference 4.9e-8 m, JSON rounding), and the
re-run reproduces v3's vertices and triangles exactly.

## Round 3: shared head kit L0 (`build_kit.py`, checkpoint `v4_kit`)

The spine tube ends in a 4-vertex neck ring whose vertices ARE the kit's neck loop. They are
shared 1:1 with no bridge, so the head costs exactly its 22 triangles. I did not use
`catlib.headkit.attach_head`, because it always adds a bridge (+8 triangles). The kit's
triangles, regions, materials and its 6 `landmark_attach` entries are copied unchanged, with
indices remapped. All 10 non-loop kit vertices are frozen in every refine run. `unfold.py` now
takes the freeze list and never flips an edge that touches the kit.

| region | v4_kit allocation |
|---|---|
| spine | 28 = rump fan 4 + 3 bands × 8 (box rings rump / waist / chest at torso t = 0.78 → kit loop) |
| head kit L0 | 22 |
| legs | 36 = FR 6, FL 6, HR 12, HL 12 (the HL hock is back) |
| tail | 14 |
| total | 100 (the chin and chest splits from v3 are gone; their triangles went to the HL hock) |

With the chest ring at t = 1.0, the chest ring and the kit loop overlapped in x and the neck band
folded badly (29k px of visible back-face). Moving the chest ring to t = 0.78 fixed it.

The refinement chain (`reproduce_v4.sh`, deterministic, re-run reproduces the vertices exactly;
timelapse in `timelapse_v4/`) has five runs, with an `unfold` pass after each of the first four.
The head is frozen throughout.

| run | iterations | seed | w_lm | resolutions |
|---|---|---|---|---|
| r1 | 15000 | default | default | default |
| r2 | 8000 | 2 | default | default |
| r3 | 8000 | 3 | 10 | default |
| r4 | 12000 | 6 | 10 | 384,768,1536 |
| r5 | 12000 | 7 | 10 | 384,768,1536 |

| tag | IoU | bnd / HD95 | sharp lm | legs_gap | head/tail/legs/torso | nv IoU | chamfer | self-int / hidden |
|---|---|---|---|---|---|---|---|---|
| v3_welded | **0.9432** | **4.92 / 18** | 8.57 | 0.913 | **.939**/.862/.883/**.992** | **0.897** | **6.77** | **3** / 1 |
| v4_kit | 0.9382 | 5.41 / 19 | **8.22** | **0.916** | .926/**.867**/**.887**/.986 | 0.863 | 7.02 | 7 / **0** |

**What the views show.**
- The kit's ears are symmetric and clean.
- The v4 face plane is mostly one big facet running from the nose down into the chest. The kit
  has no muzzle volume of its own, and the body's chest band reaches up to the throat vertex.
  v3's face has cheek and muzzle facets and reads more like a face in the reference view.
- The v4 novel views are a little worse: the chest below the head is a thin wedge in side_L,
  hence novel-view IoU 0.863 against 0.897.
- The tail is still a thin 3-sided blade from the top in both versions. Its lateral thickness
  collapses during refinement, because only the reference silhouette and a weak SDF term
  constrain it. This is not fixed.

**Pick: v3_welded stays.** It is better on IoU, boundary, novel-view IoU, chamfer, head IoU and
self-intersections (3 vs 7). v4 wins narrowly on sharp landmarks, legs_gap and the tail and legs
windows, and has the HL hock and 0 hidden triangles. v4 remains a valid A/B alternative.
