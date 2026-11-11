# Budget 500: notes (worker log)

All triangle counts are exact `len(triangles)` values, and every official evaluation agreed on A=B=C=D=500.
"Official" means `scripts/evaluate.py` (Blender silhouette through the locked camera).
"Unofficial" means the before/after numbers printed by `scripts/refine_vertices.py` (catlib.raster).

## Topology (chosen: config D = A + modelled face + fixes)

Everything is one closed manifold: 1 component, 0 boundary edges, 0 non-manifold edges.
Legs, tail, neck and ears are welded into holes cut in the parent tube. Nothing is buried or capped inside another part.

| Region | Tris | Construction |
|---|---|---|
| torso | 98 | 8-sided main tube, rings D(rump dome) R0..R5 + chest-bend ring C. Four 1-col x 2-band leg holes (lower right/left columns) and a 1x2 tail hole on the top of the rump are removed. The rump is closed with a 6-tri fan. |
| neck | 32 | C -> N1 band (16) + N1 -> 8-loop hole on the head underside (16) |
| head | 69 | 8-sided x 4 rings along the proxy head axis (turned to camera), 8-tri apex back cap, 3-column neck hole, 2 ear holes. Face patch: rim = last ring, interior = 2 eye triangles (3 own verts each), nose-pad triangle (3 verts), mouth/chin vertex, Delaunay in the face plane (26 tris) |
| eyes | 2 | one `eye` triangle per eye, landmark = centroid |
| nose | 1 | nose-pad triangle, landmark = centroid |
| ears | 24 | 2 x (base quad -> narrower 4-vert mid ring with the front pair pushed back (cup) -> apex). The inner faces are `ear_inner` |
| leg_FR / leg_FL | 52 / 52 | 6-sided. Hole loop -> elbow (z 0.075) -> wrist (z 0.045) -> ankle -> toe ring, then a 4-tri toe fan. The lowest 2 verts of the ankle and toe rings sit at z = 0, giving a flat sole quad |
| leg_HR / leg_HL | 52 / 52 | 6-sided. Hole loop (haunch) -> lower thigh (z 0.088) -> hock -> heel -> toe, then a 4-tri fan with a flat sole |
| tail | 66 | 6-sided x 5 rings (t 0.16..0.88) + 6-tri apex tip, alternate stripe bands |
| **total** | **500** | |

## Allocation candidates (same short refine: 6000 iters, seed 1, unofficial)

| Candidate | Layout | IoU | bnd px | lm px | nv IoU | chamfer mm | Looks |
|---|---|---|---|---|---|---|---|
| **A** | 8-sided torso x 7 + 2 neck rings. Head 8 x 5 rings + muzzle cone (head+eyes+nose 70). Legs 6-sided x 4 rings (52 each). Tail 6 x 5 (66) | 0.9724 | 2.36 | 6.89 | 0.954 | 3.83 | rounder flank/back in novel views |
| B | 6-sided torso x 8 + neck. Head 10-sided x 5 (head+eyes+nose 90). Front toes apex. Torso+neck only 108 | 0.9727 | 2.29 | 8.99 | 0.953 | 4.44 | boxy hexagonal flank, top and side_L look slab-like |

A was picked. Silhouette numbers are equal, A has better landmarks and proxy chamfer, and its 8-sided torso reads round from the hidden side.
A 10-sided head (B) did not visibly help the face, because the face problem was topological (see v1 below).
The baselines were not tried as a design, only as controls.

## Checkpoints (official)

| tag | IoU | bnd mean / HD95 px | sharp lm px | fuzzy px | legs_gap | head / tail / legs / torso | nv IoU | chamfer mm | self-int pairs | hidden |
|---|---|---|---|---|---|---|---|---|---|---|
| v0_initial (A, unrefined) | 0.9158 | 7.70 / 27.81 | 22.83 | 14.64 | 0.894 | .921/.887/.864/.965 | 0.9185 | 4.90 | 28 | 5 |
| v1_refined (A, 40k iters + flips) | 0.9821 | 1.48 / 5.10 | 2.19 | 29.84 | 0.975 | .984/.952/.967/.996 | 0.9608 | 3.24 | 48 | 4 |
| v2_face (C: face patch, 40k + flips) | 0.9819 | 1.49 / 5.39 | 2.30 | 35.65 | 0.976 | .980/.951/.968/.996 | 0.9627 | 3.38 | 76 | 2 |
| v3_nose_init (D, unrefined) | 0.9144 | 7.83 / 27.56 | 14.08 | 14.29 | 0.894 | .921/.881/.864/.963 | 0.9135 | 5.00 | 25 | 4 |
| **v3_nose (D, 40k + flips, face interior frozen)** | **0.9830** | **1.40 / 5.00** | 2.76 | 24.19 | **0.977** | .983/.964/.968/.995 | 0.9612 | 3.50 | 95 | 0 |
| v4_reg (D, w_sdf 0.4, w_nv 1, w_reg 0.06) | 0.9793 | 1.72 / 6.00 | 9.14 | 23.04 | 0.971 | .979/.953/.961/.994 | 0.9633 | 3.36 | 79 | 0 |
| control: decimate/500 | 0.9418 | 5.03 / 15.00 | 6.44 | 19.59 | 0.927 | .956/.923/.903/.980 | 0.9754 | 2.22 | 0 | 0 |
| control: decimate_refined/500 | 0.9826 | 1.43 / 5.00 | 2.05 | 22.01 | 0.974 | .983/.966/.965/.995 | 0.9744 | 2.47 | 3 | 0 |

What each step changed:

- **v1:** Refinement fixed the silhouette, but the head was broken.
  The eyes and nose sat on the side bands and the muzzle cone of a tube head. Pulling them to the landmarks folded the face into a deep cavity: eye_L was occluded and a ledge-like muzzle block formed. This is a better score with a worse shape, so it was rejected on visual grounds.
- **v2:** The last head band and the muzzle cone were replaced by a modelled face patch: forehead vertex, eye triangles, a single nose tip and a mouth vertex.
  The cavity is gone. But the optimiser pulled the nose tip into a forward/down "beak" spike, visible in side_L and threeq_front_L. Nose faces were long slivers.
- **v3:** The forehead vertex and nose tip were replaced by a 3-vertex nose-pad triangle. To hit 500, the rump apex (8) became a fan (6).
  Face-interior vertices are frozen during refinement (`freeze_verts.json`), so the eyes and nose stay where the proxy put them. The rest moves.
  The tail's first ring moved off the rump and the chest-to-neck bend became gentler, which took the unrefined self-intersections from 28 to 25.
  Result: no beak, the face reads (brow, eyes, small pink nose, muzzle), and this is the best silhouette.
  Sharp landmarks are 2.76 px vs 2.19 px in v1, because the frozen eyes sit 2-4 px off.
- **v4:** Stronger proxy/quality weights. Every reference metric got worse, and self-intersections only fell to 79. Rejected.

## Honest comparison with refined decimation

v3_nose is equal or slightly ahead of decimate_refined on IoU (0.9830 vs 0.9826), boundary mean (1.40 vs 1.43), legs_gap (0.977 vs 0.974) and the legs window (.968 vs .965).

Decimation wins on:

- sharp landmarks: 2.05 vs 2.76 px
- tail window: .966 vs .964
- novel-view IoU vs proxy: 0.974 vs 0.961
- chamfer: 2.47 vs 3.50 mm
- self-intersections: 3 vs 95

In other words, its hidden side follows the proxy more closely.

Mine has the intended structure: clean 6-sided limb loops welded into shoulder/haunch, ears with thickness and an inner cup, a face with eye, nose and muzzle faces, flat soles, and an anatomical region and material per face.

## Disclosures

- v3_nose is 1 component, closed manifold, 0 hidden triangles and 0 flat-redundant vertices. It has **95 self-intersecting triangle pairs**:
  - 39 head/neck: the jaw rim overlaps the neck-to-head bridge
  - 16 tail/torso: the tail root band crosses the rump surface
  - 14 head/torso
  - 11 neck/torso: chest bend vs neck ring
  - 8 ears/ears: the ear_R mid ring is folded, and ear_R shows a small curl in the front view
  - plus a few within-part pairs

  These are refinement-induced folds at the welded junctions (the unrefined mesh has 25). Refine_vertices has no intersection constraint.
- Fuzzy landmarks are attached to faces and not optimised, so they drift (24 px; the controls show 20-22).
- A visible "brow ledge" crosses the face at eye level, because the frozen eye vertices sit slightly inside the refined rim.

## Rebuild

```
.venv/bin/python variants/500/build.py --config D --out variants/500/checkpoints/v3_nose_init/mesh.json
.venv/bin/python scripts/refine_vertices.py --mesh variants/500/checkpoints/v3_nose_init/mesh.json \
    --out variants/500/checkpoints/v3_nose/mesh.json --iters 40000 --flips --seed 1 \
    --freeze-verts variants/500/checkpoints/v3_nose_init/freeze_verts.json
.venv/bin/python scripts/evaluate.py --mesh variants/500/checkpoints/v3_nose/mesh.json --views
```

Earlier checkpoints:

- v0/v1: `--config A`, refined without freeze.
- v2: `--config C` into `v2_face_init`, then refined without freeze.

Helpers (CPU only, not official):

- `variants/500/preview.py`: raster preview
- `variants/500/mosaic.py`: render tiles
- `variants/500/summ.py`: metric table

## With 20 % more triangles (~100)

- Fix the junctions instead of adding lumps:
  - an extra neck ring plus a 4-column head-underside hole (~+24), so the jaw no longer overlaps the bridge
  - a second rump ring around the tail root (+16)
- Give the head a 5th ring between the ear band and the face rim, to remove the brow ledge (+16).
- Give each eye a split pupil triangle (+4).
- Give each near-side leg upper segment 7 sides for the thigh/forearm outline (~+20).

## Round 2, part 1: v5_body (body prepared for the shared head kit)

`build.py --config E` builds the body plus the round-1 head (494 triangles; used for testing only).
`--config E_body` builds the body alone (382 triangles), ending in an open 8-ring `neck_ring`.

Changes to the body:

| Region | v3 | E body | Change |
|---|---|---|---|
| Neck | C→N1 (16) + weld (16) | C→N0→N1 (32) + kit weld | Extra neck ring; rings moved forward and up so the nape no longer folds into the withers |
| Front legs | 52 / 52 | FR 40 / FL 36 | Wrist ring removed (outline is straight); FL is 5-sided below the elbow |
| Hind legs | 52 / 52 | HR 52 / HL 46 | HL is 5-sided below the thigh |
| Tail | 66 (hole on top of rump) | 76 | Root ring sits just outside a 1-band hole on the rear-top; 6 rings; rounder tip; rings re-rounded after refinement |
| Torso | 98 | 100 | — |

Head allowance for the kit: 500 − 382 = 118 triangles, including the bridge.

Refinement guard (`refine_guarded.py`, chunked mode):

- 16 refine chunks of 2500 proposals each at resolution 768, with the face frozen.
- After each chunk, the vertices of any self-intersecting pair are blended back toward the last accepted state.
- Then the tail rings are re-rounded (camera-plane extent kept) and repaired.
- Self-intersecting pairs go from 22 (unrefined, all at the round-1 head weld) through about 60 per raw chunk to **14 in the final mesh**. The validator count agrees: 6 tail/torso at the tail root, 1 tail/tail, 7 at the temporary head/neck weld.
- Without the guard, the same refinement produced 114 pairs.

Official results:

| tag | tris | IoU | bnd / HD95 | lm | gap | nv | chamfer | pairs |
|---|---|---|---|---|---|---|---|---|
| v3_nose | 500 | 0.9830 | 1.40 / 5.00 | 2.76 | 0.977 | 0.961 | 3.50 | 95 |
| v5_body (494, round-1 head) | 494 | 0.9812 | 1.55 / 5.00 | 4.39 | 0.973 | 0.957 | 3.67 | 14 |

Interpretation:

- The silhouette is slightly worse: the guard reverts moves, and the legs now have fewer rings.
- Intersections dropped from 95 to 14. The tail is round in the back and top views. There is no nape fold.
- The jaw pocket remains; it belongs to the temporary head weld.
- Landmarks: the largest errors are tail_base 17.8 px (the root ring now sits higher on the rump than the photo's tail/rump junction) and paw_HL 9.3 px.

`body/body_v5_refined.json` is the refined body only: 382 triangles, with the same vertex ids as the first 196 vertices of v5_body. It carries `neck_ring` = [72..79].
