# Lessons

What went wrong in the cat run and the rule each failure became. One photo of an orange tabby, budgets 100,
200, 500 and 1000, one worker per budget, two full rounds and part of a third before the user stopped mesh
iteration. Every number is from that run's evidence in `examples/cat/` (`metrics/`, each budget's
`NOTES.md`, `critic/`, `REPORT.md`).

## Structure

- **Interpenetrating kits waste triangles and show their joins.** In round 1 every worker built closed parts
  (torso, legs, tail, ears) and pushed them into each other, or left open tubes with their rims buried. At
  1000, 161 triangles (16 % of the budget) were buried inside other parts, the joins read as steps and slabs,
  and the mesh lost to plain decimation on every number. At 100 the leg tops stayed frozen inside the body
  and the thighs could not form. Rule: **one closed surface; weld limbs, tail and ears into the body**
  (extrude from body faces, bridge ring-to-ring into sockets, cut holes and weld, or a boolean union followed
  by edge collapses to exactly N). After welding: 100 went from IoU 0.923 to 0.943 and its leg gap from 0.870
  to 0.913; 1000 went from 161 buried triangles to 1.
- **Count by construction.** A closed genus-0 surface has T = 2V - 4, so "exactly 200 triangles" is "exactly
  102 vertices", and a closed loft with ring sizes n_i has 2·Σn_i triangles. Build scripts assert the count;
  no padding is ever added to reach it.

## The optimiser

- **The refiner folds faces.** Pulling eyes and nose to their landmarks with no symmetry folded the 500 face
  into a cavity and then pulled a beak-like spike; at 1000 it made a fold between the right eye and the nose.
  Each time the score improved. Rule: freeze the face interior (or the whole head) during refinement and let
  only the body move; a better score with a worse shape is a regression.
- **The refiner creates self-intersections.** It has no intersection term. The 500 final went from 25
  intersecting pairs unrefined to 95 refined, all at welded junctions. Guards that help: revert the vertices
  of intersecting pairs after each refinement chunk (500 body: 114 down to 14 pairs), a targeted relax
  (1000: down to 4 pairs; relaxing to 0 cost IoU 0.9844 to 0.9805 because thin legs shrank), an untangle pass
  (200: 0 pairs). Report the count either way.
- **It flattens thin parts.** Tails collapsed to blades at 100 and 200: only the reference silhouette and a
  weak SDF term constrain their width. Leg walls folded at 100 and needed an unfold pass after each run.
- **The optimiser is most of the gain, for everyone.** The control (decimation plus the same refiner) tied the
  designed meshes at 500 and 1000. Hand-placed topology bought readable parts more than numbers.

## Scores

- **A score alone passes wrong shapes.** Every failure above improved at least one number. Rule: a result is
  better only when IoU, boundary distance, landmarks and leg gaps agree and the clay and eight views look
  better. A stronger landmark weight at 100 reached 2.7 px and folded the mesh; repairing the folds dropped
  IoU to 0.888, so it was rejected.
- **Score the negative space.** The IoU of the background inside the legs window says whether the gaps
  between the legs are open; a global IoU can hide merged legs.
- **Agreement with the proxy is not truth.** Novel-view IoU and Chamfer distance against the inferred proxy
  favour the decimation control (it is made from the proxy) and say nothing about whether the hidden side is
  right.

## Heads

- **Four improvised heads were the weakest part at every budget.** The blind reviewer called them a box, a
  fox-like shield, a shelf across the face and a box with a chin ledge in round 1, and in round 2 (the
  finals) a box with no muzzle, a faceless head on a collar ledge, a visor-like shelf and a featureless
  sphere. Rule: design the head once, symmetric, at several levels of detail (the shared kit: 22, 32, 50, 120
  and 224 triangles), fit it to the photo once with symmetry preserved, weld one level onto each body and keep
  it rigid.
- **Depth-align about the camera centre.** One photo cannot tell a smaller head near the camera from a larger
  one farther away; the photo fit put the kit 3-5 cm too close, off the proxy the bodies were built on (mean
  |SDF| 29.7 mm). Scaling every kit vertex about the camera centre, X' = C + s(X - C), with one s (1.029)
  moved it along the viewing rays: 0 px change in the photo, mean |SDF| 4.2 mm. Kit-head checkpoints scored
  slightly lower in the reference view than the folded but photo-fitted heads; the accepted finals kept
  their original heads when iteration stopped. This is unfinished.

## The hidden side

- **A narrow prior gives a slab.** The proxy torso was fitted at the narrow end of its width prior (about
  0.55 of its height). The second reviewer read every final as a thin slab from the front and top. Every
  budget inherited it, because all of them follow the proxy. Rule: look at the proxy from the front and the
  top before any budget is built, and say the width is a guess.

## Process

- **Freeze the targets before scoring.** Mask, landmarks, windows, camera, scene, proxy and evaluation code
  were hashed before any budget was scored; every final carries identical hashes, and the aggregation
  refuses rows that do not.
- **Keep the build reproducible.** Construction scripts are deterministic and refinement is seeded; the
  timelapse re-runs the recorded commands with snapshots and checks that the last snapshot reproduces the
  checkpoint (to 5e-8 m). One worker's last run overwrote the snapshots of the accepted mesh; they had to be
  regenerated from the recorded command.
- **A reviewer who has not seen the work.** The critic sees the photo and neutral-named clay, wire and outline
  renders, never scores or names. Its items become budget-neutral geometric edits; texture, card or
  uncounted-geometry suggestions are rejected.
