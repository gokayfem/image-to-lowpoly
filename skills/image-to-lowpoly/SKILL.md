---
name: image-to-lowpoly
description: >-
  Turns one reference photo of a subject (an animal, a character, an object) into editable low-poly 3D meshes
  under hard triangle budgets (for example 100, 200, 500 and 1000), each matched to the photo through one
  locked camera: fixed targets made once (mask, landmarks, image windows, all hashed), a camera fitted jointly
  with a parametric proxy that supplies the inferred hidden side, a build script per budget that allocates
  triangles by region and welds every part into one closed surface, a shared seeded vertex optimiser that
  never changes the count, a four-way triangle count (mesh file, evaluated .blend, re-imported GLB, parsed
  GLB), official metrics from a Blender render (silhouette IoU, boundary distance, landmark error, leg-gap
  IoU), clay and wireframe renders from eight angles, a decimation control at every budget and a blind visual
  reviewer. Ships the tools (`scripts/`) and a worked cat case. Use when the user wants a low-poly model of a
  photo "in N triangles", several triangle budgets of the same subject, a check that a low-poly mesh really has
  N triangles, or evidence of where extra triangles help. Not for textured or photoreal models, image-to-3D
  services, several photos of one subject, animation or rigging, or scenes with many objects (use
  blockout-to-real for materials and light on an existing scene).
license: MIT
metadata:
  hermes:
    tags:
      - 3d
      - blender
      - low-poly
      - mesh
      - triangle-budget
      - single-view-reconstruction
    category: graphics
---

# Image 2 Low Poly

One photo in, one editable low-poly mesh per triangle budget out, each scored against the photo through the
same locked camera and counted four independent ways. The low-poly geometry itself does the approximation:
no texture, billboard, projected photo or uncounted geometry stands in for shape.

Inputs: the reference image (never changed), the subject, the budgets. Worked example: `examples/cat/` in the
repository (one orange tabby, budgets 100 / 200 / 500 / 1000).

## Before starting

| Situation | Do this |
|---|---|
| One photo, whole subject visible, plain background | the main path |
| A four-legged animal | the shipped proxy template (`fit_camera_proxy.py`, `build_proxy.py`) is a quadruped cat; adjust its priors |
| Any other subject | write a proxy template for it first (generalised cylinders per part, `catlib/skeleton.py` format); the rest of the tools are subject-agnostic |
| Several photos, a video, a scan | out of scope: one view, one camera |
| The user wants colour or texture | flat per-face colours only; say so before starting |
| No Blender | not supported: counting, assembly and every official number need Blender 5.2+ |

Say before starting that everything the photo does not show is inferred, not verified.

## Workflow

`S` below means `skills/image-to-lowpoly/scripts`. Every script takes `--case CASE` (or `LOWPOLY_CASE`). Post
a labelled image at every checkpoint (overlays, clay, wire, eight views): images first, numbers second.

1. **Case and reference.** Make `CASE/reference/reference.png` (byte-identical copy of the user's image) and a
   provenance file with its SHA-256. Never regenerate, resize or edit it.
2. **Fixed targets.** Annotate landmarks and image windows by looking at zoomed crops (12 sharp: ear tips,
   eyes, nose, four paw contacts, tail base/mid/tip for a cat; adapt the list to the subject), then
   `python3 $S/preflight_reference.py --case CASE`. **Look at `mask_overlay.png` and `landmarks_overlay.png`.**
   The mask excludes the floor shadow and whiskers.
3. **Camera and proxy.** `fit_camera_proxy.py --case CASE` (CPU, tens of minutes), then `build_proxy.py`,
   `build_proxy.py --lock` and `build_proxy.py --report`. Look at the proxy from all eight views: it is the
   answer for everything hidden. Then freeze: write the hashes of reference, mask, landmarks, regions,
   camera, scene, proxy and the evaluation code to `contract/FROZEN.json` before any budget is scored
   (`references/contract.md`).
4. **Controls.** `make_decimation_baseline.py --case CASE --budget N` for every budget: Decimate of the proxy
   to exactly N, and the same after the shared refiner. These are the numbers to beat.
5. **Allocate and build, per budget.** Give each budget its own worker and directory `variants/<N>/`
   (`references/budget-worker.md`). Write the allocation table (region, triangles, rings x sides, reason),
   build at least two materially different allocations, give each the same short refinement, compare, choose.
   `variants/<N>/build.py` writes a `catmesh-1` `mesh.json` with exactly N triangles (assert it), regions,
   materials and landmark attachments. **One closed surface: weld limbs, tail and ears into the body.**
6. **Initial checkpoint.** `evaluate.py --case CASE --mesh variants/<N>/checkpoints/v0_initial/mesh.json
   --views`. This is the "initial" row.
7. **Refine, measure, look, edit, repeat.** `refine_vertices.py` (`--flips`, `--freeze-verts` for faces and
   any frozen kit), `evaluate.py`, then **look** at `ref_overlay.png`, `ref_clay.png`, `ref_wire.png` and the
   eight views. Make targeted topology edits where the overlay shows the budget misallocated. Keep the best
   validated checkpoint; the newest need not win.
8. **Blind review.** `make_critic_packet.py --case CASE --round K` packs the photo and neutral-named clay,
   wire and outline renders. Start a **fresh subagent** that sees only that folder; turn its findings into
   budget-neutral geometric edits. Reject any suggestion that is a texture, a card or uncounted geometry.
9. **Finalize and aggregate.** `finalize_variant.py --case CASE --budget N --checkpoint TAG --reason "..."`
   copies the checkpoint and re-scores it; `aggregate_metrics.py` builds the cross-budget table and refuses
   mismatched hashes or counts; `make_comparison.py` makes the sheets.
10. **Report and stop.** Present the clay/wire sheet with the four counts, the table of initial, final and
    both controls per budget, the allocation tables, the reviewer's verdicts, and what is wrong. Say where the
    designed mesh does not beat the control. The user decides what next.
11. **Timelapse (optional).** Record construction stages (`--stages-dir` in your build script) and optimiser
    snapshots (`refine_vertices.py --snapshot-dir DIR --snapshot-every K`), then `timelapse/timelapse.py
    --case CASE`: the mesh drawn over the photo through the locked camera.

## Hard rules

- **The count is the evaluated, exported geometry.** Four counts (mesh file, evaluated `.blend` over every
  object, re-imported GLB, parsed GLB) must agree and be at most N; aim for exactly N. Never Blender's polygon
  count (`references/counting.md`).
- **Nothing hides triangles or shape.** Triangles only, one object, no modifiers, no subdivision, curves,
  hair, instances, alpha cards, duplicate or degenerate padding, billboards or flat silhouette extrusions.
- **The target never moves.** Reference, mask, landmarks, windows and camera are made once and hashed. A
  genuine correction is disclosed and every budget is re-scored.
- **No single number decides.** IoU, boundary distance, landmarks and leg gaps must not contradict each other,
  and the clay and eight views must look better. A better score with a worse shape is a regression.
- **Weld, do not interpenetrate.** Buried triangles are wasted budget and the joins show.
- **The optimiser is shared; the allocation is the experiment.** Same refiner, same settings available to
  every budget and to the control.
- **Landmarks move with geometry.** A landmark is attached to the feature it names; never slide the
  attachment to improve a number.
- **One render at a time.** Only `render_variant.py` renders, and it takes the lock itself.
- **Materials never count as shape.** Flat per-face colours with anatomical meaning; rank geometry on clay
  and silhouettes only.
- **Say what it is not.** Inferred hidden side; a ties-the-control result is a tie; heads and thin parts are
  where these meshes are weakest.

## References

| File | Read when |
|---|---|
| `references/contract.md` | setting up a case: layout, units, camera, mesh format, metrics, renders, proxy, landmarks, head kit |
| `references/budget-worker.md` | delegating or doing one budget: procedure, priorities, what to look at |
| `references/counting.md` | the four-way count, hard failures, what is disclosed |
| `references/lessons.md` | what went wrong in the cat run and the rule each failure became |
