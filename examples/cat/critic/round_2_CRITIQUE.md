# Independent visual critique — round 2 (the final meshes)

Reviewer: a fresh `visual-critic` subagent (Opus 5.5 Medium). It saw only the images in
this folder (`PACKET.md` lists them): the reference photo and, per budget, clay / wire /
flat colour / outline-on-photo at the reference camera plus eight other viewpoints. It saw
no scores, checkpoint names or implementation notes. The photo shows only the cat's right
side, so remarks about the left side, back, top and depth are plausibility judgments.

Meshes reviewed (the finals): 100 `v3_welded`, 200 `v5u_untangled`, 500 `v3_nose`,
1000 `v7_body`.

## Shared by all four (high)

- In the front, back and top views every model is a thin slab: the body is about a third
  as wide as a real cat's. The photo cannot show depth, but this is anatomically
  implausible, and it is why every head and chest looks pinched.
- In the reference clay every head is tilted up and turned away, so the face points at the
  sky. In the photo the cat looks straight into the camera; we should see a flat face with
  a muzzle that sticks out.

## 100

Verdict: reads as a generic origami cat, not this cat. First wrong thing: the flat,
axe-shaped tail.

- Tail (high): flat blade with one sharp kink at its base; pointed tip where the photo's is
  rounded. Fix: a thin round tube following the photo's S-curve.
- Torso width (high): slab. Fix: push side vertices out until chest and hips are about
  0.5-0.6 of body height.
- Head (high): a box with no muzzle; chin sits low into the chest. Fix: nose-tip vertex
  toward the camera, raise the chin.
- Legs (medium): back legs are wedge-shaped flaps, no paws. Fix: narrow the tops, flare the
  bottom vertices forward.
- Chest (medium): front edge bulges past the photo outline.
- Surface: one continuous surface; large facets crease the hip.
- Wire: the tail blade spends area doing nothing.

## 200

Verdict: recognisably a cat, outline clearly closer than the 100. First wrong thing: the
faceless, upturned head on a visible collar ledge.

- Head/neck (high): a step where head meets neck. Fix: weld the neck ring into the back of
  the skull and rotate the head to face the camera.
- Torso width (high): slab.
- Paws (medium): sharp forward spikes. Fix: blunt into rounded wedges.
- Tail (medium): still flat, with a corner at its base. Fix: round it, move the base higher
  on the rump.
- Back line (low): too flat; lift the hips slightly.
- Surface: one surface, but the head looks like a separate block pushed onto the body.

## 500

Verdict: reads as a cat; the paws are good. First wrong thing: a horizontal ledge across
the face, like a visor or an open box lid.

- Face (high): the muzzle is a flat shelf under a recessed brow; from the front it looks
  like a dark open mouth. Fix: bring the brow forward, round the muzzle into a small
  pyramid, remove the overhang.
- Neck (high): a vertical crease down the shoulder splits head from body.
- Torso width (high): slab.
- Tail base (medium): a step plus a small tongue of geometry where the tail meets the body.
- Ears (low): curled tubes; flatten into forward-facing cupped triangles.
- Surface: reads as parts stuck together at the neck and tail.
- Wire: the tail is evenly dense and the inner-ear triangles are tiny; they would do more on
  the face and chest.

## 1000

Verdict: the best outline match to the photo; only the tail tip and paws drift. First wrong
thing: a round blob of a head with no muzzle or brow.

- Head (high): a featureless sphere. Fix: use the dense head triangles for a brow,
  cheekbones, a protruding muzzle and a nose.
- Torso width (high): slab, less extreme than the others.
- Tail base (medium): a spike sticks up at the tail root (visible in the outline overlay
  and the side view).
- Flank (medium): small dents and creases on the hidden left flank and hip.
- Tail tip (low): a knob; taper it.
- Surface: the most continuous of the four, apart from the dents and the tail-root spike.
- Wire: head and flank are uniformly dense but still shapeless — the clearest waste; the
  legs have more triangles than their straight profile needs.

## Across budgets

- Improves steadily with triangles: outline fit (100 clearly worse than the rest), tail
  roundness, separate paws, smoothness of the hip and leg outline.
- Does not improve at all: torso width (a slab at every budget), head direction, the
  absence of a muzzle.
- Gets worse: the 500 face is worse than the 200 face (visor shelf); the 1000 head has less
  structure than the 500 head (extra triangles became a blob); the 1000 introduces dents and
  a tail-root spike the 200 does not have.
- Best likeness per triangle, in the critic's judgment: **the 200** — most of the outline
  accuracy of the higher budgets for 20-40 % of their triangles, with faults that are a
  seam and a faceless head rather than structural damage.

Three most valuable changes:

1. Widen the torso at every budget so chest, ribs and hips have real width from the front,
   back and top.
2. Rebuild the head to face the camera with a forward muzzle, nose and brow ridge, welded to
   the neck without a step.
3. Make every tail a round tube that tapers to a rounded tip and rises cleanly from the
   rump; spend the triangles saved on the tail and legs on the face.

## Status of these findings

Mesh iteration was stopped by the user before this review ("current forms are enough"), so
none of these items was acted on in the delivered meshes. They are recorded as known
limitations in `REPORT.md`. Items 2 and 3 were already being addressed when work stopped
(the shared head kit in `head/`, kit-head checkpoints, round-tail candidates); item 1 traces
to the inferred proxy, whose torso was fitted narrow.
