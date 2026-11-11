# Independent visual critique — round 1

Reviewer: fresh `visual-critic` subagent (Opus 5.5 Medium). It saw only the images in this
folder (`PACKET.md` lists them): the reference photo and, per budget, clay / wire / flat
colour / outline-on-photo at the reference camera plus eight other viewpoints. It saw no
scores, checkpoint names or implementation notes. The photo shows only the cat's right
side, so its remarks about the left side, back, top and depth are plausibility judgments.

Checkpoints reviewed: 100 `v2_hrflip`, 200 `v3_tailbase`, 500 `v3_nose`,
1000 `v1_roots_refined` (the 1000 model was already being rebuilt as one fused surface
when this review came back).

## 100

Verdict: reads as a generic standing cat, not this cat. First wrong thing: the head, a
flat-topped vertical box fused into the chest with no neck, face or muzzle.

- Head and chest (high): one tall slab from ears to front legs; outline runs straight down
  the right edge and misses the cheek bulge and chin. Fix: pull 2-3 vertices forward for a
  muzzle point and chin notch, pinch a neck at chin height; take triangles from the flat
  torso side.
- Ears (medium): thin spikes on the box corners, too narrow at the base, tipped outward.
  Fix: widen each base to about a third of head width.
- Front legs (medium): straight sticks with no paws; near leg slightly too narrow. Fix:
  flare the bottom ring forward.
- Far hind leg (medium): lower leg is a separate wedge with a backward kink; looks broken
  in side views. Fix: align with the photo's hock angle.
- Tail (low): straight segment plus a flat blade tip; flat fin from behind. Fix: round
  cross-section.
- Wire: torso side is a few huge triangles (acceptable here); the flat head front spends
  triangles that should go to the face.
- Colour (not geometry): face is a dark brown patch where the photo is pale.

## 200

Verdict: recognisable as a cat. First wrong thing: the head, an inverted shield with a
downward chin point that looks fox- or owl-like.

- Neck junction (high): a large flat flap sticks out left of the head at chest height.
  Fix: weld it back onto the chest surface.
- Head (high): too narrow at the cheeks, too pointed at the chin; the photo's face is wide
  and round with a short muzzle. Fix: cheeks outward, blunt the chin.
- Torso (medium): flat-topped box with a hard ridge along the back, plus a hip spike. Fix:
  round the back, collapse the spike.
- Legs and paws (medium): no paws; legs end in points; front legs look like stilts. Fix:
  square off the feet with a forward flare.
- Tail (low-medium): segmented blade with a kink near the base; flat from front/back.
- Wire: long slivers along the back, a crowded fan on the front leg, wasted flap triangles.
- Colour: pale face block too large; no eye or nose colours.

## 500

Verdict: the closest to the cat; outline fits tightly everywhere. First wrong thing: a
horizontal shelf across the face at mouth level, reading as a box with an open mouth.

- Face (high): deep horizontal step across the muzzle; in front and three-quarter-left
  views a dark cavity under the face / on the left of the neck (hole or inward fold). Fix:
  make the muzzle one convex wedge, close the cavity.
- Chin and neck (medium): hard step under the chin. Fix: blend into the chest.
- Ears (medium): blunt and short; the photo's are tall and sharply pointed.
- Chest (low-medium): vertical crease where the head joins the shoulder.
- Paws (low): good, separate and in the right places.
- Tail (low): good arc but flat at the end; plank-like from behind (depth is a guess).
- Wire: density is balanced (paws and face dense, torso sparse); small facets at ear roots.
- Colour: eyes almost invisible, nose a tiny sliver.

## 1000 (pre-fusion checkpoint)

Verdict: outline as good as 500's or slightly better, but the inside looks like separate
boxes stacked together; reads less like this cat than 500.

- Hind thigh (high): rectangular slab overlapping the torso with hard step edges.
- Chest and neck (high): overhanging collar/bib block, step under the head.
- Near front leg (high): upper leg is a separate plate in front of the chest, with a stray
  fragment.
- Head (medium): a box with a chin ledge; turned too far toward the camera.
- Ears (medium): still short and rounded.
- Tail (low): smooth arc, tip is a flat cut.
- Wire: face very dense while torso and thigh slabs are coarse; overlapping and buried
  parts waste triangles; seams visible.
- Colour: eyes and nose readable but the eye patch is too big and smeared.

## Across budgets

- 100 → 200: small gain; the 200 head is arguably worse than the 100 box.
- 200 → 500: almost everything improves (outline fit, paws, tail curve, torso roundness,
  leg proportions).
- 500 → 1000: only tail smoothness and face detail improve; the outline barely changes and
  how the pieces join gets clearly worse, so 1000 looks worse than 500 in that respect.
- Not improved by any budget: ears (short and blunt from 500 up), tails flat in front/back
  views at every budget; paws only read from 500.

Three changes that would help most:

1. Make the body one continuous surface (weld thigh, shoulder and chest slabs into the
   torso; remove flaps, collars and stray fragments) — most important for 1000 and 200.
2. Rebuild the head as a round skull with one short convex muzzle wedge and a chin that
   blends into the neck, no face shelf or chin point, using triangles from flat torso sides.
3. Taller, pointed ears with wider bases, and a round tail cross-section.

## What the orchestrator did with it

- (1) was already under way for 1000 (fused to one closed surface in round 2) and was
  ordered for 100 and 200; 500 was already one manifold.
- (2) and the ear part of (3) became a shared, bilaterally symmetric head kit at five
  levels of detail (`head/`), fitted once to the photo and welded onto every budget's body,
  instead of four improvised heads.
- Tail cross-section and the per-budget items were sent to each budget worker as concrete
  geometric edits. Suggestions were budget-neutral; none involved textures or uncounted
  geometry.
