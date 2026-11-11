# Budget 1000 — notes (worker: BUDGET-1000)

## Construction
`build.py`: every part of `proxy/skeleton.json` becomes a closed **loft** (rings with
per-ring side counts, unequal neighbours zipped, cone tips at both ends). A closed loft
with ring sizes n_i has exactly 2*sum(n_i) triangles, so the allocation is done in ring
vertices: sum = 500 -> exactly 1000 triangles (asserted). 10 closed interpenetrating
components (torso, neck, head, 2 ears, 4 legs, tail). Ring placement along each part:
`--spacing curv` (density 1 + 3*change, change = centreline bend*radius + |dr/dt|).
Soles: every vertex at z < 1 mm is clamped to z = 0 (flat sole, pinned by the refiner).
`--features v1`: eye = fan of the head vertex nearest the proxy eye landmark, vertex
pushed 2.5 mm in (socket), pupil = fan face under the landmark, eye landmark = socket
vertex; nose = upper half of the muzzle-tip fan, landmark = tip vertex.
`--design v1/v2`: legs, neck and tail start with a **root ring** (scaled copy pulled
towards the torso axis) so they taper into the host instead of ending in caps poking
out of the flank (the v0 stumps).
Helpers: `preview.py` (CPU numpy preview, NOT official), `inside_check.py` (winding
number: which part vertices are buried; writes `freeze.json` for `--freeze-verts`).

## Allocation (triangles = 2 x ring vertices)
| region | head_rich (v0) | body_rich (rejected) | v1 / v2 (chosen) |
|---|---|---|---|
| torso | 240 (10 rings x 12) | 308 (11 x 14) | 176 ([8,12x6,8]) — flat flanks cheap |
| neck/chest | 32 ([8,8]) | 32 | 76 (root 8 + [10,10,10]) — chest/bib is a visible silhouette |
| head+eyes+nose | 212 ([10,14,16,16,16,14,12,8]) | 152 | 200 ([8,14,16,14,14,14,12,8]) |
| ears | 64 ([6,6,4] x2) | 56 | 56 ([5,5,4] x2) |
| front legs | 82 each ([8,8,7,6,6,6]) | 82 | 92 each (root 6 + [8,7,7,6,6,6]) |
| hind legs | 96 each ([8,8,8,6,6,6,6]) | 96 | 106 each (root 6 + [8,8,7,6,6,6,6]) |
| tail | 96 (8 x 6) | 96 | 96 (root 6 + 7 x 6) |

Candidate comparison (same short refinement, 6000 proposals, unofficial raster):
- head_rich: IoU 0.9795, bnd 1.70 px, lm 5.60 px, nv 0.9605, chamfer 4.12 mm
- body_rich: IoU 0.9797, bnd 1.68 px, lm 5.29 px, nv 0.9585, chamfer 4.06 mm
- head_rich, uniform spacing: IoU 0.9794, bnd 1.71, lm 7.67 (curvature spacing kept)
Numbers are a tie; visually (cand/sheet_A_crop.png) body_rich's 76-vertex head crumples
at the muzzle while head_rich keeps a round skull and clean eye line, and the extra
torso sides of body_rich buy nothing on flat flanks. Chose head-rich.
Refiner with interpenetrating parts: without freezing, the SDF/silhouette terms drag
buried vertices to the surface and produce shards (cand/sheet_v1c.png,
sheet_v1c_reg.png; higher --w-reg 0.2/0.5 did not fix it). Freezing the buried vertices
(`inside_check.py` -> freeze.json, 145-150 verts) removed the shards (sheet_v1c_frz.png).

## Official checkpoints (scripts/evaluate.py, all 4 counts = 1000)
| tag | IoU | bnd mean / HD95 px | sharp lm px | fuzzy | legs_gap | head/tail/legs/torso IoU | nv IoU | chamfer mm | hidden | self-int pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| v0_initial (head_rich, unrefined) | 0.9388 | 5.25 / 18 | 13.50 | 18.9 | 0.930 | .942/.915/.904/.968 | 0.934 | 6.28 | 86 | 314 |
| v1_roots_refined | 0.9834 | 1.36 / 5 | 3.83 | 19.2 | 0.972 | .986/.974/.962/.996 | 0.964 | 5.78 | 161 | 440 |
| v2_shallow_roots (+flips) | 0.9833 | 1.37 / 5 | 5.51 | 28.4 | 0.970 | .989/.973/.960/.997 | 0.966 | 5.76 | 118 | 479 |
| control: decimate/1000 | 0.9480 | 4.51 / 14 | 5.11 | 19.5 | 0.940 | .96/.936/.921/.979 | 0.987 | 2.00 | 0 | 0 |
| control: decimate_refined/1000 | **0.9842** | **1.32 / 5** | **1.64** | 20.5 | **0.979** | .99/.967/.971/.995 | 0.981 | 2.22 | 0 | 6 |

v1: v1 design (roots, chest-neck, eye sockets) + 20000 proposals with buried vertices
frozen. Large gain over v0 on every reference-view number; leg stumps gone.
v2: shallower roots + neck ending inside the head + `--flips`. Hidden triangles 161->118,
IoU equal, but landmarks worse (5.51 px) and a flap appeared on the chest. Kept v1.

## Honest comparison with the control
The refined decimation of the proxy wins on every headline number and has a single
closed component with no buried triangles. My design spends 161 triangles (16 %) on
geometry buried inside other parts (root rings, torso under legs/neck, head back inside
neck) — that is the main misallocation. What my mesh has that the control lacks: clean
ring loops per part, explicit eye sockets / nose / ear-inner faces, flat soles, cheap
flanks; but the visible seams where legs/neck enter the torso read as separate parts.

## Remaining mismatches (v1)
Leg-to-torso and neck-to-torso seams visible as steps; a small fold on the near front
shoulder; chest/neck reads boxy; muzzle protrusion is weak; hidden 161, self-int 440.
With 20 % more triangles (or by recovering the 161 hidden ones): weld legs and neck into
the torso (bridge leg first ring to an opening cut in the torso ring loops) instead of
interpenetrating, which removes both the buried waste and the seams.

## Rebuild
    .venv/bin/python variants/1000/build.py --design v1 --features v1 --out variants/1000/checkpoints/v1_roots_refined/built.json
    .venv/bin/python variants/1000/inside_check.py --mesh variants/1000/checkpoints/v1_roots_refined/built.json
    .venv/bin/python scripts/refine_vertices.py --mesh variants/1000/checkpoints/v1_roots_refined/built.json --out variants/1000/checkpoints/v1_roots_refined/mesh.json --iters 20000 --quiet --freeze-verts variants/1000/checkpoints/v1_roots_refined/freeze.json
    .venv/bin/python scripts/evaluate.py --mesh variants/1000/checkpoints/v1_roots_refined/mesh.json --views
v0: `build.py --design head_rich --spacing curv` (features v0), evaluated unrefined.
v2: as v1 with `--design v2`, refine adds `--flips`.

# Round 2: one fused manifold (route A)

`fuse.py` + `fuse_blender.py`: lofts of design `f1`/`f2` (no root rings; parts simply
overlap) -> background Blender (CPU only, no render, no GPU lock) EXACT boolean union
into the torso, merge-by-distance 1.5 mm, dissolve degenerates, triangulate
(f1: 1124 loft tris -> 1390 after union) -> 194 `catlib.topo.edge_collapse` steps to
exactly 1000, cost = edge length x region weight (union slivers x0.3 first, torso 1,
tail 1.6, neck/legs 2, head 3, ears/eyes/nose never; link condition, no normal turn
> 60 deg, no new edge < 2 mm) -> union-created vertices + 1-ring moved half-way to the
proxy surface (fillets, 265 verts) -> region/material from the nearest source-loft
face (normal-compatible); eyes/nose re-derived on the fused surface around the source
feature vertices; landmarks re-attached per §12 -> shared refiner with --flips, no
freeze list. Exact count from union output, no padding.

## Allocation before/after (triangles by region)
| region | v1 (10 parts) | v6 (fused) |
|---|---|---|
| torso (incl. shoulder/haunch/belly fillets) | 176 | 227 |
| neck/chest | 76 | 76 |
| head + eyes + nose | 200 | 163 |
| ears | 56 | 48 |
| legs FR/FL/HR/HL | 92/92/106/106 | 85/87/100/100 |
| tail | 96 | 114 |
| buried (hidden) | 161 | 5 |
The ~156 recovered triangles went mainly into the junction surfaces (torso count
includes shoulder, haunch and tail-root fillets created by the union) and the tail;
the head lost ~37 because its back half was inside the neck. f2 (torso 14->12 sides,
18-sided cheek ring, +1 ring per leg) was tried as v4.

## Official checkpoints (round 2)
| tag | IoU | bnd mean/HD95 | sharp lm | fuzzy | legs_gap | head/tail/legs/torso | nv IoU | chamfer | comp | hidden | self-int |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v3_fused (f1, 20k flips) | 0.9854 | 1.20/4.0 | 2.02 | 20.6 | 0.9757 | .989/.976/.967/.996 | 0.9735 | 2.55 | 1 | 3 | 25 |
| v4_fused_headshift (f2, 30k flips) | 0.9862 | 1.14/4.0 | 2.48 | 27.3 | 0.9761 | .991/.976/.968/.997 | 0.9714 | 2.59 | 1 | 1 | 26 |
| v5_fused_nv (v3 + 15k, w-nv 1.5) | **0.9860** | 1.15/**3.61** | **2.22** | 21.1 | **0.9767** | .989/.977/.969/.997 | **0.9766** | **2.52** | 1 | 2 | 39 |
| v6_fused_eyes (f1, w-lm 1.5, 3-face eyes) | 0.9854 | 1.20/4.12 | 3.20 | 23.3 | 0.9753 | .990/.978/.967/.996 | 0.9740 | 2.54 | 1 | 5 | 33 |
| control decimate_refined/1000 | 0.9842 | 1.32/5.0 | 1.64 | 20.5 | 0.9787 | .99/.967/.971/.995 | 0.981 | 2.22 | 1 | 0 | 6 |

v4: moving flank triangles to the head/legs helped IoU/boundary but not landmarks or
novel views. v5: novel-view weight pushes nv to 0.9766. v6: v5's face had a fold between
the right eye and the nose and the eye fans (6 large faces each) merged with the nose
patch; v6 uses only the 3 camera-facing socket faces per eye and a gentler landmark
weight (1.5) - two distinct level eyes, clean nose, no facial fold, at the cost of
~1 px landmark error. BEST = v6 (visual), v5 if ranked on numbers.
Remaining: small dark notch at the tail root (side_L), legs region IoU and legs_gap
still below the control, paw landmarks 4-8 px, shoulder_R fuzzy 37 px.

Rebuild v6:
    .venv/bin/python variants/1000/fuse.py --design f1 --out variants/1000/checkpoints/v6_fused_eyes/built.json
    .venv/bin/python scripts/refine_vertices.py --mesh .../built.json --out .../stage1.json --iters 20000 --flips --w-lm 1.5 --quiet
    .venv/bin/python scripts/refine_vertices.py --mesh .../stage1.json --out .../mesh.json --iters 15000 --flips --w-nv 1.5 --w-lm 1.5 --quiet
    .venv/bin/python scripts/evaluate.py --mesh variants/1000/checkpoints/v6_fused_eyes/mesh.json --views
(v3/v5 were fused before the 3-face eye change in fuse.py; v5 = v3 mesh +
`refine_vertices.py --iters 15000 --flips --w-nv 1.5`.)

# Round 3 part 1: v7_body (body polish + head socket)

Changes (`build.py` design `f3`, `fuse.py`):
- `--paw-align`: before the union each lower leg slides (weight 1 at the ground, 0 at
  10 cm) so its sole centre sits on the ground point under the photo's paw landmark
  (shifts FR -11.9/+5.3, FL +10.6/-8.8, HR -6.7/-7.3, HL +1.7/-13.7 mm in x/y);
  paw landmarks now attach to the centre of each paw's ground-footprint vertices.
- tail: start tip 30 mm inside the rump (no notch at the root), dome ring (0.65 scale)
  before the apex (rounded tip); tail_base/tail_mid/fuzzy landmarks attach to the proxy
  feature point on the named region (the loft's tail_base ring is buried by the union).
- `--body-only --target 760`: head and ears left out of the union, the neck loft's top
  tip fan removed -> `checkpoints/v7_body/body.json` (catbody-1, 760 tris, ONE open neck
  ring of 10 vertices, ids [276,277,279,342,282,284,286,288,289,290], centre
  (110.5, 9.8, 256.0) mm, normal ~(0.13,-0.10,-0.99), planar to 0.09 mm rms, mean radius
  43 mm). This body.json is the UNREFINED fused body (refinement happens after the weld).
- `fix_selfint.py` + `selfint_blender.py`: targeted edit relaxing the vertices of
  self-intersecting triangles (Blender BVH test identical to the validator). Full relax
  to 0 pairs cost IoU 0.9844 -> 0.9805 and legs_gap 0.974 -> 0.964 (thin legs shrink),
  so the kept recipe is refine -> relax(0.2) -> short refine (4 pairs left).
- `--stages-dir` + refiner snapshots -> `timelapse/stages.json` (17 stages; last
  snapshots reproduce stage meshes to 5e-8 m).

| tag | IoU | bnd mean/HD95 | sharp lm | legs_gap | tail/legs IoU | nv | chamfer | hidden | self-int |
|---|---|---|---|---|---|---|---|---|---|
| v5_fused_nv | 0.9860 | 1.15/3.61 | 2.22 | 0.9767 | .977/.969 | 0.9766 | 2.52 | 2 | 39 |
| v6_fused_eyes | 0.9854 | 1.20/4.12 | 3.20 | 0.9753 | .978/.967 | 0.9740 | 2.54 | 5 | 33 |
| v7_body (first try, full relax, overwritten) | 0.9805 | 1.61/6.71 | 2.27 | 0.9643 | .972/.952 | 0.9718 | 2.47 | 1 | 0 |
| **v7_body** | 0.9844 | 1.29/5.00 | **1.59** | 0.9742 | .973/.966 | 0.9742 | 2.45 | 1 | 4 |
| control decimate_refined | 0.9842 | 1.32/5.0 | 1.64 | **0.9787** | .967/.971 | **0.981** | **2.22** | 0 | 6 |
Rebuild: fuse.py --design f3 --paw-align (--stages-dir variants/1000/timelapse);
refine 20000 --flips --w-lm 1.5; refine 15000 --flips --w-nv 1.5 --w-lm 1.5;
fix_selfint.py --alpha 0.2; refine 6000 --flips --w-nv 1.5 --w-lm 1.5 (all with
--snapshot-dir timelapse/refine/snapsN --snapshot-every 500).
