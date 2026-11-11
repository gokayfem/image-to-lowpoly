#!/usr/bin/env python3
"""Build the head-kit templates (head-local frame) at five LODs, validate, preview.

    python3 scripts/build_head_kit.py [--shape head/pose.json] [--out head/template]
                                               [--preview head/renders/template_preview.png]

Writes head/template/head_L{0..4}_local.json (head-local metres, template proportions or
the fitted ones from --shape), head/template/features.json (the per-LOD feature table),
and a CPU flat-shaded preview sheet (front / reference-like 3/4 / side / top / back).
CPU only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the folder that holds catlib/
from catlib import paths as _paths  # noqa: E402  (resolves the case: --case / LOWPOLY_CASE / cwd)
ROOT = _paths.ROOT

from catlib import headkit as hk  # noqa: E402
from catlib.camera import Camera, look_at_matrix  # noqa: E402
from catlib.raster import id_buffer  # noqa: E402

MAT_RGB = {"fur_orange": (214, 137, 70), "fur_cream": (236, 214, 178), "ear_inner": (222, 150, 150),
           "eye": (196, 160, 40), "pupil": (25, 20, 15), "nose": (225, 130, 130), "fur_stripe": (160, 90, 40),
           "neck": (150, 150, 150)}

# what survives at each LOD (written to features.json and README)
FEATURE_TABLE = {
    0: "skull block with crown ridge, cheeks = lower neck-loop corners, single nose point, "
       "throat/chin break on the neck loop, 3-sided ear pyramids (front face + 2 back faces); no eyes",
    1: "L0 + brow (forehead plane), separate blunt chin, cheek pair, eye-area vertex (eye landmark on a "
       "head face), 6-vertex neck loop",
    2: "L1 + nose pad facet (2 'nose' tris), whisker pads, temple, skull side, occiput, one flat 'eye' "
       "triangle per eye, 4-sided ear pyramids with a separate ear_inner front face",
    3: "L2 + mouth step, nose bridge, forehead, back of skull, chin sides, jaw, cheekbones, brows, mouth "
       "corners, inset 4-sided eye sockets with a centre vertex, cupped two-segment ears with an "
       "ear_inner cup and real thickness, 8-vertex neck loop",
    4: "L3 + lower lip, philtrum, rounder skull (crown back, skull high, back low), whisker-pad "
       "rims, cheek high/low, brow outer, temple low, 6-sided eyes with a vertical-slit pupil (2 "
       "'pupil' tris), ears with a fur rim around an inset cup, 10-vertex neck loop",
}


def shade(V, T, mats, cam, size=512, bg=(245, 245, 245)):
    ids, _ = id_buffer(V, T, cam, size)
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = cam.matrix_world[:3, :3] @ np.array([-0.4, 0.5, 0.75])
    light /= np.linalg.norm(light)
    lam = 0.35 + 0.65 * np.clip(np.abs(n @ light), 0, 1)
    col = np.array([MAT_RGB.get(m, (200, 200, 200)) for m in mats], float) * lam[:, None]
    img = np.zeros(ids.shape + (3,))
    img[:] = bg
    m = ids >= 0
    img[m] = col[ids[m]]
    # draw edges: pixels whose neighbour id differs
    e = np.zeros_like(m)
    e[:-1] |= ids[:-1] != ids[1:]
    e[:, :-1] |= ids[:, :-1] != ids[:, 1:]
    e &= m | np.roll(m, 1, 0)
    img[e] = img[e] * 0.35
    return img.astype(np.uint8)


def view_cam(target, az, el, dist, size, f_mm=85.0):
    a, e = math.radians(az), math.radians(el)
    loc = np.asarray(target) + dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    up = np.array([1.0, 0, 0]) if abs(el) > 89 else None
    M = look_at_matrix(loc, target, up_hint=up)
    return Camera(size, size, f_mm / 36.0 * size, M)


def preview_sheet(lods: dict, out: Path, size=420, views=None):
    """lods: {lod: (V metres, T, mats)}. Rows = LOD, cols = views."""
    from PIL import Image, ImageDraw
    views = views or [("front", 0, 3), ("3q_R", -40, 3), ("side_R", -90, 0), ("top", 0, 90), ("3q_back_L", 140, 20)]
    rows = []
    for lod, (V, T, mats) in lods.items():
        c = 0.5 * (V.min(0) + V.max(0))
        r = np.linalg.norm(V.max(0) - V.min(0))
        dist = r * 2.6
        tiles = []
        for name, az, el in views:
            img = shade(V, T, mats, view_cam(c, az, el, dist, size), size)
            im = Image.fromarray(img)
            ImageDraw.Draw(im).text((6, 4), f"L{lod} {name} {len(T)} tris", fill=(0, 0, 0))
            tiles.append(np.asarray(im))
        rows.append(np.concatenate(tiles, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=Path, default=None, help="pose.json with fitted 'shape'")
    ap.add_argument("--out", type=Path, default=ROOT / "head" / "template")
    ap.add_argument("--preview", type=Path, default=ROOT / "head" / "renders" / "template_preview.png")
    ap.add_argument("--lods", default="0,1,2,3,4")
    a = ap.parse_args()
    S = dict(hk.DEFAULT_SHAPE)
    if a.shape:
        S.update(json.loads(a.shape.read_text())["shape"])
    a.out.mkdir(parents=True, exist_ok=True)
    a.preview.parent.mkdir(parents=True, exist_ok=True)
    prev, summary = {}, {}
    for lod in [int(x) for x in a.lods.split(",")]:
        tpl = hk.build_template(lod)            # connectivity from template proportions
        d = hk.export_lod(tpl, S)
        chk = hk.check_lod(d)
        d["checks"] = chk
        d["feature_table"] = FEATURE_TABLE[lod]
        (a.out / f"head_L{lod}_local.json").write_text(json.dumps(d))
        summary[lod] = {"triangles": chk["triangles"], "neck_loop": len(d["neck_loop"]),
                        "ok": chk["ok"], "mirror_err": chk.get("mirror_max_err_m"),
                        "sym_tris": chk.get("triangulation_symmetric"),
                        "min_area_mm2": round(chk["min_triangle_area_mm2"], 2),
                        "min_edge_mm": round(chk["min_edge_mm"], 2),
                        "self_int": chk["self_intersecting_pairs"],
                        "regions": {r: d["tri_region"].count(r) for r in sorted(set(d["tri_region"]))}}
        print(f"L{lod}", json.dumps(summary[lod]))
        prev[lod] = (np.asarray(d["vertices"]), np.asarray(d["triangles"]), d["tri_material"])
    (a.out / "features.json").write_text(json.dumps({"feature_table": FEATURE_TABLE, "summary": summary,
                                                     "shape": S}, indent=1))
    preview_sheet(prev, a.preview)
    print("preview", a.preview)


if __name__ == "__main__":
    main()
