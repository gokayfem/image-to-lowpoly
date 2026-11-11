#!/usr/bin/env python3
"""Preflight stage: derive the fixed evaluation targets from reference/cat.png.

Regenerates everything in reference/derived/ from
  - reference/cat.png                         (read-only, never modified)
  - reference/derived/landmarks_manual.json   (hand annotations, read only)
  - reference/derived/regions_manual.json     (hand-drawn windows, read only)

Outputs (reference/derived/):
  mask.png, mask_meta.json, reference_bbox.json, landmarks.json, regions.json,
  mask_overlay.png, landmarks_overlay.png, crops/annot_*.png (gridded crops used to
  read landmark coordinates), crops/check_*.png (zoomed mask/landmark overlays).

Pixel convention (contract §3): u right, v down, origin at the top-left corner of the
top-left pixel, pixel centres at integer + 0.5.  Pixel (row r, col c) has centre
(u, v) = (c + 0.5, r + 0.5).

The script never invents landmarks: it copies the manual uv values verbatim and only
adds class / sigma / note fields that are themselves stored in the manual file.

Run:  python3 scripts/preflight_reference.py [--skip-crops]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the folder that holds catlib/
from catlib import paths as _paths  # noqa: E402  (resolves the case: --case / LOWPOLY_CASE / cwd)
ROOT = _paths.ROOT
REF = ROOT / "reference" / "cat.png"
OUT = ROOT / "reference" / "derived"
CROPS = OUT / "crops"

SHARP = ["ear_tip_R", "ear_tip_L", "eye_R", "eye_L", "nose", "paw_FR", "paw_FL",
         "paw_HR", "paw_HL", "tail_base", "tail_mid", "tail_tip"]
FUZZY = ["shoulder_R", "hip_R", "hock_HR", "chin"]
REGION_NAMES = ["head", "tail", "legs", "torso"]

# ---------------------------------------------------------------- mask parameters
P = {
    "colour_space": "CIELAB (OpenCV 8-bit Lab, a/b re-centred at 128)",
    "feature": "chroma C = sqrt(a^2 + b^2); background and floor shadow are neutral grey "
               "(C ~ 0-1), all fur incl. pale cream has C >= ~8",
    "median_blur_ksize": 5,
    "chroma_fg_threshold": 5.0,         # initial binary mask: C >= this
    "grabcut_sure_fg_chroma": 12.0,     # C >= this, eroded -> GC_FGD
    "grabcut_sure_fg_erode_px": 6,
    "grabcut_sure_bg_chroma": 3.0,      # C < this, eroded -> GC_BGD
    "grabcut_sure_bg_erode_px": 4,
    "grabcut_iters": 5,
    "grabcut_scale": 0.5,               # run GrabCut on a half-res image, upsample result
    "combine": "final = grabcut_fg AND (C_blur >= chroma_fg_threshold_loose)",
    "chroma_fg_threshold_loose": 3.5,
    "largest_component": True,
    "fill_holes_max_area_px": 4000,     # only enclosed holes smaller than this are filled
    "open_radius_px": 2,                # drops whiskers / stray hairs (elliptical SE)
    "close_radius_px": 2,               # seals toe-gap notches on paw outlines
    "final_smooth": "Gaussian sigma 1.0 on the binary mask, re-threshold at 0.5",
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def chroma_map(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    return np.hypot(a, b)


def build_mask(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    C = chroma_map(cv2.medianBlur(rgb, P["median_blur_ksize"]))
    init = C >= P["chroma_fg_threshold"]
    # GrabCut on half resolution for speed, seeded from chroma.
    s = P["grabcut_scale"]
    small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    Cs = cv2.resize(C, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gc = np.where(Cs >= P["chroma_fg_threshold"], cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    sure_fg = cv2.erode((Cs >= P["grabcut_sure_fg_chroma"]).astype(np.uint8),
                        disk(max(1, int(P["grabcut_sure_fg_erode_px"] * s))))
    sure_bg = cv2.erode((Cs < P["grabcut_sure_bg_chroma"]).astype(np.uint8),
                        disk(max(1, int(P["grabcut_sure_bg_erode_px"] * s))))
    gc[sure_fg > 0] = cv2.GC_FGD
    gc[sure_bg > 0] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd,
                P["grabcut_iters"], cv2.GC_INIT_WITH_MASK)
    gfg = np.isin(gc, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.float32)
    gfg = cv2.resize(gfg, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR) >= 0.5
    m = gfg & (C >= P["chroma_fg_threshold_loose"])
    # morphology
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, disk(P["open_radius_px"])) > 0
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, disk(P["close_radius_px"])) > 0
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    holes = ndi.binary_fill_holes(m) & ~m
    hl, hn = ndi.label(holes)
    if hn:
        hs = ndi.sum(holes, hl, range(1, hn + 1))
        small_ids = 1 + np.nonzero(hs < P["fill_holes_max_area_px"])[0]
        m = m | np.isin(hl, small_ids)
    m = ndi.gaussian_filter(m.astype(np.float32), 1.0) >= 0.5
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    stats = {
        "initial_chroma_fg_px": int(init.sum()),
        "grabcut_fg_px": int(gfg.sum()),
        "final_fg_px": int(m.sum()),
        "unfilled_large_holes": int(hn - (len(small_ids) if hn else 0)) if hn else 0,
    }
    return m, stats


# ---------------------------------------------------------------- drawing helpers
def font(size: int):
    for p in ["/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def grid_crop(rgb: np.ndarray, box, scale: int, path: Path, minor=10, major=50,
              mask: np.ndarray | None = None, points: dict | None = None) -> None:
    """Upscaled crop with a pixel grid (minor every `minor` px, labelled every `major`)."""
    u0, v0, u1, v1 = box
    crop = rgb[v0:v1, u0:u1]
    img = Image.fromarray(crop).resize(((u1 - u0) * scale, (v1 - v0) * scale), Image.NEAREST)
    pad = 44
    canvas = Image.new("RGB", (img.width + pad, img.height + pad), (255, 255, 255))
    canvas.paste(img, (pad, pad))
    d = ImageDraw.Draw(canvas)
    f = font(13)
    if mask is not None:
        mc = mask[v0:v1, u0:u1].astype(np.uint8)
        edge = (mc - cv2.erode(mc, np.ones((3, 3), np.uint8))) > 0
        rr, cc = np.nonzero(edge)
        for r, c in zip(rr, cc):
            x = pad + c * scale
            y = pad + r * scale
            d.rectangle([x, y, x + scale - 1, y + scale - 1], outline=(0, 255, 255))
    for g in range((u0 // minor) * minor, u1 + 1, minor):
        if g < u0:
            continue
        x = pad + (g - u0) * scale
        maj = g % major == 0
        d.line([(x, pad), (x, canvas.height)], fill=(255, 0, 255) if maj else (90, 90, 255), width=1)
        if maj:
            d.text((x + 2, 2), str(g), fill=(0, 0, 0), font=f)
    for g in range((v0 // minor) * minor, v1 + 1, minor):
        if g < v0:
            continue
        y = pad + (g - v0) * scale
        maj = g % major == 0
        d.line([(pad, y), (canvas.width, y)], fill=(255, 0, 255) if maj else (90, 90, 255), width=1)
        if maj:
            d.text((2, y + 2), str(g), fill=(0, 0, 0), font=f)
    if points:
        for name, (u, v) in points.items():
            if u0 <= u < u1 and v0 <= v < v1:
                x = pad + (u - u0) * scale
                y = pad + (v - v0) * scale
                r = max(4, scale)
                d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=2)
                d.line([(x - 2 * r, y), (x + 2 * r, y)], fill=(255, 0, 0))
                d.line([(x, y - 2 * r), (x, y + 2 * r)], fill=(255, 0, 0))
                d.text((x + r + 3, y - r - 14), name, fill=(255, 255, 0), font=font(14),
                       stroke_width=2, stroke_fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


# crop windows used for annotation and checking: name -> (box, scale)
ANNOT_CROPS = {
    "ears": ((920, 330, 1270, 480), 4),
    "face": ((1000, 460, 1240, 680), 4),
    "paws_front": ((900, 1240, 1250, 1380), 4),
    "paws_hind": ((390, 1190, 780, 1310), 4),
    "tail_tip": ((100, 330, 260, 470), 6),
    "tail_full": ((100, 330, 560, 780), 2),
    "tail_base": ((380, 560, 580, 780), 4),
    "shoulder_hip": ((380, 600, 1250, 1100), 1),
    "hock": ((380, 1000, 560, 1220), 4),
    "chin": ((1050, 580, 1230, 720), 5),
    "leg_gaps": ((380, 1000, 1260, 1360), 2),
    "head": ((900, 330, 1280, 720), 2),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-crops", action="store_true", help="do not write crops/")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rgb = np.array(Image.open(REF).convert("RGB"))
    H, W = rgb.shape[:2]
    assert (W, H) == (1536, 1536), (W, H)
    ref_sha = sha256(REF)

    # ---------------- mask
    m, mstats = build_mask(rgb)
    Image.fromarray((m * 255).astype(np.uint8), mode="L").save(OUT / "mask.png")
    rr, cc = np.nonzero(m)
    bbox = {"u0": int(cc.min()), "v0": int(rr.min()), "u1": int(cc.max()) + 1, "v1": int(rr.max()) + 1}
    bbox["diag_px"] = float(np.hypot(bbox["u1"] - bbox["u0"], bbox["v1"] - bbox["v0"]))
    bbox["area_px"] = int(m.sum())
    bbox["note"] = ("pixel-edge bbox of mask.png (u0,v0 inclusive top-left edge, u1,v1 exclusive "
                    "= right/bottom edge of last cat pixel); diag_px = hypot(u1-u0, v1-v0)")
    (OUT / "reference_bbox.json").write_text(json.dumps(bbox, indent=2) + "\n")
    lab, ncomp = ndi.label(m)
    holes_left = int((ndi.binary_fill_holes(m) & ~m).sum())
    meta = {
        "source": "reference/cat.png",
        "source_sha256": ref_sha,
        "image_size": [W, H],
        "method": "chroma threshold + GrabCut refinement + morphology (see params)",
        "params": P,
        "stats": mstats,
        "connected_components": int(ncomp),
        "enclosed_background_px": holes_left,
        "binarise": ">= 128 (file stores 0/255)",
        "generator": "scripts/preflight_reference.py",
    }
    (OUT / "mask_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"mask: area={bbox['area_px']} bbox={bbox['u0'],bbox['v0'],bbox['u1'],bbox['v1']} "
          f"diag={bbox['diag_px']:.1f} components={ncomp}")

    # mask overlay: photo with boundary | mask alone
    mu8 = m.astype(np.uint8)
    edge = (mu8 - cv2.erode(mu8, np.ones((3, 3), np.uint8))) > 0
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    ov = rgb.copy()
    ov[edge] = (0, 220, 255)
    tint = ov.astype(np.float32)
    tint[~m] = tint[~m] * 0.6 + np.array([60, 0, 90]) * 0.4
    side = np.dstack([mu8 * 255] * 3)
    Image.fromarray(np.concatenate([tint.astype(np.uint8), side], axis=1)).save(OUT / "mask_overlay.png")

    # ---------------- annotation crops (always the same windows)
    if not args.skip_crops:
        for name, (box, sc) in ANNOT_CROPS.items():
            grid_crop(rgb, box, sc, CROPS / f"annot_{name}.png")

    # ---------------- landmarks
    man_p = OUT / "landmarks_manual.json"
    reg_p = OUT / "regions_manual.json"
    if not man_p.exists() or not reg_p.exists():
        print("landmarks_manual.json / regions_manual.json missing: wrote mask + annotation crops only")
        return 0
    man = json.loads(man_p.read_text())
    lms = {}
    missing = [n for n in SHARP + FUZZY if n not in man["landmarks"]]
    if missing:
        print("ERROR: manual landmarks missing:", missing)
        return 2
    for n in SHARP + FUZZY:
        e = man["landmarks"][n]
        cls = "sharp" if n in SHARP else "fuzzy"
        if e.get("class", cls) != cls:
            print(f"ERROR: {n} class {e.get('class')} != contract class {cls}")
            return 2
        lms[n] = {"uv": [float(e["uv"][0]), float(e["uv"][1])], "class": cls,
                  "sigma_px": float(e["sigma_px"]), "note": e["note"]}
        u, v = lms[n]["uv"]
        inside = bool(m[int(v), int(u)])
        lms[n]["in_mask"] = inside
        if not inside:
            print(f"WARNING: landmark {n} at {u,v} is outside the mask")
    out_lm = {
        "image_size": [W, H],
        "pixel_convention": "u right, v down, origin top-left corner of top-left pixel, "
                            "pixel centres at integer+0.5",
        "source_sha256": ref_sha,
        "annotation": man.get("annotation", "manual"),
        "left_right": man.get("left_right", ""),
        "landmarks": lms,
    }
    (OUT / "landmarks.json").write_text(json.dumps(out_lm, indent=2) + "\n")

    reg = json.loads(reg_p.read_text())
    regions = {k: [int(x) for x in reg["regions"][k]] for k in REGION_NAMES}
    out_reg = dict(regions)
    (OUT / "regions.json").write_text(json.dumps(out_reg, indent=2) + "\n")
    (OUT / "regions_notes.json").write_text(json.dumps(
        {"source_sha256": ref_sha, "format": "{name: [u0, v0, u1, v1]} pixel edges, u1/v1 exclusive",
         "notes": reg.get("notes", {})}, indent=2) + "\n")
    for k, (a, b, c, d) in regions.items():
        sub = m[b:d, a:c]
        print(f"region {k:6s} {a,b,c,d} fg_frac={sub.mean():.3f}")

    # landmark overlay
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    cols = {"head": (255, 60, 60), "tail": (60, 160, 255), "legs": (60, 220, 60), "torso": (255, 200, 0)}
    for k, (a, b, c, e) in regions.items():
        d.rectangle([a, b, c - 1, e - 1], outline=cols[k], width=3)
        d.text((a + 6, b + 4), k, fill=cols[k], font=font(28), stroke_width=2, stroke_fill=(0, 0, 0))
    for n, e in lms.items():
        u, v = e["uv"]
        col = (255, 0, 0) if e["class"] == "sharp" else (255, 0, 255)
        r = 7
        d.ellipse([u - r, v - r, u + r, v + r], fill=col, outline=(255, 255, 255), width=2)
        d.text((u + 10, v - 10), n, fill=(255, 255, 255), font=font(22), stroke_width=3, stroke_fill=(0, 0, 0))
    img.save(OUT / "landmarks_overlay.png")

    # zoomed check crops: mask edge + landmark dots
    if not args.skip_crops:
        pts = {n: e["uv"] for n, e in lms.items()}
        checks = {
            "paw_FR": ((930, 1270, 1060, 1380), 5), "paw_FL": ((1090, 1250, 1240, 1370), 5),
            "paw_HR": ((400, 1210, 530, 1310), 5), "paw_HL": ((640, 1200, 770, 1300), 5),
            "head": ((920, 330, 1270, 700), 2), "leg_gaps": ((380, 1000, 1260, 1380), 2),
            "tail_tip": ((100, 330, 260, 470), 6), "tail": ((100, 330, 580, 780), 2),
            "ears": ((920, 330, 1270, 480), 4), "chin_neck": ((1030, 560, 1250, 760), 4),
            "belly": ((450, 900, 1000, 1150), 2),
        }
        for name, (box, sc) in checks.items():
            grid_crop(rgb, box, sc, CROPS / f"check_{name}.png", mask=m, points=pts)
    print("wrote", sorted(p.name for p in OUT.iterdir()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
