"""ref_overlay.png: reference photo + rendered silhouette contour + landmarks (venv only, PIL)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import paths
from .metrics import boundary


def make_overlay(out_path, rendered_mask: np.ndarray, metrics: dict, regions: dict | None,
                 ref_mask: np.ndarray | None = None, reference_png=None, title: str = "") -> Path:
    from PIL import Image, ImageDraw, ImageFont
    ref = Image.open(reference_png or paths.REFERENCE_PNG).convert("RGB")
    W, H = ref.size
    base = np.asarray(ref).astype(np.float32) * 0.75 + 255 * 0.25          # lighten photo
    img = base.astype(np.uint8)
    if ref_mask is not None:                                               # faint ref outline
        bR = boundary(ref_mask)
        img[bR] = (0, 150, 0)
    bM = boundary(rendered_mask)
    from scipy.ndimage import binary_dilation
    bM = binary_dilation(bM, iterations=1)
    img[bM] = (255, 0, 255)                                                # render contour
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
        big = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 30)
    except OSError:
        font = big = ImageFont.load_default()
    for name, (u0, v0, u1, v1) in (regions or {}).items():
        dr.rectangle([u0, v0, u1, v1], outline=(40, 90, 255, 110), width=2)
        dr.text((u0 + 5, v0 + 3), name, fill=(40, 90, 255, 160), font=font)
    for name, row in metrics.get("landmarks", {}).items():
        pu, pv = row["proj_uv"]
        ru = row.get("ref_uv")
        if ru is not None:
            dr.line([tuple(ru), (pu, pv)], fill=(255, 220, 0, 255), width=3)
            r = 9
            dr.ellipse([ru[0] - r, ru[1] - r, ru[0] + r, ru[1] + r], outline=(0, 170, 0, 255),
                       width=4)
        r = 7
        col = (220, 0, 0, 255)
        dr.ellipse([pu - r, pv - r, pu + r, pv + r], fill=col)
        flag = ("*" if row.get("occluded") else "") + ("!" if row.get("off_surface") else "")
        dr.text((pu + 10, pv - 12), name + flag, fill=(150, 0, 0, 255), font=font)
    m = metrics
    lines = [title,
             ("OFFICIAL (Blender silhouette)" if m.get("official") else
              "UNOFFICIAL (catlib.raster silhouette, no render)"),
             f"IoU {m.get('iou', float('nan')):.4f}   boundary mean {m.get('boundary_mean_px') or float('nan'):.1f}px"
             f"   HD95 {m.get('boundary_hd95_px') or float('nan'):.1f}px",
             f"landmarks sharp mean {m.get('landmark_sharp_mean_px') or float('nan'):.1f}px"
             f"   legs_gap_iou {m.get('legs_gap_iou') if m.get('legs_gap_iou') is not None else float('nan'):.3f}",
             "magenta = rendered contour, green outline = reference mask,",
             "green ring = reference landmark, red dot = projected (* occluded, ! off-surface)"]
    y = 12
    for ln in lines:
        if ln:
            dr.rectangle([8, y - 2, 8 + 16 * len(ln), y + 32], fill=(255, 255, 255, 170))
            dr.text((12, y), ln, fill=(0, 0, 0, 255), font=big if y == 12 else font)
            y += 36
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    return out_path
