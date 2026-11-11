"""Neutral image packet for an independent visual critic (PRIOR_EXPERIMENT_LESSONS 4).

    python3 scripts/make_critic_packet.py --round K [--checkpoints 100:TAG,...] \
        [--variants-root variants] [--out outputs]

Writes ONLY these files to <out>/critic/round_K/ (the directory is recreated):

  reference.png                 the fixed reference photo (pixels unchanged, metadata stripped)
  budget_<N>_clay.png           reference-view clay render
  budget_<N>_wire.png           reference-view clay + triangle edges
  budget_<N>_beauty.png         reference-view flat-colour render
  budget_<N>_overlay_outline.png  reference photo with ONLY the rendered silhouette outline
                                (regenerated from renders/ref_silhouette.png; ref_overlay.png is
                                never used because it prints scores and landmark dots)
  budget_<N>_views_clay.png     contact sheet of the eight novel clay views, view names only
  sheet_clay_wire.png           reference + four budgets, clay and wire rows, budget labels only
  PACKET.md                     neutral list of the files

No scores, distances, checkpoint tags, "initial/final" wording or implementation notes appear in
file names, pixels or PNG metadata (every PNG is re-encoded without text chunks).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the folder that holds catlib/
from catlib import paths as _paths  # noqa: E402  (resolves the case: --case / LOWPOLY_CASE / cwd)
ROOT = _paths.ROOT
sys.path.insert(0, str(_paths.SCRIPTS))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from catlib import paths  # noqa: E402
import integ_common as ic  # noqa: E402
from integ_common import BG, INK, MUTED, RULE, font, save_png, load_rgb, load_mask  # noqa: E402

OUTLINE_RGB = (0, 200, 255)


def overlay_outline(ref: Image.Image, M: np.ndarray, width: int = 4) -> Image.Image:
    a = np.asarray(ref).copy()
    edge = ic.outline(M, width)
    a[edge] = OUTLINE_RGB
    return Image.fromarray(a, "RGB")


def views_sheet(e: ic.Entry, views, crops, cell=520) -> Image.Image:
    cols = 4
    rows = (len(views) + cols - 1) // cols
    ch = round(cell / 1.25)
    lab_h, pad, gap = 44, 36, 18
    W = pad * 2 + cols * cell + (cols - 1) * gap
    H = pad * 2 + 70 + rows * (ch + lab_h) + (rows - 1) * gap
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((pad, pad), f"{e.budget}-triangle budget: novel views (clay)", font=font(34, "bold"), fill=INK)
    for i, v in enumerate(views):
        r, c = divmod(i, cols)
        x = pad + c * (cell + gap)
        y = pad + 70 + r * (ch + lab_h + gap)
        name = ic.VIEW_LABEL.get(v, v)
        if v in ic.INFERRED_VIEWS:
            name += " (side not visible in the photo)"
        d.text((x, y + 8), name, font=font(22), fill=MUTED)
        im = load_rgb(e.render(f"view_{v}_clay.png")).crop(tuple(crops[v]))
        img.paste(ic.fit(im, cell, ch, bg=(255, 255, 255)), (x, y + lab_h))
        d.rectangle([x - 1, y + lab_h - 1, x + cell, y + lab_h + ch], outline=RULE)
    return img


def sheet(entries, box, cell=640) -> Image.Image:
    bw, bh = box[2] - box[0], box[3] - box[1]
    ch = round(cell * bh / bw)
    pad, gap, head = 40, 18, 64
    rowlab = 150
    n = 1 + len(entries)
    W = pad * 2 + rowlab + n * cell + (n - 1) * gap
    H = pad * 2 + head + 2 * ch + gap
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    ref = load_rgb(paths.REFERENCE_PNG).crop(tuple(box))
    cols = [("Reference", ref, ref)]
    for e in entries.values():
        cols.append((f"{e.budget}-triangle budget", load_rgb(e.render("ref_clay.png")).crop(tuple(box)),
                     load_rgb(e.render("ref_wire.png")).crop(tuple(box))))
    for i, (lab, a, b) in enumerate(cols):
        x = pad + rowlab + i * (cell + gap)
        d.text((x + cell // 2, pad + 10), lab, font=font(30, "bold"), fill=INK, anchor="ma")
        for r, im in enumerate((a, b)):
            y = pad + head + r * (ch + gap)
            img.paste(ic.fit(im, cell, ch, bg=(255, 255, 255)), (x, y))
            d.rectangle([x - 1, y - 1, x + cell, y + ch], outline=RULE)
    for r, lab in enumerate(("clay", "wireframe")):
        d.text((pad, pad + head + r * (ch + gap) + ch // 2), lab, font=font(28, "bold"), fill=INK, anchor="lm")
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Neutral critic packet (no scores).")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--checkpoints", default=None, help="100:TAG,200:TAG,500:TAG,1000:TAG (default: final/)")
    ap.add_argument("--variants-root", type=Path, default=paths.VARIANTS)
    ap.add_argument("--out", type=Path, default=paths.OUTPUTS)
    a = ap.parse_args(argv)
    entries = ic.load_entries(a.variants_root, a.checkpoints)
    pk = (a.out / "critic" / f"round_{a.round}").resolve()
    if pk.exists():
        shutil.rmtree(pk)
    pk.mkdir(parents=True)

    sys.path.insert(0, str(_paths.SCRIPTS))
    from make_comparison import ref_crop_box, view_names, view_crops, VIEW_ASPECT  # noqa: E402
    ref = load_rgb(paths.REFERENCE_PNG)
    save_png(ref, pk / "reference.png")
    views = view_names(entries)
    crops = view_crops(entries, views, VIEW_ASPECT) if views else {}
    files = [("reference.png", "the reference photo")]
    for N, e in entries.items():
        for kind, src in (("clay", "ref_clay.png"), ("wire", "ref_wire.png"), ("beauty", "ref_beauty.png")):
            save_png(load_rgb(e.render(src)), pk / f"budget_{N}_{kind}.png")
        save_png(overlay_outline(ref, load_mask(e.render("ref_silhouette.png"))), pk / f"budget_{N}_overlay_outline.png")
        files += [(f"budget_{N}_clay.png", f"{N}-triangle budget, reference camera, grey clay"),
                  (f"budget_{N}_wire.png", f"{N}-triangle budget, reference camera, clay with triangle edges"),
                  (f"budget_{N}_beauty.png", f"{N}-triangle budget, reference camera, flat colours"),
                  (f"budget_{N}_overlay_outline.png",
                   f"reference photo with the outline of the {N}-triangle model drawn in cyan")]
        if views:
            save_png(views_sheet(e, views, crops), pk / f"budget_{N}_views_clay.png")
            files.append((f"budget_{N}_views_clay.png", f"{N}-triangle budget, {len(views)} other viewpoints, clay"))
    save_png(sheet(entries, ref_crop_box(entries)), pk / "sheet_clay_wire.png")
    files.append(("sheet_clay_wire.png", "reference and all four budgets side by side, clay and wireframe rows"))
    md = [f"# Review packet, round {a.round}", "",
          "Images of one fixed reference photo of a cat and four low-poly 3D models of it, made under "
          "triangle budgets of 100, 200, 500 and 1000. All renders use the same camera, lights and "
          "presentation. The photo shows the cat's right side; views of the cat's left side, back and top "
          "show shape that is not visible in the photo.", "", "| File | Content |", "|---|---|"]
    md += [f"| `{f}` | {t} |" for f, t in files]
    (pk / "PACKET.md").write_text("\n".join(md) + "\n")
    print(f"wrote {len(files) + 1} files to {pk}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
