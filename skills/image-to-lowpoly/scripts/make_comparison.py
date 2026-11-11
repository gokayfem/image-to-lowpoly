"""Comparison sheets for the four budget variants (EXPERIMENT_BRIEF "Required deliverables").

    python3 scripts/make_comparison.py [--variants-root variants] [--out outputs] \
        [--checkpoints 100:TAG,200:TAG,500:TAG,1000:TAG] [--baselines-root baselines] \
        [--proxy-report proxy/fit_report.json] [--cell 900] [--view-cell 640]

Reads, per budget, final/ (default) or checkpoints/<TAG>/ (mid-run): renders/ref_*.png,
renders/view_*_{clay,wire}.png, metrics.json, validation.json; plus reference/cat.png and
reference/derived/mask.png.  Writes to --out:

  comparison_reference_plus_four.png  reference photo + four beauty renders (reference camera),
                                      target budget and ACTUAL validated triangle count
  comparison_clay_wire.png            columns reference/100/200/500/1000; rows clay, wire,
                                      silhouette difference; 4-way counts + agreement printed
  comparison_multiview_clay.png       rows = 8 novel views, cols = 4 budgets, one shared crop per view
  comparison_multiview_wire.png
  comparison_metrics_table.png        official metrics, initial vs final (+ baselines)
  comparison_manifest.json            what was read (dirs, tags, hashes, crops)

All crops of one sheet row use one crop box shared by every column, so sizes are comparable.
Purely 2D layout: no metric is computed here except the per-pixel silhouette difference image.
"""
from __future__ import annotations

import argparse
import json
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
from integ_common import (BG, BUDGETS, INK, MUTED, RULE, OK_GREEN, BAD_RED, font, draw_centered,  # noqa: E402
                          save_png, load_rgb, load_mask)

PAD = 56          # outer margin
GAP = 24          # gap between cells


# --------------------------------------------------------------------------- generic grid
def compose(title: str, subtitle: str, col_heads: list[list[tuple]], row_labels: list[list[tuple]],
            cells: list[list[Image.Image]], cell_w: int, cell_h: int, col_feet: list[list[tuple]] | None = None,
            footer: list[tuple] | None = None, legend: list[tuple] | None = None,
            row_label_w: int = 0, warn: str | None = None) -> Image.Image:
    """Lines are (text, font, colour).  cells[r][c] already sized cell_w x cell_h."""
    tmp = ImageDraw.Draw(Image.new("RGB", (8, 8)))

    def block_h(lines):
        return sum(ic.text_size(tmp, t or "Ag", f)[1] + 10 for t, f, _ in lines) if lines else 0

    n_r, n_c = len(cells), len(cells[0])
    f_title, f_sub = font(46, "bold"), font(26)
    head_h = max(block_h(h) for h in col_heads) + 14
    feet_h = (max(block_h(h) for h in col_feet) + 18) if col_feet else 0
    top = PAD + ic.text_size(tmp, title, f_title)[1] + 18 + (ic.text_size(tmp, subtitle, f_sub)[1] + 14 if subtitle else 0)
    if warn:
        top += 52
    top += 30
    grid_w = row_label_w + n_c * cell_w + (n_c - 1) * GAP
    W = PAD * 2 + grid_w
    legend_h = 70 if legend else 0
    foot_h = block_h(footer) + 20 if footer else 0
    H = top + head_h + n_r * cell_h + (n_r - 1) * GAP + feet_h + legend_h + foot_h + PAD
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = PAD
    d.text((PAD, y), title, font=f_title, fill=INK)
    y += ic.text_size(d, title, f_title)[1] + 18
    if subtitle:
        d.text((PAD, y), subtitle, font=f_sub, fill=MUTED)
        y += ic.text_size(d, subtitle, f_sub)[1] + 14
    if warn:
        d.rectangle([PAD, y, W - PAD, y + 40], fill=(253, 236, 234))
        d.text((PAD + 14, y + 20), warn, font=font(24, "bold"), fill=BAD_RED, anchor="lm")
        y += 52
    y += 14
    d.line([PAD, y, W - PAD, y], fill=RULE, width=2)
    y += 16
    x0 = PAD + row_label_w
    # column heads
    for c, lines in enumerate(col_heads):
        cx = x0 + c * (cell_w + GAP) + cell_w // 2
        yy = y
        for t, f, col in lines:
            draw_centered(d, cx, yy, t, f, col)
            yy += ic.text_size(d, t or "Ag", f)[1] + 10
    y += head_h
    for r in range(n_r):
        ry = y + r * (cell_h + GAP)
        if row_label_w and row_labels:
            yy = ry + cell_h // 2 - block_h(row_labels[r]) // 2
            for t, f, col in row_labels[r]:
                d.text((PAD, yy), t, font=f, fill=col)
                yy += ic.text_size(d, t or "Ag", f)[1] + 10
        for c in range(n_c):
            cx = x0 + c * (cell_w + GAP)
            img.paste(cells[r][c], (cx, ry))
            d.rectangle([cx - 1, ry - 1, cx + cell_w, ry + cell_h], outline=RULE, width=1)
    y += n_r * cell_h + (n_r - 1) * GAP + 16
    if col_feet:
        for c, lines in enumerate(col_feet):
            cx = x0 + c * (cell_w + GAP) + cell_w // 2
            yy = y
            for t, f, col in lines:
                draw_centered(d, cx, yy, t, f, col)
                yy += ic.text_size(d, t or "Ag", f)[1] + 10
        y += feet_h
    if legend:
        d.line([PAD, y, W - PAD, y], fill=RULE, width=2)
        y += 22
        x = PAD
        fl = font(26)
        for t, colr in legend:
            d.rectangle([x, y, x + 34, y + 30], fill=colr, outline=(90, 90, 90))
            d.text((x + 46, y + 15), t, font=fl, fill=INK, anchor="lm")
            x += 46 + ic.text_size(d, t, fl)[0] + 56
        y += legend_h - 22
    if footer:
        d.line([PAD, y, W - PAD, y], fill=RULE, width=2)
        y += 16
        for t, f, col in footer:
            d.text((PAD, y), t, font=f, fill=col)
            y += ic.text_size(d, t or "Ag", f)[1] + 10
    return img


def crop_cell(img: Image.Image, box, w: int, h: int) -> Image.Image:
    return ic.fit(img.crop(tuple(box)), w, h, bg=(255, 255, 255))


# --------------------------------------------------------------------------- sheet builders
def provisional_warning(entries) -> str | None:
    st = ic.camera_status(entries)
    msgs = []
    if st != "locked":
        msgs.append(f"Camera status '{st}': NOT final, not comparable to final tables.")
    mism = ic.provenance_mismatch([e.metrics for e in entries.values()])
    if mism:
        msgs.append("Columns scored with different " + ", ".join(mism) + ": re-evaluate before comparing.")
    return "  ".join(msgs) or None


def tag_suffix(e: ic.Entry) -> str:
    return "" if e.label == "final" else f"  [{e.label}]"


def actual_line(e: ic.Entry) -> tuple[str, tuple]:
    n, c = e.actual, e.counts
    if not c.get("all_agree"):
        return f"counts DISAGREE (see below)", BAD_RED
    if n is None:
        return "count unavailable", BAD_RED
    if n > e.budget:
        return f"{n} triangles: OVER BUDGET", BAD_RED
    if n == e.budget:
        return f"{n} triangles (validated)", OK_GREEN
    return f"{n} triangles (validated, {e.budget - n} unused)", INK


def ref_crop_box(entries, W=1536, H=1536, margin=0.05):
    R = load_mask(paths.MASK_PNG)
    boxes = [ic.bbox_of(R)] + [ic.bbox_of(load_mask(e.render("ref_silhouette.png"))) for e in entries.values()]
    return ic.expand_box(ic.union_bbox(boxes), margin, W, H)


def sheet_reference_plus_four(entries, out: Path, cell_w: int, box) -> Path:
    bw, bh = box[2] - box[0], box[3] - box[1]
    cell_h = round(cell_w * bh / bw)
    ref = load_rgb(paths.REFERENCE_PNG)
    cells = [[crop_cell(ref, box, cell_w, cell_h)] +
             [crop_cell(load_rgb(e.render("ref_beauty.png")), box, cell_w, cell_h) for e in entries.values()]]
    fh, fs = font(34, "bold"), font(26)
    heads = [[("Reference photo", fh, INK), ("MiniMax H3 Image, fixed input", fs, MUTED)]]
    for e in entries.values():
        t, col = actual_line(e)
        heads.append([(f"{e.budget}-triangle budget{tag_suffix(e)}", fh, INK), (t, fs, col)])
    footer = [("Beauty renders: flat per-face material colours, flat shading, identical camera / lights / "
               "crop for every column. Triangle counts are the validated evaluated-mesh counts "
               "(mesh.json = .blend evaluated = GLB reimport = GLB parse).", font(22), MUTED)]
    img = compose("One cat, four triangle budgets", "Reference camera view, beauty renders", heads, [], cells,
                  cell_w, cell_h, footer=footer, warn=provisional_warning(entries))
    return save_png(img, out / "comparison_reference_plus_four.png")


def diff_image(R: np.ndarray, M: np.ndarray) -> Image.Image:
    a = np.zeros(R.shape + (3,), np.uint8)
    a[:] = ic.DIFF_BG
    a[R & M] = ic.DIFF_AGREE
    a[M & ~R] = ic.DIFF_OUTSIDE
    a[R & ~M] = ic.DIFF_UNCOVERED
    return Image.fromarray(a, "RGB")


def sheet_clay_wire(entries, out: Path, cell_w: int, box) -> Path:
    bw, bh = box[2] - box[0], box[3] - box[1]
    cell_h = round(cell_w * bh / bw)
    ref = load_rgb(paths.REFERENCE_PNG)
    R = load_mask(paths.MASK_PNG)
    mask_img = diff_image(R, R)
    col_ref = [crop_cell(ref, box, cell_w, cell_h), crop_cell(ref, box, cell_w, cell_h),
               crop_cell(mask_img, box, cell_w, cell_h)]
    cols = [col_ref]
    for e in entries.values():
        M = load_mask(e.render("ref_silhouette.png"))
        cols.append([crop_cell(load_rgb(e.render("ref_clay.png")), box, cell_w, cell_h),
                     crop_cell(load_rgb(e.render("ref_wire.png")), box, cell_w, cell_h),
                     crop_cell(diff_image(R, M), box, cell_w, cell_h)])
    cells = [[cols[c][r] for c in range(len(cols))] for r in range(3)]
    fh, fs, fb = font(36, "bold"), font(24), font(30, "bold")
    heads = [[("Reference", fh, INK), ("photo · photo · mask", fs, MUTED)]]
    feet = [[("reference/cat.png", fs, MUTED), ("reference/derived/mask.png", fs, MUTED)]]
    for e in entries.values():
        heads.append([(f"Budget {e.budget}{tag_suffix(e)}", fh, INK), (actual_line(e)[0], fs, actual_line(e)[1])])
        c = e.counts
        agree = bool(c.get("all_agree")) and c.get("final_triangles") is not None and c["final_triangles"] <= e.budget
        feet.append([
            (f"Actual triangles: {c.get('final_triangles')} / {e.budget}", fb, OK_GREEN if agree else BAD_RED),
            (f"mesh.json {c.get('A_mesh_json')}  ·  evaluated .blend {c.get('B_blend_evaluated')}", fs, INK),
            (f"reimported GLB {c.get('C_glb_reimported')}  ·  parsed GLB {c.get('D_glb_parsed')}", fs, INK),
            (("all four counts agree, within budget" if agree else "COUNTS DISAGREE OR OVER BUDGET"), fs,
             OK_GREEN if agree else BAD_RED),
        ])
    fr = font(30, "bold")
    rows = [[("Clay", fr, INK), ("flat shaded", font(22), MUTED)],
            [("Wireframe", fr, INK), ("every triangle", font(22), MUTED), ("edge, hidden", font(22), MUTED),
             ("lines removed", font(22), MUTED)],
            [("Silhouette", fr, INK), ("difference", fr, INK), ("reference mask", font(22), MUTED),
             ("vs rendered", font(22), MUTED)]]
    legend = [("silhouettes agree", ic.DIFF_AGREE), ("rendered outside the reference mask", ic.DIFF_OUTSIDE),
              ("reference not covered by the render", ic.DIFF_UNCOVERED)]
    footer = [(f"Same {ic.camera_status(entries)} reference camera, lighting and crop in every column. Rendered mask = "
               "official Blender silhouette render (renders/ref_silhouette.png), binarised at 128.", font(22), MUTED)]
    img = compose("Clay and wireframe at the reference view", "Actual validated triangle counts printed per budget",
                  heads, rows, cells, cell_w, cell_h, col_feet=feet, legend=legend, footer=footer,
                  row_label_w=230, warn=provisional_warning(entries))
    return save_png(img, out / "comparison_clay_wire.png")


def view_names(entries) -> list[str]:
    have = None
    for e in entries.values():
        s = {p.name[len("view_"):-len("_clay.png")] for p in e.renders.glob("view_*_clay.png")}
        have = s if have is None else have & s
    have = have or set()
    order = [v for v in ic.VIEW_ORDER if v in have] + sorted(have - set(ic.VIEW_ORDER))
    return order


def view_crops(entries, views, aspect: float, margin=0.06) -> dict[str, list[int]]:
    crops = {}
    for v in views:
        boxes = []
        size = None
        for e in entries.values():
            im = load_rgb(e.render(f"view_{v}_wire.png"))
            size = im.size
            boxes.append(ic.bbox_of(ic.geometry_mask_from_wire(im)))
        crops[v] = ic.expand_box(ic.union_bbox(boxes), margin, size[0], size[1], aspect=aspect)
    return crops


def sheet_multiview(entries, out: Path, kind: str, cell_w: int, views, crops) -> Path:
    cell_h = round(cell_w / VIEW_ASPECT)
    cells = []
    for v in views:
        cells.append([crop_cell(load_rgb(e.render(f"view_{v}_{kind}.png")), crops[v], cell_w, cell_h)
                      for e in entries.values()])
    fh, fs = font(34, "bold"), font(24)
    heads = []
    for e in entries.values():
        t, col = actual_line(e)
        heads.append([(f"Budget {e.budget}{tag_suffix(e)}", fh, INK), (t, fs, col)])
    rows = []
    for v in views:
        name = ic.VIEW_LABEL.get(v, v)
        parts = [name[:len("three-quarter")], name[len("three-quarter") + 1:]] if name.startswith("three-quarter") else [name]
        lab = [(t, font(28, "bold"), INK) for t in parts]
        if v in ic.INFERRED_VIEWS:
            lab += [("INFERRED", font(22, "bold"), ic.DIFF_OUTSIDE), ("cat's left side is", font(21), MUTED),
                    ("not visible in the", font(21), MUTED), ("reference photo", font(21), MUTED)]
        elif v in ("back", "top"):
            lab += [("mostly inferred", font(21), MUTED)]
        rows.append(lab)
    what = "clay (flat shaded)" if kind == "clay" else "clay + every triangle edge (hidden lines removed)"
    footer = [(f"Fixed novel views, orbit at the reference camera distance, 85 mm. One crop box per view (union of "
               "the four silhouettes + margin) shared by all budgets, so sizes are comparable.", font(22), MUTED)]
    img = compose(f"Novel views: {what}", "Rows: fixed views (CONTRACT §8).  Columns: triangle budgets.",
                  heads, rows, cells, cell_w, cell_h, footer=footer, row_label_w=290,
                  warn=provisional_warning(entries))
    return save_png(img, out / f"comparison_multiview_{kind}.png")


VIEW_ASPECT = 1.25


# --------------------------------------------------------------------------- metrics table
COLS = [  # key, header, sub, digits, better
    ("tris", "Tris", "actual", 0, None),
    ("iou", "IoU", "higher", 4, "hi"),
    ("boundary_mean_px", "Bnd mean", "px, lower", 1, "lo"),
    ("boundary_hd95_px", "Bnd HD95", "px, lower", 1, "lo"),
    ("landmark_sharp_mean_px", "Sharp LM", "px, lower", 1, "lo"),
    ("region_iou_head", "Head IoU", "higher", 3, "hi"),
    ("region_iou_tail", "Tail IoU", "higher", 3, "hi"),
    ("region_iou_legs", "Legs IoU", "higher", 3, "hi"),
    ("region_iou_torso", "Torso IoU", "higher", 3, "hi"),
    ("legs_gap_iou", "Leg gaps", "IoU, higher", 3, "hi"),
    ("novel_view_iou_vs_proxy", "Novel IoU", "vs proxy*", 3, "hi"),
    ("chamfer_mm_vs_proxy", "Chamfer", "mm vs proxy*", 1, "lo"),
]


def table_rows(entries, baselines_root: Path, proxy_report: Path | None):
    rows = []
    for e in entries.values():
        init, tag = ic.initial_entry(e.vdir)
        h0 = ic.headline(init)
        h0["tris"] = (init or {}).get("triangles", {}).get("mesh_json")
        h1 = ic.headline(e.metrics)
        h1["tris"] = e.actual
        rows.append({"group": e.budget, "label": f"initial ({tag})", "vals": h0, "kind": "initial"})
        rows.append({"group": e.budget, "label": "final" if e.label == "final" else f"selected ({e.label})",
                     "vals": h1, "kind": "final", "base": h0})
        for k, m in ic.baselines_for(baselines_root, e.budget).items():
            hb = ic.headline(m)
            hb["tris"] = (m.get("triangles") or {}).get("mesh_json")
            rows.append({"group": e.budget, "label": f"baseline: {k}", "vals": hb, "kind": "baseline"})
    if proxy_report and proxy_report.is_file():
        pr = json.loads(proxy_report.read_text())
        met = pr.get("metrics") or pr.get("official_metrics") or pr
        hp = {k: met.get(k) if isinstance(met, dict) else None for k, *_ in COLS}
        if isinstance(met, dict) and isinstance(met.get("region_iou"), dict):
            for r in ("head", "tail", "legs", "torso"):
                hp[f"region_iou_{r}"] = met["region_iou"].get(r)
        rows.append({"group": "proxy", "label": "inferred high-res proxy", "vals": hp, "kind": "proxy"})
    return rows


def sheet_metrics_table(entries, out: Path, baselines_root: Path, proxy_report: Path | None) -> Path:
    rows = table_rows(entries, baselines_root, proxy_report)
    f_h, f_s, f_c, f_cb, f_g = font(26, "bold"), font(20), font(26), font(26, "bold"), font(28, "bold")
    col_w = [150, 320] + [158] * len(COLS)
    row_h, head_h = 54, 86
    tmp = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    W = PAD * 2 + sum(col_w)
    title_h = 150
    warn = provisional_warning(entries)
    foot_lines = [
        "Official metrics: Blender silhouette render through the shared camera at 1536 px (CONTRACT §7). "
        "Bnd = symmetric boundary distance; Sharp LM = mean error of the 12 sharp landmarks.",
        "Region IoU inside the fixed windows of reference/derived/regions.json; Leg gaps = IoU of the background "
        "inside the legs window (negative space).",
        "* Inferred proxy geometry is NOT ground truth (secondary diagnostic). Green / red final values: better / "
        "worse than that budget's initial checkpoint.",
    ]
    prov = set((e.metrics.get("provenance") or {}).get("tooling_hash") for e in entries.values())
    bl_docs = [m for e in entries.values() for m in ic.baselines_for(baselines_root, e.budget).values()]
    if bl_docs:
        bm = ic.provenance_mismatch([next(iter(entries.values())).metrics] + bl_docs)
        foot_lines.append("Baseline rows: " + ("identical hashes to the selected rows." if not bm else
                          "scored with different " + ", ".join(bm) + " - NOT comparable."))
    foot_lines.append(f"Camera status: {ic.camera_status(entries)}.  Tooling hash(es): "
                      + ", ".join(sorted(str(p)[:12] for p in prov)) + ".")
    H = PAD + title_h + (52 if warn else 0) + head_h + row_h * len(rows) + 30 + 36 * len(foot_lines) + PAD
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((PAD, PAD), "Official metrics per budget: initial vs final", font=font(46, "bold"), fill=INK)
    mism = ic.provenance_mismatch([e.metrics for e in entries.values()] + [ic.initial_entry(e.vdir)[0]
                                                                            for e in entries.values()])
    if mism:
        d.text((PAD, PAD + 66), "NOT COMPARABLE: rows were scored with different " + ", ".join(mism)
               + " (re-evaluate before comparing)", font=font(26, "bold"), fill=BAD_RED)
    else:
        d.text((PAD, PAD + 66), "Verified identical reference, mask, landmarks, regions, camera, scene and tooling "
               "hashes for every budget row (baselines: see footnote)", font=font(26), fill=MUTED)
    y = PAD + title_h
    if warn:
        d.rectangle([PAD, y - 20, W - PAD, y + 20], fill=(253, 236, 234))
        d.text((PAD + 14, y), warn, font=font(24, "bold"), fill=BAD_RED, anchor="lm")
        y += 52
    # header
    x = PAD
    heads = [("Budget", ""), ("Row", "")] + [(h, s) for _, h, s, _, _ in COLS]
    for (h, s), w in zip(heads, col_w):
        d.text((x + w - 14 if x > PAD + 300 else x + 8, y + 10), h, font=f_h, fill=INK,
               anchor="ra" if x > PAD + 300 else "la")
        if s:
            d.text((x + w - 14, y + 46), s, font=f_s, fill=MUTED, anchor="ra")
        x += w
    y += head_h
    d.line([PAD, y - 6, W - PAD, y - 6], fill=INK, width=2)
    prev = None
    for i, r in enumerate(rows):
        if r["group"] != prev and prev is not None:
            d.line([PAD, y, W - PAD, y], fill=RULE, width=2)
        if r["kind"] == "final":
            d.rectangle([PAD, y + 2, W - PAD, y + row_h - 2], fill=(239, 243, 248))
        x = PAD
        if r["group"] != prev:
            d.text((x + 8, y + row_h // 2), str(r["group"]), font=f_g, fill=INK, anchor="lm")
        prev = r["group"]
        x += col_w[0]
        d.text((x + 8, y + row_h // 2), r["label"], font=f_cb if r["kind"] == "final" else f_c,
               fill=INK if r["kind"] != "baseline" else MUTED, anchor="lm")
        x += col_w[1]
        for (k, _, _, nd, better), w in zip(COLS, col_w[2:]):
            v = r["vals"].get(k)
            colr = INK if r["kind"] != "baseline" else MUTED
            if r["kind"] == "final" and better and v is not None and r["base"].get(k) is not None:
                b = r["base"][k]
                if abs(v - b) > 1e-9:
                    colr = OK_GREEN if ((v > b) == (better == "hi")) else BAD_RED
            d.text((x + w - 14, y + row_h // 2), ic.fmt(v, nd), font=f_cb if r["kind"] == "final" else f_c,
                   fill=colr, anchor="rm")
            x += w
        y += row_h
    d.line([PAD, y, W - PAD, y], fill=INK, width=2)
    y += 24
    for t in foot_lines:
        d.text((PAD, y), t, font=font(22), fill=MUTED)
        y += 36
    return save_png(img, out / "comparison_metrics_table.png")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the four-budget comparison sheets.")
    ap.add_argument("--variants-root", type=Path, default=paths.VARIANTS)
    ap.add_argument("--out", type=Path, default=paths.OUTPUTS)
    ap.add_argument("--checkpoints", default=None, help="mid-run: 100:TAG,200:TAG,500:TAG,1000:TAG instead of final/")
    ap.add_argument("--baselines-root", type=Path, default=paths.BASELINES)
    ap.add_argument("--proxy-report", type=Path, default=ROOT / "proxy" / "fit_report.json")
    ap.add_argument("--cell", type=int, default=900, help="cell width of the reference-view sheets (px)")
    ap.add_argument("--view-cell", type=int, default=640, help="cell width of the multiview sheets (px)")
    a = ap.parse_args(argv)
    entries = ic.load_entries(a.variants_root, a.checkpoints)
    for e in entries.values():
        need = ["ref_silhouette.png", "ref_clay.png", "ref_wire.png", "ref_beauty.png"]
        miss = [n for n in need if not e.render(n).is_file()]
        if miss:
            raise SystemExit(f"budget {e.budget} ({e.dir}): missing renders {miss}")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    box = ref_crop_box(entries)
    written = [sheet_reference_plus_four(entries, out, a.cell, box), sheet_clay_wire(entries, out, a.cell, box)]
    views = view_names(entries)
    crops = {}
    if len(views) < 8:
        print(f"WARNING: only {len(views)} novel views common to all four budgets: {views}", file=sys.stderr)
    if views:
        crops = view_crops(entries, views, VIEW_ASPECT)
        for kind in ("clay", "wire"):
            written.append(sheet_multiview(entries, out, kind, a.view_cell, views, crops))
    else:
        print("WARNING: no novel views rendered (evaluate with --views); multiview sheets skipped", file=sys.stderr)
    written.append(sheet_metrics_table(entries, out, a.baselines_root, a.proxy_report))
    manifest = {
        "sources": {N: {"dir": str(e.dir), "label": e.label, "mesh_sha256": e.metrics["provenance"]["mesh_sha256"],
                        "triangles": e.counts} for N, e in entries.items()},
        "reference_crop_box": box, "view_crop_boxes": crops, "camera_status": ic.camera_status(entries),
        "files": [str(p) for p in written],
    }
    (out / "comparison_manifest.json").write_text(json.dumps(manifest, indent=1))
    for p in written:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
