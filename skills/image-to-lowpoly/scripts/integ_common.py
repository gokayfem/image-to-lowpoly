"""Shared helpers for the integration scripts (finalize / comparison / critic packet /
aggregate / comparison blend / turntable mosaic).  Analysis venv only (numpy + pillow).

Not part of the official tooling hash: nothing here changes a metric, it only reads the
per-checkpoint files of CONTRACT sections 6-9 and lays them out.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the folder that holds catlib/
from catlib import paths as _paths  # noqa: E402  (resolves the case: --case / LOWPOLY_CASE / cwd)
ROOT = _paths.ROOT

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from catlib import paths  # noqa: E402

BUDGETS = (100, 200, 500, 1000)
VIEW_ORDER = ("front", "side_R", "threeq_back_R", "back", "threeq_back_L", "side_L", "threeq_front_L", "top")
VIEW_LABEL = {
    "front": "front", "side_R": "right side", "back": "back", "side_L": "left side",
    "threeq_front_L": "three-quarter front left", "threeq_back_L": "three-quarter back left",
    "threeq_back_R": "three-quarter back right", "top": "top",
}
INFERRED_VIEWS = {"side_L", "threeq_front_L", "threeq_back_L"}  # the cat's left side is never seen

# palette (sRGB 8-bit)
BG = (250, 250, 248)
INK = (28, 28, 30)
MUTED = (110, 110, 116)
RULE = (214, 214, 218)
OK_GREEN = (36, 125, 70)
BAD_RED = (190, 40, 40)
DIFF_AGREE = (150, 150, 150)
DIFF_OUTSIDE = (230, 97, 1)      # rendered, not in reference (orange)
DIFF_UNCOVERED = (44, 123, 182)  # reference, not rendered (blue)
DIFF_BG = (255, 255, 255)


# --------------------------------------------------------------------------- fonts / drawing
_FONT_FILES = [("/System/Library/Fonts/HelveticaNeue.ttc", {"regular": 0, "bold": 1, "medium": 10, "light": 7}),
               ("/System/Library/Fonts/Helvetica.ttc", {"regular": 0, "bold": 1, "medium": 1, "light": 0})]
_font_cache: dict = {}


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    key = (size, weight)
    if key not in _font_cache:
        f = None
        for path, idx in _FONT_FILES:
            try:
                f = ImageFont.truetype(path, size, index=idx.get(weight, 0))
                break
            except OSError:
                continue
        _font_cache[key] = f or ImageFont.load_default(size)
    return _font_cache[key]


def text_size(draw: ImageDraw.ImageDraw, s: str, f) -> tuple[int, int]:
    x0, y0, x1, y1 = draw.textbbox((0, 0), s, font=f)
    return x1 - x0, y1 - y0


def draw_text(draw, xy, s, f, fill=INK, anchor="la"):
    draw.text(xy, s, font=f, fill=fill, anchor=anchor)


def draw_centered(draw, cx, y, s, f, fill=INK):
    draw.text((cx, y), s, font=f, fill=fill, anchor="ma")


def save_png(img: Image.Image, path: Path) -> Path:
    """Save without any text chunks / metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = img.convert("RGB")
    clean = Image.new("RGB", img.size)
    clean.paste(img)
    clean.save(path, format="PNG", optimize=True)
    return path


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) >= 128


# --------------------------------------------------------------------------- geometry of crops
def bbox_of(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def union_bbox(boxes):
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def expand_box(b, margin_frac: float, W: int, H: int, aspect: float | None = None):
    """Grow box by margin_frac of its larger side; optionally pad to width/height == aspect;
    clamp into the image (shifting rather than shrinking when possible)."""
    u0, v0, u1, v1 = b
    m = margin_frac * max(u1 - u0, v1 - v0)
    u0, v0, u1, v1 = u0 - m, v0 - m, u1 + m, v1 + m
    if aspect:
        w, h = u1 - u0, v1 - v0
        if w / h < aspect:
            d = (h * aspect - w) / 2; u0 -= d; u1 += d
        else:
            d = (w / aspect - h) / 2; v0 -= d; v1 += d
    w, h = u1 - u0, v1 - v0
    w, h = min(w, W), min(h, H)
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    u0 = min(max(cu - w / 2, 0), W - w)
    v0 = min(max(cv - h / 2, 0), H - h)
    return [int(round(u0)), int(round(v0)), int(round(u0 + w)), int(round(v0 + h))]


def geometry_mask_from_wire(img: Image.Image, thresh: int = 75) -> np.ndarray:
    """Pixels of the mesh in a wire render: the wire overlay draws every triangle edge,
    including the silhouette outline, in near-black; shadows / background never get that dark."""
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    return a.max(axis=2) < thresh


def fit(img: Image.Image, w: int, h: int, bg=BG) -> Image.Image:
    """Letterbox img into w x h, high-quality resample."""
    s = min(w / img.width, h / img.height)
    nw, nh = max(1, round(img.width * s)), max(1, round(img.height * s))
    r = img.resize((nw, nh), Image.LANCZOS)
    out = Image.new("RGB", (w, h), bg)
    out.paste(r, ((w - nw) // 2, (h - nh) // 2))
    return out


def outline(mask: np.ndarray, width: int = 3) -> np.ndarray:
    from scipy.ndimage import binary_dilation, binary_erosion
    edge = mask & ~binary_erosion(mask)
    return binary_dilation(edge, iterations=max(0, width // 2)) if width > 1 else edge


# --------------------------------------------------------------------------- data access
def read_json(p: Path):
    return json.loads(Path(p).read_text())


def parse_checkpoints(spec: str | None) -> dict[int, str]:
    if not spec:
        return {}
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+):([A-Za-z0-9_.\-]+)", part)
        if not m:
            raise SystemExit(f"bad --checkpoints entry {part!r}; expected N:TAG")
        out[int(m.group(1))] = m.group(2)
    return out


@dataclass
class Entry:
    budget: int
    vdir: Path
    dir: Path                 # final/ or checkpoints/<tag>/
    label: str                # "final" or the checkpoint tag
    metrics: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)

    @property
    def renders(self) -> Path:
        return self.dir / "renders"

    @property
    def counts(self) -> dict:
        return self.validation.get("counts", {})

    @property
    def actual(self):
        return self.counts.get("final_triangles")

    def render(self, name: str) -> Path:
        return self.renders / name


def load_entry(variants_root: Path, N: int, tag: str | None) -> Entry:
    vdir = Path(variants_root) / str(N)
    d = vdir / "final" if tag in (None, "final") else vdir / "checkpoints" / tag
    if not d.is_dir():
        raise SystemExit(f"budget {N}: {d} does not exist")
    mp, vp = d / "metrics.json", d / "validation.json"
    if not mp.is_file() or not vp.is_file():
        raise SystemExit(f"budget {N}: {d} lacks metrics.json / validation.json (not evaluated?)")
    return Entry(N, vdir, d, "final" if tag in (None, "final") else tag, read_json(mp), read_json(vp))


def load_entries(variants_root: Path, checkpoints: str | None, budgets=BUDGETS) -> dict[int, Entry]:
    spec = parse_checkpoints(checkpoints)
    if spec and set(spec) != set(budgets):
        missing = sorted(set(budgets) - set(spec))
        raise SystemExit(f"--checkpoints must name all budgets {list(budgets)}; missing {missing}")
    return {N: load_entry(variants_root, N, spec.get(N)) for N in budgets}


def initial_entry(vdir: Path) -> tuple[dict | None, str]:
    """The first official evaluation of this budget: checkpoints/v0_initial if it has official
    metrics, otherwise the first official history row that still has a metrics.json."""
    ck = vdir / "checkpoints" / "v0_initial" / "metrics.json"
    if ck.is_file():
        m = read_json(ck)
        if m.get("official"):
            return m, "v0_initial"
    hp = vdir / "history.json"
    if hp.is_file():
        for row in read_json(hp):
            if not row.get("official") or row.get("tag") == "final":
                continue
            mp = ROOT / row["mesh"] if not Path(row["mesh"]).is_absolute() else Path(row["mesh"])
            mp = mp.parent / "metrics.json"
            if mp.is_file():
                m = read_json(mp)
                if m.get("official"):
                    return m, row["tag"]
    return None, "none"


def metric_doc(path_dir: Path) -> dict | None:
    """metrics.json for a baseline-style directory: <dir>/metrics.json, <dir>/final/metrics.json,
    else the newest checkpoints/*/metrics.json."""
    for p in (path_dir / "metrics.json", path_dir / "final" / "metrics.json"):
        if p.is_file():
            return {"path": str(p), **read_json(p)}
    cks = sorted((path_dir / "checkpoints").glob("*/metrics.json"), key=lambda p: p.stat().st_mtime)
    if cks:
        return {"path": str(cks[-1]), **read_json(cks[-1])}
    return None


def baselines_for(baselines_root: Path, N: int) -> dict[str, dict]:
    out = {}
    root = Path(baselines_root)
    if not root.is_dir():
        return out
    for kd in sorted(root.glob("decimate*")):
        if kd.is_dir() and (kd / str(N)).is_dir():
            m = metric_doc(kd / str(N))
            if m:
                out[kd.name] = m
    return out


def headline(m: dict | None) -> dict:
    """Flat dict of the headline official metrics from a metrics.json doc (None-safe)."""
    met = (m or {}).get("metrics") or {}
    sec = met.get("secondary") or {}
    reg = met.get("region_iou") or {}
    return {
        "iou": met.get("iou"),
        "boundary_mean_px": met.get("boundary_mean_px"),
        "boundary_hd95_px": met.get("boundary_hd95_px"),
        "boundary_max_px": met.get("boundary_max_px"),
        "boundary_mean_norm": met.get("boundary_mean_norm"),
        "ref_uncovered_frac": met.get("ref_uncovered_frac"),
        "render_outside_frac": met.get("render_outside_frac"),
        "landmark_sharp_mean_px": met.get("landmark_sharp_mean_px"),
        "landmark_sharp_max_px": met.get("landmark_sharp_max_px"),
        "landmark_sharp_mean_norm": met.get("landmark_sharp_mean_norm"),
        "landmark_fuzzy_mean_px": met.get("landmark_fuzzy_mean_px"),
        "region_iou_head": reg.get("head"),
        "region_iou_tail": reg.get("tail"),
        "region_iou_legs": reg.get("legs"),
        "region_iou_torso": reg.get("torso"),
        "legs_gap_iou": met.get("legs_gap_iou"),
        "novel_view_iou_vs_proxy": sec.get("novel_view_iou_vs_proxy"),
        "chamfer_mm_vs_proxy": sec.get("chamfer_mm_vs_proxy"),
        "raster_vs_blender_iou": met.get("raster_vs_blender_iou"),
    }


def fmt(x, nd=3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{nd}f}"


def count_line(e: Entry) -> tuple[str, bool]:
    c = e.counts
    agree = bool(c.get("all_agree"))
    s = (f"mesh.json {c.get('A_mesh_json')} · .blend {c.get('B_blend_evaluated')} · "
         f"GLB reimport {c.get('C_glb_reimported')} · GLB parse {c.get('D_glb_parsed')}")
    return s, agree


def camera_status(entries: dict[int, Entry]) -> str:
    st = {e.metrics.get("camera_status") for e in entries.values()}
    return st.pop() if len(st) == 1 else "MIXED:" + ",".join(sorted(map(str, st)))


HASH_KEYS = ("reference_sha256", "mask_sha256", "landmarks_sha256", "regions_sha256", "camera_sha256",
             "scene_sha256", "tooling_hash")


def provenance_mismatch(docs) -> list[str]:
    """Names of the provenance hashes that are not identical across the given metrics.json docs."""
    docs = [d for d in docs if d]
    bad = []
    for k in HASH_KEYS:
        if len({(d.get("provenance") or {}).get(k) for d in docs}) > 1:
            bad.append(k)
    return bad
