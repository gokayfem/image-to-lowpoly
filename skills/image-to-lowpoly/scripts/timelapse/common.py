"""Shared pieces of the cat release film and the build timelapse (CONTRACT §13).

* Compositor helpers (font, text, pills, tags, rounded frames, wipes, background) are ADAPTED from the earlier
  "Blockout 2 Real" film, the Blockout 2 Real project (gokayfem/blockout-to-real), release_video/
  make_film.py (read-only source; copied and trimmed here, same typography and colour language).
* Checkpoint selection: the final (variants/<N>/final/) when it exists, else a STAND-IN (worker BEST.json, else
  the highest official IoU in history.json). Every stand-in is recorded in STANDINS so nothing is passed off as final.
* `overlay_mesh()`: the CPU mesh-on-photo renderer. Projection is catlib.camera (the locked camera, CONTRACT §3);
  hidden lines are removed with catlib.raster's z-buffered triangle-ID image, edges are where the ID changes.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # scripts/ (holds catlib/)
from catlib import paths as _paths  # noqa: E402  (the case: --case / LOWPOLY_CASE / cwd)
ROOT = _paths.ROOT

from catlib import raster  # noqa: E402
from catlib.camera import Camera  # noqa: E402

OUT = ROOT / "outputs" / "timelapse"
WORK = ROOT / ".cache" / "timelapse"
FOOT = ROOT / ".cache" / "footage"
CACHE = WORK / "cache"
BUDGETS = (100, 200, 500, 1000)
W, H, FPS = 1920, 1080, 24
PHOTO = _paths.REFERENCE_PNG
DERIVED = ROOT / "reference" / "derived"

# One crop of the 1536² reference used by every photo panel (6:5, contains the whole mask bbox + windows).
CROP = (36, 268, 1440, 1438)
CROP_W, CROP_H = CROP[2] - CROP[0], CROP[3] - CROP[1]

# ------------------------------------------------------------------ colours (same language as the earlier film)
WHITE = (255, 255, 255)
CYAN = (32, 211, 238)
GREY = (150, 163, 186)
DIM = (104, 116, 138)
SOFT = (214, 223, 237)
AMBER = (243, 178, 84)
RED = (236, 112, 112)
GREEN = (120, 220, 150)
PANEL = (16, 23, 38)

# tri_region tints for the overlay: cool tones so they separate from the orange fur
REGION_RGB = {
    "torso": (32, 211, 238), "neck": (70, 175, 250), "head": (120, 140, 255), "ears": (178, 128, 255),
    "eyes": (245, 245, 245), "nose": (255, 120, 170), "leg_FR": (60, 225, 170), "leg_FL": (40, 190, 150),
    "leg_HR": (110, 230, 120), "leg_HL": (80, 200, 100), "tail": (250, 196, 90),
}
REGION_ORDER = ["torso", "neck", "head", "ears", "eyes", "nose", "leg_FR", "leg_FL", "leg_HR", "leg_HL", "tail"]

STANDINS: list[str] = []      # every stand-in / missing final, listed by make_film and timelapse
PLACEHOLDERS: list[str] = []  # inputs that are missing outright (drawn as visible placeholders)


def note_standin(msg):
    if msg not in STANDINS:
        STANDINS.append(msg)


def note_placeholder(msg):
    if msg not in PLACEHOLDERS:
        PLACEHOLDERS.append(msg)


def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def read_json(p):
    return json.loads(Path(p).read_text())


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ small maths
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def ss(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def eo(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def appear(t, t0, d=0.25):
    """An entrance centred on t0 (a beat): half there on the beat."""
    return ss((t - t0 + d / 2) / d)


def pingpong(k, n):
    k %= 2 * n - 2
    return k if k < n else 2 * n - 2 - k


# ------------------------------------------------------------------ fonts and text (adapted from the earlier film)
FONT_VF = Path("/System/Library/Fonts/SFNS.ttf")
WGHT = {"B": 700, "M": 600, "R": 400}
PAD = 6


@functools.lru_cache(maxsize=None)
def font(weight, size):
    if not FONT_VF.exists():
        raise SystemExit(f"font missing: {FONT_VF} (no fallback is substituted)")
    f = ImageFont.truetype(str(FONT_VF), size)
    vals = []
    for ax in f.get_variation_axes():
        n = ax["name"].decode() if isinstance(ax["name"], bytes) else ax["name"]
        if n == "Weight":
            vals.append(WGHT[weight])
        elif n == "Optical Size":
            vals.append(min(max(size, ax["minimum"]), ax["maximum"]))
        else:
            vals.append(ax["default"])
    f.set_variation_by_axes(vals)
    return f


@functools.lru_cache(maxsize=8192)
def text_w(text, size, weight="R"):
    return font(weight, size).getlength(text)


@functools.lru_cache(maxsize=4096)
def text_mask(text, size, weight):
    f = font(weight, size)
    asc, desc = f.getmetrics()
    im = Image.new("L", (int(math.ceil(text_w(text, size, weight))) + 2 * PAD, asc + desc + 2 * PAD), 0)
    ImageDraw.Draw(im).text((PAD, PAD), text, font=f, fill=255)
    return im


def scaled(mask, a):
    if a >= 0.999:
        return mask
    return mask.point(lambda v: int(v * a + 0.5))


def draw_text(c, text, x, y, size, weight="R", color=WHITE, a=1.0, anchor="l", rise=0.0):
    """y is the top of the line box."""
    if a <= 0.003 or not text:
        return
    m = text_mask(text, size, weight)
    tw = text_w(text, size, weight)
    x0 = x - PAD - (tw if anchor == "r" else tw / 2 if anchor == "c" else 0)
    c.paste(color, (int(round(x0)), int(round(y - PAD + rise))), scaled(m, a))


def draw_text_in(c, text, x, y, size, weight, color, t, t0, anchor="l", d=0.25):
    a = appear(t, t0, d)
    draw_text(c, text, x, y, size, weight, color, a, anchor, rise=(1 - a) * 12)


@functools.lru_cache(maxsize=512)
def rmask(w, h, r):
    s = 4
    im = Image.new("L", (w * s, h * s), 0)
    ImageDraw.Draw(im).rounded_rectangle((0, 0, w * s - 1, h * s - 1), r * s, fill=255)
    return im.resize((w, h), Image.LANCZOS)


@functools.lru_cache(maxsize=512)
def outline_mask(w, h, r, width):
    s = 4
    im = Image.new("L", (w * s, h * s), 0)
    ImageDraw.Draw(im).rounded_rectangle((0, 0, w * s - 1, h * s - 1), r * s, outline=255, width=width * s)
    return im.resize((w, h), Image.LANCZOS)


@functools.lru_cache(maxsize=128)
def shadow_mask(w, h, r, pad=60, blur=26):
    im = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
    ImageDraw.Draw(im).rounded_rectangle((pad, pad, pad + w - 1, pad + h - 1), r, fill=255)
    return im.filter(ImageFilter.GaussianBlur(blur))


@functools.lru_cache(maxsize=64)
def dot_mask(d):
    s = 4
    im = Image.new("L", (d * s, d * s), 0)
    ImageDraw.Draw(im).ellipse((0, 0, d * s - 1, d * s - 1), fill=255)
    return im.resize((d, d), Image.LANCZOS)


@functools.lru_cache(maxsize=64)
def ring_mask(d, width):
    s = 4
    im = Image.new("L", (d * s, d * s), 0)
    ImageDraw.Draw(im).ellipse((0, 0, d * s - 1, d * s - 1), outline=255, width=width * s)
    return im.resize((d, d), Image.LANCZOS)


def rect(c, x, y, w, h, r, color, a=1.0):
    if a <= 0.003 or w < 1 or h < 1:
        return
    c.paste(color, (int(x), int(y)), scaled(rmask(int(w), int(h), int(r)), a))


def outline(c, x, y, w, h, r, color, a=1.0, width=2):
    if a <= 0.003:
        return
    c.paste(color, (int(x), int(y)), scaled(outline_mask(int(w), int(h), int(r), width), a))


def paste_frame(c, img, x, y, r=14, a=1.0, shadow=True):
    if a <= 0.003:
        return
    w, h = img.size
    if shadow:
        pad = 60
        c.paste((1, 2, 5), (int(x) - pad, int(y) - pad + 16), scaled(shadow_mask(w, h, r), 0.55 * a))
    c.paste(img, (int(x), int(y)), scaled(rmask(w, h, r), a))


def pill(c, text, x, y, a=1.0, anchor="c", size=24, dot=CYAN, bg=PANEL, bga=0.82, color=SOFT, border=True):
    """A rounded chip. (x, y) is the anchor point on its top edge. Returns its width."""
    tw = int(text_w(text, size, "R"))
    padx, h = 22, int(size * 1.95)
    w = padx * 2 + tw + (20 if dot else 0)
    if a <= 0.003:
        return w
    x0 = int(x - (w if anchor == "r" else w / 2 if anchor == "c" else 0))
    rect(c, x0, y, w, h, h // 2, bg, bga * a)
    if border:
        outline(c, x0, y, w, h, h // 2, WHITE, 0.10 * a, 1)
    tx = x0 + padx
    if dot:
        c.paste(dot, (tx, int(y + h / 2 - 5)), scaled(dot_mask(10), a))
        tx += 20
    asc, desc = font("R", size).getmetrics()
    draw_text(c, text, tx, y + (h - asc - desc) / 2 + 1, size, "R", color, a)
    return w


def tag(c, text, x, y, a=1.0, anchor="l", color=WHITE, size=22):
    if a <= 0.003:
        return
    tw = int(text_w(text, size, "M"))
    w, h = tw + 28, int(size * 1.82)
    x0 = int(x - (w if anchor == "r" else w / 2 if anchor == "c" else 0))
    rect(c, x0, y, w, h, 10, (8, 12, 20), 0.72 * a)
    asc, desc = font("M", size).getmetrics()
    draw_text(c, text, x0 + 14, y + (h - asc - desc) / 2 + 1, size, "M", color, a)


def wipe(a, b, p, line=True):
    """b enters from the left over a; p in 0..1."""
    if p <= 0:
        return a
    if p >= 1:
        return b
    w, h = a.size
    x = int(round(p * w))
    out = a.copy()
    out.paste(b.crop((0, 0, x, h)), (0, 0))
    if line:
        ImageDraw.Draw(out).rectangle((x - 1, 0, x + 1, h), fill=(235, 250, 255))
    return out


def make_bg():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = yy / (H - 1)
    top, bot = np.array([6, 9, 15], np.float32), np.array([15, 20, 31], np.float32)
    img = top[None, None] * (1 - v[..., None]) + bot[None, None] * v[..., None]
    d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.98) / (H * 0.9)) ** 2)
    img += (np.clip(1 - d, 0, 1) ** 2)[..., None] * np.array([4, 5, 9], np.float32)
    img += np.random.default_rng(7).uniform(-0.6, 0.6, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


@functools.lru_cache(maxsize=1)
def BG():
    return make_bg()


def placeholder_img(w, h, text):
    im = Image.new("RGB", (w, h), (28, 34, 48))
    d = ImageDraw.Draw(im)
    for x in range(-h, w, 28):
        d.line((x, 0, x + h, h), fill=(36, 44, 60), width=10)
    c = im
    draw_text(c, "PLACEHOLDER", w / 2, h / 2 - 40, 34, "M", AMBER, 1, "c")
    draw_text(c, text, w / 2, h / 2 + 8, 20, "R", SOFT, 1, "c")
    return im


# ------------------------------------------------------------------ images
@functools.lru_cache(maxsize=64)
def _open(path):
    return Image.open(path).convert("RGB")


@functools.lru_cache(maxsize=256)
def sized(path, w, h):
    """Image resized to cover (w, h), centre-cropped."""
    im = _open(str(path))
    s = max(w / im.width, h / im.height)
    nw, nh = max(w, round(im.width * s)), max(h, round(im.height * s))
    im = im.resize((nw, nh), Image.LANCZOS)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return im.crop((x0, y0, x0 + w, y0 + h))


@functools.lru_cache(maxsize=64)
def ref_crop(path, w, h):
    """A 1536² reference-view image (photo or a reference-view render) cropped to CROP and resized to (w, h)."""
    im = _open(str(path))
    if im.size != (1536, 1536):
        raise ValueError(f"{path}: expected a 1536² reference-view image, got {im.size}")
    return im.crop(CROP).resize((w, h), Image.LANCZOS)


def photo(w, h):
    return ref_crop(str(PHOTO), w, h)


def to_crop_px(uv, w):
    """Reference pixel-convention coords (1536²) -> panel pixel coords for a CROP panel of width w."""
    s = w / CROP_W
    uv = np.asarray(uv, float)
    return np.stack([(uv[..., 0] - CROP[0]) * s, (uv[..., 1] - CROP[1]) * s], -1)


# ------------------------------------------------------------------ checkpoints, metrics, stand-ins
def history(N):
    p = ROOT / "variants" / str(N) / "history.json"
    return read_json(p) if p.is_file() else []


def select_checkpoint(N, tag=None) -> dict:
    """{'N','tag','dir','mesh','status': 'final'|'stand-in','why'}. Records stand-ins in STANDINS."""
    vdir = ROOT / "variants" / str(N)
    pin = HERE / "finals.json"
    if tag is None and pin.is_file():
        tag = read_json(pin)["checkpoints"].get(str(N))
        if tag:
            d = vdir / "checkpoints" / tag
            if not (d / "mesh.json").is_file():
                raise SystemExit(f"budget {N}: pinned checkpoint {tag} has no mesh.json")
            return {"N": N, "tag": tag, "dir": d, "mesh": d / "mesh.json", "status": "final",
                    "why": "release_video/finals.json (pinned final)"}
    fin = vdir / "final"
    if tag is None and (fin / "mesh.json").is_file():
        ch = read_json(fin / "CHOSEN.json") if (fin / "CHOSEN.json").is_file() else {}
        return {"N": N, "tag": ch.get("checkpoint", "final"), "dir": fin, "mesh": fin / "mesh.json",
                "status": "final", "why": ch.get("reason", "variants/<N>/final")}
    if tag is None:
        b = vdir / "BEST.json"
        if b.is_file():
            tag, why = read_json(b)["checkpoint"], "worker BEST.json"
        else:
            rows = [r for r in history(N) if r.get("valid") and r.get("official")
                    and (r.get("triangles") or {}).get("all_agree")]
            if not rows:
                raise SystemExit(f"budget {N}: no final, no BEST.json and no valid official checkpoint")
            r = max(rows, key=lambda r: r["headline"]["iou"])
            tag, why = r["tag"], "highest official IoU in history.json (no BEST.json yet)"
    else:
        why = "--checkpoint"
    d = vdir / "checkpoints" / tag
    if not (d / "mesh.json").is_file():
        raise SystemExit(f"budget {N}: checkpoint {tag} has no mesh.json")
    note_standin(f"budget {N}: final not delivered yet -> stand-in checkpoint {tag} ({why})")
    return {"N": N, "tag": tag, "dir": d, "mesh": d / "mesh.json", "status": "stand-in", "why": why}


def headline_from(m: dict | None) -> dict:
    met = (m or {}).get("metrics") or {}
    sec = met.get("secondary") or {}
    return {"iou": met.get("iou"), "boundary_mean_px": met.get("boundary_mean_px"),
            "boundary_hd95_px": met.get("boundary_hd95_px"),
            "landmark_sharp_mean_px": met.get("landmark_sharp_mean_px"),
            "landmark_fuzzy_mean_px": met.get("landmark_fuzzy_mean_px"),
            "legs_gap_iou": met.get("legs_gap_iou"), "region_iou": met.get("region_iou") or {},
            "novel_view_iou_vs_proxy": sec.get("novel_view_iou_vs_proxy"),
            "chamfer_mm_vs_proxy": sec.get("chamfer_mm_vs_proxy"),
            "raster_vs_blender_iou": met.get("raster_vs_blender_iou")}


def official(sel: dict) -> dict:
    """Official numbers for a selected checkpoint: outputs/metrics.json when it covers this mesh, else the
    checkpoint's own metrics.json (official Blender-render numbers too, but not yet integrated)."""
    N = sel["N"]
    mesh_sha = sha256(sel["mesh"])
    agg = ROOT / "outputs" / "metrics.json"
    if agg.is_file():
        for v in read_json(agg).get("variants", []):
            if v.get("budget") == N and (v.get("final") or {}).get("mesh_sha256") == mesh_sha:
                fm = v["final"]["metrics"]
                t = v["triangles"]
                return {"source": "outputs/metrics.json", "iou": fm.get("iou"),
                        "boundary_mean_px": fm.get("boundary_mean_px"),
                        "boundary_hd95_px": fm.get("boundary_hd95_px"),
                        "landmark_sharp_mean_px": fm.get("landmark_sharp_mean_px"),
                        "landmark_fuzzy_mean_px": fm.get("landmark_fuzzy_mean_px"),
                        "legs_gap_iou": fm.get("legs_gap_iou"),
                        "region_iou": {k: fm.get(f"region_iou_{k}") for k in ("head", "tail", "legs", "torso")},
                        "novel_view_iou_vs_proxy": fm.get("novel_view_iou_vs_proxy"),
                        "chamfer_mm_vs_proxy": fm.get("chamfer_mm_vs_proxy"),
                        "counts": {"A": t.get("mesh_json"), "B": t.get("blend_evaluated"),
                                   "C": t.get("glb_reimported"), "D": t.get("glb_parsed"),
                                   "agree": t.get("all_agree")},
                        "baselines": {k: {"metrics": b.get("metrics"), "triangles": b.get("triangles"),
                                          "comparable": b.get("comparable")}
                                      for k, b in (v.get("baselines") or {}).items()}}
    mp = Path(sel["dir"]) / "metrics.json"
    if not mp.is_file():
        note_placeholder(f"budget {N}: {rel(mp)} missing")
        return {"source": "missing"}
    m = read_json(mp)
    if m.get("provenance", {}).get("mesh_sha256") not in (None, mesh_sha):
        note_placeholder(f"budget {N}: {rel(mp)} is stale for its mesh.json")
    t = m.get("triangles") or {}
    note_standin(f"budget {N}: numbers from {rel(mp)} (outputs/metrics.json not written yet)")
    out = {"source": rel(mp), **headline_from(m),
           "counts": {"A": t.get("mesh_json"), "B": t.get("blend_evaluated"), "C": t.get("glb_reimported"),
                      "D": t.get("glb_parsed"), "agree": t.get("all_agree")}}
    out["baselines"] = {}
    for kind in ("decimate", "decimate_refined"):
        bm = ROOT / "baselines" / kind / str(N) / "checkpoints" / "v0" / "metrics.json"
        if bm.is_file():
            b = read_json(bm)
            hb = headline_from(b)
            out["baselines"][kind] = {"metrics": hb, "triangles": (b.get("triangles") or {}).get("mesh_json"),
                                      "comparable": all((b.get("provenance") or {}).get(k) ==
                                                        (m.get("provenance") or {}).get(k)
                                                        for k in ("reference_sha256", "camera_sha256",
                                                                  "scene_sha256", "tooling_hash"))}
    return out


# ------------------------------------------------------------------ meshes
def load_min(path) -> dict:
    d = read_json(path)
    return {"vertices": np.asarray(d["vertices"], float), "triangles": np.asarray(d["triangles"], np.int64),
            "tri_region": list(d.get("tri_region") or ["torso"] * len(d["triangles"]))}


@functools.lru_cache(maxsize=1)
def ref_camera() -> Camera:
    cam = Camera.from_json()
    if cam.status != "locked":
        print("WARNING: contract/camera.json is not locked", file=sys.stderr)
    return cam


@functools.lru_cache(maxsize=8)
def _light_dir():
    mw = ref_camera().matrix_world
    right, up, back = mw[:3, 0], mw[:3, 1], mw[:3, 2]
    L = -0.35 * right + 0.6 * up + 0.75 * back
    return L / np.linalg.norm(L)


def overlay_mesh(base: Image.Image, V, T, regions, tri_alpha=None, ss_=2, face_alpha=0.42, line_px=1.5,
                 edge_rgb=(246, 250, 255), edge_alpha=0.92, halo=0.35, darken=0.0):
    """Draw a mesh over a CROP panel `base` (size w x h) through the locked reference camera.
    Faces: flat Lambert tinted by tri_region, semi-transparent. Edges: triangle-ID changes of a z-buffered ID
    image at ss_ x resolution (hidden lines removed), dilated to ~line_px output px, box-downsampled.
    tri_alpha: per-triangle visibility 0..1 (for parts fading in). Returns a new RGB image."""
    w, h = base.size
    V = np.asarray(V, float)
    T = np.asarray(T, np.int64)
    cam = ref_camera()
    u, v, depth = cam.project(V)
    s = w / CROP_W
    sx = (u - CROP[0]) * s * ss_
    sy = (v - CROP[1]) * s * ss_
    P = np.stack([sx, sy], 1)[T]
    key = (1.0 / depth)[T]
    Ws, Hs = w * ss_, h * ss_
    ids, _ = raster.id_buffer_screen(P, key, Ws, Hs)
    ta = np.ones(len(T)) if tri_alpha is None else np.asarray(tri_alpha, float)
    # per-face shade + tint
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    n = np.cross(B - A, C - A)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    view = cam.location[None] - (A + B + C) / 3
    n *= np.where((n * view).sum(1) < 0, -1.0, 1.0)[:, None]
    lam = np.clip(n @ _light_dir(), 0, 1)
    shade = 0.5 + 0.55 * lam
    tint = np.array([REGION_RGB.get(r, (200, 200, 200)) for r in regions], float) / 255.0
    fcol = np.clip(tint * shade[:, None], 0, 1)
    valid = ids >= 0
    idc = np.maximum(ids, 0)
    a_face = np.where(valid, face_alpha * ta[idc], 0.0)
    col = np.where(valid[..., None], fcol[idc], 0.0)
    # edges: id differs from right / down neighbour (either side drawn), weighted by the nearer tri's alpha
    e = np.zeros((Hs, Ws))
    ea = np.where(valid, ta[idc], 0.0)
    dx = ids[:, 1:] != ids[:, :-1]
    wx = np.maximum(ea[:, 1:], ea[:, :-1]) * dx
    e[:, 1:] = np.maximum(e[:, 1:], wx)
    e[:, :-1] = np.maximum(e[:, :-1], wx)
    dy = ids[1:, :] != ids[:-1, :]
    wy = np.maximum(ea[1:, :], ea[:-1, :]) * dy
    e[1:, :] = np.maximum(e[1:, :], wy)
    e[:-1, :] = np.maximum(e[:-1, :], wy)
    rad = max(0, int(round(line_px * ss_ / 2 - 1)))
    if rad > 0:
        e = np.asarray(Image.fromarray((e * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * rad + 1)),
                       float) / 255.0

    def down(x):
        if x.ndim == 2:
            return x.reshape(h, ss_, w, ss_).mean((1, 3))
        return x.reshape(h, ss_, w, ss_, x.shape[2]).mean((1, 3))

    af = down(a_face)
    cf = down(col * a_face[..., None])
    ae = down(e) * edge_alpha
    img = np.asarray(base, float) / 255.0
    if darken:
        img = img * (1 - darken)
    out = img * (1 - af[..., None]) + cf
    if halo:
        hal = np.asarray(Image.fromarray((ae * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.6)),
                         float) / 255.0
        out = out * (1 - halo * hal[..., None])
    out = out * (1 - ae[..., None]) + np.array(edge_rgb, float)[None, None] / 255.0 * ae[..., None]
    return Image.fromarray(np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8))


def silhouette_contour(mask: np.ndarray, w, h, width=2.0):
    """Alpha (h, w) of the outline of a 1536² binary mask, cropped to CROP and scaled to the panel."""
    m = Image.fromarray((mask[CROP[1]:CROP[3], CROP[0]:CROP[2]] * 255).astype(np.uint8))
    m = m.resize((w * 2, h * 2), Image.BILINEAR)
    a = np.asarray(m, float) / 255.0 >= 0.5
    edge = a ^ np.asarray(Image.fromarray(a.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3)), bool)
    em = Image.fromarray(edge.astype(np.uint8) * 255)
    k = max(1, int(round(width)))
    em = em.filter(ImageFilter.MaxFilter(2 * (k // 2) + 1 if k > 1 else 1)) if k > 1 else em
    return em.resize((w, h), Image.BOX)


@functools.lru_cache(maxsize=4)
def ref_mask() -> np.ndarray:
    return np.asarray(Image.open(DERIVED / "mask.png").convert("L")) >= 128


def working_iou(V, T, res=512) -> float:
    """Silhouette IoU at `res` px through the locked camera with catlib.raster (CPU) — the optimiser's kind of
    number, NOT an official metric."""
    cam = ref_camera().scaled(res)
    m = raster.silhouette(V, T, cam)
    R = _mask_at(res)
    u = (m | R).sum()
    return float((m & R).sum() / u) if u else 0.0


@functools.lru_cache(maxsize=4)
def _mask_at(res):
    M = ref_mask()
    k = M.shape[0] // res
    return M.reshape(res, k, res, k).mean((1, 3)) >= 0.5
