#!/usr/bin/env python
"""Build timelapse ON TOP OF THE CAT PHOTO through the locked camera (CONTRACT §13).  CPU only.

    python3 skills/image-to-lowpoly/scripts/timelapse/timelapse.py [--budgets 100,200,500,1000] [--seconds 12] [--jobs 6]
        [--at 0,3.5,11.9] [--no-all]

-> release_video/out/timelapse_<N>.mp4 (1920x1080, 24 fps) per budget and release_video/out/timelapse_all.mp4
(four-up). --at writes stills to release_video/out/at/timelapse_<N>_t<sec>.png instead of encoding.

Stages come from variants/<N>/timelapse/stages.json (worker-written, takes precedence) or, failing that, from
release_video/work/<N>/stages.json (release_video/make_stages.py; run automatically when missing or stale).
Each frame draws the mesh on screen as semi-transparent flat faces tinted by tri_region plus its triangle edges
with hidden lines removed (common.overlay_mesh: catlib.camera projection + catlib.raster z-buffered ID image).
`construct` stages fade parts in in build order; `refine` stages play real optimiser snapshots with linear
interpolation between CONSECUTIVE snapshots only; `edit` stages cross-fade; `final` holds the checkpoint mesh.
The silhouette readout is the optimiser's kind of number (catlib.raster at 512 px, labelled as a working
estimate) until the final hold, where the official numbers of that checkpoint (Blender render) replace it.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import subprocess
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
from common import (AMBER, CYAN, DIM, GREY, PANEL, SOFT, WHITE, appear, draw_text, outline, paste_frame,  # noqa: E402
                    pill, rect, ss, tag, text_w)

FPS = C.FPS
PANEL_DIR = C.WORK / "panels"
EST_RES = 512

# film-wide main-frame geometry (also used by make_film's build beat)
FX, FY, FH = 180, 150, 820
FW = round(FH * C.CROP_W / C.CROP_H)          # 984
PX, PW = FX + FW + 60, C.W - 180 - (FX + FW + 60)


# ------------------------------------------------------------------ stages
def stages_path(N) -> tuple[Path, str]:
    w = C.ROOT / "variants" / str(N) / "timelapse" / "stages.json"
    if w.is_file():
        return w, "worker"
    return C.WORK / str(N) / "stages.json", "fallback"


def _final_of(p: Path):
    d = json.loads(p.read_text())
    fin = Path(d["stages"][-1]["mesh"])
    return fin if fin.is_absolute() else p.parent / fin


def ensure_stages(N) -> tuple[dict, Path, str]:
    p, kind = stages_path(N)
    sel = C.select_checkpoint(N)
    if kind == "worker":
        fin = _final_of(p)
        if not fin.is_file() or C.sha256(fin) != C.sha256(sel["mesh"]):
            C.note_standin(f"budget {N}: worker stages.json ends on {C.rel(fin)}, not the selected checkpoint "
                           f"{sel['tag']} -> fallback stages used")
            p, kind = C.WORK / str(N) / "stages.json", "fallback"
    if kind == "fallback":
        stale = True
        if p.is_file():
            d = json.loads(p.read_text())
            fin = Path(d["stages"][-1]["mesh"])
            fin = fin if fin.is_absolute() else p.parent / fin
            stale = not fin.is_file() or C.sha256(fin) != C.sha256(sel["mesh"])
        if stale:
            print(f"budget {N}: building fallback stages for {sel['tag']} (make_stages.py)", flush=True)
            import make_stages
            make_stages.main(["--budget", str(N)])
    d = json.loads(p.read_text())
    return d, p.parent, kind


def _res(base: Path, s) -> Path:
    s = Path(s)
    return s if s.is_absolute() else (base / s).resolve()


def _tri_key(T):
    return [tuple(t) for t in np.asarray(T, np.int64).tolist()]


class Timeline:
    """Maps time (s) to what is on screen for one budget."""

    def __init__(self, N: int, duration: float, fracs=(0.28, 0.52, 0.20)):
        self.N, self.D = N, float(duration)
        self.doc, self.base, self.source = ensure_stages(N)
        self.sel = C.select_checkpoint(N)
        st = self.doc["stages"]
        self.final_mesh = _res(self.base, st[-1]["mesh"])
        cons = [s for s in st if s["kind"] == "construct"]
        refs = [s for s in st if s["kind"] == "refine"]
        edits = [s for s in st if s["kind"] == "edit"]
        t_edit = 0.5 * len(edits)
        f_c, f_r, f_f = fracs
        if not refs:
            f_c, f_r = f_c + f_r * 0.5, 0.0
        avail = self.D - t_edit
        t_c, t_r = avail * f_c, avail * f_r
        self.segs = []       # (t0, t1, kind, payload)
        t = 0.0
        prev_keys = set()
        for s in cons:
            d = C.load_min(_res(self.base, s["mesh"]))
            keys = _tri_key(d["triangles"])
            new = np.array([k not in prev_keys for k in keys])
            dt = t_c / max(len(cons), 1)
            self.segs.append((t, t + dt, "construct", {"label": s["label"], "mesh": d, "new": new,
                                                        "prev_count": len(prev_keys)}))
            prev_keys = set(keys)
            t += dt
        # refine stages -> groups of keyframes: optimiser snapshots ("snapshots" dir) or, for workers that
        # recorded real accepted intermediate meshes instead, runs of consecutive refine stages with "mesh".
        groups, cur = [], None
        for s in st:
            if s["kind"] == "refine" and "snapshots" in s:
                cur = None
                files = sorted(_res(self.base, s["snapshots"]).glob("snap_*.json"))
                groups.append({"stage": s, "files": files, "labels": None, "mode": "snapshots"})
            elif s["kind"] == "refine" and "mesh" in s:
                if cur is None:
                    cur = {"stage": s, "files": [], "labels": [], "mode": "meshes"}
                    groups.append(cur)
                cur["files"].append(_res(self.base, s["mesh"]))
                cur["labels"].append(s["label"])
            else:
                cur = None
                groups.append({"stage": s})
        wsum = sum(max(1, len(g["files"])) for g in groups if "files" in g) or 1
        for g in groups:
            s = g["stage"]
            if "files" in g:
                files = g["files"]
                if s.get("base_mesh"):
                    base_d = C.load_min(_res(self.base, s["base_mesh"]))
                elif g["mode"] == "meshes":
                    base_d = C.load_min(files[0])
                else:
                    base_d = None          # filled from the previous segment's end state below
                dt = t_r * max(1, len(files)) / wsum
                lab = s["label"] if g["mode"] == "snapshots" else "real intermediate meshes"
                self.segs.append((t, t + dt, "refine", {"label": lab, "files": files, "base": base_d,
                                                        "labels": g["labels"], "mode": g["mode"],
                                                        "reproduced": s.get("reproduced", True)}))
                t += dt
            elif s["kind"] == "edit":
                after = C.load_min(_res(self.base, s["mesh"]))
                before = C.load_min(_res(self.base, s["before"])) if s.get("before") else None
                self.segs.append((t, t + 0.5, "edit", {"label": s["label"], "before": before, "after": after}))
                t += 0.5
        self.t_final = t
        self.segs.append((t, self.D + 1e9, "final", {"label": "final", "mesh": C.load_min(self.final_mesh)}))
        # fill 'before' of edits / refine bases from the preceding segment's end state
        for i, sg in enumerate(self.segs):
            if sg[2] == "refine" and sg[3]["base"] is None:
                sg[3]["base"] = self._end_state(i - 1)
            if sg[2] == "edit" and sg[3]["before"] is None and i > 0:
                sg[3]["before"] = self._end_state(i - 1)
        self.n_final = len(self.segs[-1][3]["mesh"]["triangles"])
        self.official = C.official(self.sel)
        self._est_cache = self._load_est_cache()

    # ---------------- snapshot access
    @functools.lru_cache(maxsize=512)
    def _snap(self, path: Path):
        d = json.loads(Path(path).read_text())
        return np.asarray(d["vertices"], float), (np.asarray(d["triangles"], np.int64) if "triangles" in d else None), \
            (int(d["proposal"]) if "proposal" in d else None)

    @functools.lru_cache(maxsize=512)
    def _regions(self, path: Path):
        d = json.loads(Path(path).read_text())
        return d.get("tri_region")

    def _end_state(self, i):
        t0, t1, kind, p = self.segs[i]
        if kind == "construct":
            return p["mesh"]
        if kind == "refine":
            V, T, _ = self._snap(p["files"][-1])
            return {"vertices": V, "triangles": T if T is not None else p["base"]["triangles"],
                    "tri_region": self._regions(p["files"][-1]) or p["base"]["tri_region"]}
        if kind == "edit":
            return p["after"]
        return p["mesh"]

    # ---------------- working estimate (catlib.raster, 512 px), cached per real mesh state
    def _est_file(self):
        return C.WORK / str(self.N) / "working_iou_cache.json"

    def _load_est_cache(self):
        f = self._est_file()
        return json.loads(f.read_text()) if f.is_file() else {}

    def save_est_cache(self):
        f = self._est_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(self._est_cache))

    def est(self, V, T) -> float:
        k = hashlib.sha1(np.asarray(V, np.float64).tobytes() + np.asarray(T, np.int64).tobytes()).hexdigest()
        if k not in self._est_cache:
            self._est_cache[k] = C.working_iou(V, T, EST_RES)
        return self._est_cache[k]

    def _retime(self, p, u):
        """Keyframe position (float index) for segment progress u. Time per interval = 25% uniform, 75%
        proportional to the largest vertex motion in that interval, so big early optimiser moves get more frames.
        Only the timing changes; geometry is still linear between consecutive real keyframes."""
        files = p["files"]
        n = len(files)
        if n < 2:
            return 0.0
        if "cum" not in p:
            d = []
            for a, b in zip(files[:-1], files[1:]):
                Va, Vb = self._snap(a)[0], self._snap(b)[0]
                d.append(float(np.abs(Va - Vb).max()) if Va.shape == Vb.shape else 0.0)
            d = np.array(d)
            w = 0.75 * (d / d.sum() if d.sum() > 0 else np.full(len(d), 1 / len(d))) + 0.25 / len(d)
            p["cum"] = np.concatenate([[0.0], np.cumsum(w)])
            p["cum"] /= p["cum"][-1]
        return float(np.interp(u, p["cum"], np.arange(n)))

    # ---------------- state at time t
    def state(self, t: float) -> dict:
        t = min(max(t, 0.0), self.D)
        for i, (t0, t1, kind, p) in enumerate(self.segs):
            if t < t1 or kind == "final":
                break
        u = (t - t0) / max(t1 - t0, 1e-9)
        phase = {"construct": "Construct", "refine": "Refine", "edit": "Edit", "final": "Final"}[kind]
        if kind == "construct":
            d = p["mesh"]
            fade = ss(u / 0.6)
            alpha = np.where(p["new"], fade, 1.0)
            n_new = int(p["new"].sum())
            count = p["prev_count"] + int(round(n_new * fade))
            return {"phase": phase, "label": p["label"], "V": d["vertices"], "T": d["triangles"],
                    "R": d["tri_region"], "alpha": alpha, "count": count,
                    "est": self.est(d["vertices"], d["triangles"]), "seg": i}
        if kind == "refine":
            files = p["files"]
            n = len(files)
            pos = self._retime(p, u)
            a = int(min(math.floor(pos), n - 2)) if n > 1 else 0
            f = pos - a if n > 1 else 0.0
            Va, Ta, pa = self._snap(files[a])
            Vb, Tb, pb = self._snap(files[min(a + 1, n - 1)])
            Ta = Ta if Ta is not None else p["base"]["triangles"]
            Tb = Tb if Tb is not None else p["base"]["triangles"]
            fb = files[min(a + 1, n - 1)]
            Ra = self._regions(files[a]) or p["base"]["tri_region"]
            Rb = self._regions(fb) or p["base"]["tri_region"]
            if Va.shape == Vb.shape:
                V = (1 - f) * Va + f * Vb                   # linear, consecutive real keyframes only
                T, R = (Ta, Ra) if f < 0.5 else (Tb, Rb)    # flips: cut at the midpoint, never invented
            else:                                           # different connectivity: cut, no interpolation
                V, T, R = (Va, Ta, Ra) if f < 0.5 else (Vb, Tb, Rb)
            ea, eb = self.est(Va, Ta), self.est(Vb, Tb)
            last_p = self._snap(files[-1])[2]
            if pa is not None and pb is not None and last_p:
                lab = f"proposal {int(round((1 - f) * pa + f * pb)):,} / {last_p:,}"
            else:
                lab = (p["labels"] or [p["label"]] * n)[a if f < 0.5 else min(a + 1, n - 1)]
            return {"phase": phase, "label": lab, "V": V, "T": T, "R": R, "alpha": None,
                    "count": len(T), "est": (1 - f) * ea + f * eb, "seg": i,
                    "reproduced": p.get("reproduced", True)}
        if kind == "edit":
            b, a_ = p["before"], p["after"]
            return {"phase": phase, "label": p["label"], "xfade": (b, a_, ss(u)), "V": a_["vertices"],
                    "T": a_["triangles"], "R": a_["tri_region"], "alpha": None, "count": len(a_["triangles"]),
                    "est": self.est(a_["vertices"], a_["triangles"]), "seg": i}
        d = p["mesh"]
        return {"phase": phase, "label": f"checkpoint {self.sel['tag']}", "V": d["vertices"], "T": d["triangles"],
                "R": d["tri_region"], "alpha": None, "count": len(d["triangles"]),
                "est": self.est(d["vertices"], d["triangles"]), "seg": i, "final_u": t - t0}

    def key(self) -> str:
        """Content key of everything a panel depends on."""
        h = hashlib.sha256()
        h.update(json.dumps(self.doc, sort_keys=True).encode())
        h.update(C.sha256(self.final_mesh).encode())
        h.update(f"{self.D:.4f}|v5-retime".encode())
        for sg in self.segs:
            if sg[2] == "refine":
                h.update(str(len(sg[3]["files"])).encode())
                h.update(sg[3]["files"][-1].name.encode())
        return h.hexdigest()[:16]


# ------------------------------------------------------------------ panel rendering (cached, parallel)
def render_panel(tl: Timeline, t: float, w: int, h: int, line_px=1.5) -> Image.Image:
    s = tl.state(t)
    base = C.photo(w, h)
    if "xfade" in s:
        b, a_, f = s["xfade"]
        ib = C.overlay_mesh(base, b["vertices"], b["triangles"], b["tri_region"], line_px=line_px)
        ia = C.overlay_mesh(base, a_["vertices"], a_["triangles"], a_["tri_region"], line_px=line_px)
        return Image.blend(ib, ia, f)
    return C.overlay_mesh(base, s["V"], s["T"], s["R"], tri_alpha=s["alpha"], line_px=line_px)


_TL = {}


def _worker(args):
    N, D, w, h, k, path, line_px = args
    key = (N, D)
    if key not in _TL:
        _TL[key] = Timeline(N, D)
    render_panel(_TL[key], k / FPS, w, h, line_px).save(path, compress_level=1)
    return path


def panel_dir(tl: Timeline, w, h) -> Path:
    return PANEL_DIR / f"{tl.N}_{tl.key()}_{w}x{h}"


def ensure_panels(tl: Timeline, w: int, h: int, nframes: int, jobs: int = 6, line_px=1.5) -> Path:
    d = panel_dir(tl, w, h)
    d.mkdir(parents=True, exist_ok=True)
    todo = [(tl.N, tl.D, w, h, k, str(d / f"f_{k:04d}.png"), line_px) for k in range(nframes)
            if not (d / f"f_{k:04d}.png").is_file()]
    if todo:
        t0 = time.time()
        print(f"budget {tl.N}: rendering {len(todo)} panels {w}x{h} with {jobs} processes", flush=True)
        if jobs > 1:
            with Pool(jobs) as pool:
                for _ in pool.imap_unordered(_worker, todo, chunksize=4):
                    pass
        else:
            for a in todo:
                _worker(a)
        print(f"budget {tl.N}: panels done in {time.time() - t0:.1f}s", flush=True)
    for k in range(nframes):    # warm the working-estimate cache in this process (cheap) and persist it
        tl.state(k / FPS)
    tl.save_est_cache()
    return d


def panel(tl: Timeline, k: int, w: int, h: int) -> Image.Image:
    p = panel_dir(tl, w, h) / f"f_{k:04d}.png"
    if p.is_file():
        return Image.open(p).convert("RGB")
    return render_panel(tl, k / FPS, w, h)


# ------------------------------------------------------------------ HUD (film style)
PHASES = ["Construct", "Refine", "Final"]


def fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def official_lines(off: dict):
    if off.get("source") in (None, "missing"):
        return [("official metrics", "missing", AMBER)]
    return [("IoU", fmt(off.get("iou"), 4), WHITE),
            ("boundary mean", f"{off.get('boundary_mean_px', 0):.2f} px", WHITE),
            ("landmarks (12 sharp)", f"{off.get('landmark_sharp_mean_px', 0):.2f} px", WHITE)]


def draw_build_hud(c, tl: Timeline, t: float, x=None, y=None, w=None, a=1.0):
    """Right-hand panel: live triangle count, phase chips, stage label, readout, legend."""
    x = PX if x is None else x
    y = FY if y is None else y
    w = PW if w is None else w
    s = tl.state(t)
    num = f"{s['count']}"
    over = s["count"] > tl.N
    draw_text(c, num, x, y - 14, 120, "B", AMBER if over else CYAN, a)
    nw = text_w(num, 120, "B")
    draw_text(c, f"/ {tl.N}", x + nw + 14, y + 46, 44, "M", GREY, a)
    draw_text(c, "triangles on screen", x, y + 128, 28, "M", WHITE, a)
    if over:
        draw_text(c, "· a real intermediate, over budget until a later edit", x + text_w("triangles on screen", 28, "M") + 12,
                  y + 134, 20, "R", AMBER, a)
    # phase chips
    yy = y + 186
    xx = x
    cur = "Final" if s["phase"] == "Final" else ("Refine" if s["phase"] in ("Refine", "Edit") else "Construct")
    for ph in PHASES:
        on = ph == cur
        wch = int(text_w(ph, 22, "M")) + 36
        rect(c, xx, yy, wch, 42, 21, CYAN if on else PANEL, (1.0 if on else 0.85) * a)
        if not on:
            outline(c, xx, yy, wch, 42, 21, WHITE, 0.10 * a, 1)
        draw_text(c, ph, xx + 18, yy + 7, 22, "M", (6, 14, 24) if on else GREY, a)
        xx += wch + 10
    lab = s["label"] if s["phase"] != "Edit" else s["label"]
    draw_text(c, lab, x, yy + 62, 25, "R", SOFT, a)
    if s["phase"] == "Refine" and not s.get("reproduced", True):
        draw_text(c, "re-run did not reproduce: shown for shape only", x, yy + 96, 20, "R", AMBER, a)
    # readout
    ry = yy + 140
    final_on = s["phase"] == "Final" and s.get("final_u", 0) >= 0.25
    fa = appear(s.get("final_u", 0), 0.375, 0.25) if s["phase"] == "Final" else 0.0
    if fa < 1:
        aa = a * (1 - fa)
        draw_text(c, "Silhouette IoU", x, ry, 24, "M", WHITE, aa)
        draw_text(c, fmt(s["est"]), x + w, ry - 4, 34, "M", CYAN, aa, "r")
        draw_text(c, "optimiser's working estimate", x, ry + 40, 20, "R", GREY, aa)
        draw_text(c, f"CPU raster at {EST_RES} px · not an official number", x, ry + 68, 20, "R", DIM, aa)
    if fa > 0:
        aa = a * fa
        draw_text(c, "Official · Blender render, locked camera", x, ry, 22, "M", WHITE, aa)
        for j, (k_, v_, col) in enumerate(official_lines(tl.official)):
            draw_text(c, k_, x, ry + 38 + j * 36, 23, "R", GREY, aa)
            draw_text(c, v_, x + w, ry + 36 + j * 36, 26, "M", CYAN if j == 0 else col, aa, "r")
    # legend
    ly = ry + 172
    regs = [r for r in C.REGION_ORDER if r in set(tl.segs[-1][3]["mesh"]["tri_region"])]
    for j, r in enumerate(regs):
        cx = x + (j % 3) * (w // 3)
        cy = ly + (j // 3) * 32
        c.paste(C.REGION_RGB[r], (int(cx), int(cy + 8)), C.scaled(C.dot_mask(12), a))
        draw_text(c, r.replace("_", " "), cx + 20, cy, 20, "R", GREY, a)
    if tl.sel["status"] != "final":
        pill(c, f"stand-in · checkpoint {tl.sel['tag']} · final pending", x, FY + FH - 50, a, anchor="l",
             size=19, dot=AMBER, color=AMBER, bga=0.9)


SB_W, SB_H, SB_X, SB_Y = 1128, 940, 84, 70


def compose_single(tl: Timeline, k: int) -> Image.Image:
    """Simple standalone timelapse: the build on the photo, the budget and a running triangle count only."""
    c = C.BG().copy()
    s = tl.state(k / FPS)
    paste_frame(c, panel(tl, k, SB_W, SB_H), SB_X, SB_Y)
    x, y = SB_X + SB_W + 70, SB_Y + 250
    draw_text(c, f"{s['count']}", x, y, 170, "B", AMBER if s["count"] > tl.N else CYAN)
    draw_text(c, "triangles", x, y + 190, 34, "M", WHITE)
    draw_text(c, f"budget {tl.N}", x, y + 242, 28, "R", GREY)
    if tl.sel["status"] != "final":
        tag(c, "work in progress", C.W - 28, C.H - 52, 0.8, anchor="r", color=GREY, size=17)
    return c


ALL_W, ALL_H = 612, 510


def compose_all(tls, k) -> Image.Image:
    c = C.BG().copy()
    gap = 24
    x0, y0 = (C.W - 2 * ALL_W - gap) // 2, (C.H - 2 * ALL_H - gap) // 2
    for i, tl in enumerate(tls):
        x = x0 + (i % 2) * (ALL_W + gap)
        y = y0 + (i // 2) * (ALL_H + gap)
        paste_frame(c, panel(tl, k, ALL_W, ALL_H), x, y, r=12)
        s = tl.state(k / FPS)
        tag(c, f"{tl.N}", x + 16, y + 16, color=CYAN, size=26)
        draw_text(c, f"{s['count']}", x + ALL_W - 18, y + 10, 40, "B", AMBER if s["count"] > tl.N else WHITE, 1, "r")
    if any(tl.sel["status"] != "final" for tl in tls):
        tag(c, "work in progress", C.W - 28, C.H - 52, 0.8, anchor="r", color=GREY, size=17)
    return c


# ------------------------------------------------------------------ encode
def encode(frames_iter, n, out: Path, crf=16):
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [(shutil.which("ffmpeg") or "ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
           f"{C.W}x{C.H}", "-r", str(FPS), "-i", "-", "-vf",
           "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-c:v", "libx264", "-preset", "medium",
           "-crf", str(crf), "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-movflags", "+faststart", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(n):
        p.stdin.write(frames_iter(k).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit("ffmpeg failed")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budgets", default="100,200,500,1000")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--at", default=None)
    ap.add_argument("--no-all", action="store_true")
    a = ap.parse_args(argv)
    budgets = [int(x) for x in a.budgets.split(",")]
    n = int(round(a.seconds * FPS))
    t0 = time.time()
    tls = [Timeline(N, a.seconds) for N in budgets]
    for tl in tls:
        print(f"budget {tl.N}: stages from {tl.source} ({C.rel(tl.base)}), checkpoint {tl.sel['tag']} "
              f"[{tl.sel['status']}], final triangles {tl.n_final}")
    if a.at:
        d = C.OUT / "at"
        d.mkdir(parents=True, exist_ok=True)
        for tl in tls:
            for s in a.at.split(","):
                k = int(round(float(s) * FPS))
                compose_single(tl, k).save(d / f"timelapse_{tl.N}_t{float(s):05.2f}.png")
        if not a.no_all and len(tls) == 4:
            for s in a.at.split(","):
                compose_all(tls, int(round(float(s) * FPS))).save(d / f"timelapse_all_t{float(s):05.2f}.png")
        print("stills in", d)
        return 0
    for tl in tls:
        ensure_panels(tl, SB_W, SB_H, n, a.jobs)
        t1 = time.time()
        encode(lambda k: compose_single(tl, k), n, C.OUT / f"timelapse_{tl.N}.mp4")
        last = tl.state(a.seconds)
        print(f"wrote {C.rel(C.OUT / f'timelapse_{tl.N}.mp4')} ({n} frames, encode {time.time() - t1:.1f}s); "
              f"last frame shows {last['count']} triangles (final mesh has {tl.n_final})")
    if not a.no_all and len(tls) == 4:
        for tl in tls:
            ensure_panels(tl, ALL_W, ALL_H, n, a.jobs)
        encode(lambda k: compose_all(tls, k), n, C.OUT / "timelapse_all.mp4")
        print(f"wrote {C.rel(C.OUT / 'timelapse_all.mp4')}")
    if C.STANDINS:
        print("STAND-INS:\n  " + "\n  ".join(C.STANDINS))
    print(f"total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
