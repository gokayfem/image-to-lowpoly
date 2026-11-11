"""Joint fit of THE shared camera and the parametric proxy cat (CONTRACT §2, §3, §11).

CPU only (numpy + scipy + cma, catlib.raster).  Fixed seeds; reproducible.

    python3 scripts/fit_camera_proxy.py [--stages 1,2,3,4,5,6] [--workers 8]
        [--resume] [--quick]

Writes (all under proxy/):
    fit_params.json        every fitted parameter, camera, model landmarks, priors, trace summary
    skeleton.json          catskel-1 generalised-cylinder cat (catlib.skeleton reads it)
    fit_trace.json         objective trace per stage (best-so-far every CMA iteration)
    fit_stages/stageK_overlay.png   photo + proxy outline + landmark pairs after every stage
    fit_stages/stageK_params.json   checkpoint after every stage (used by --resume)

Model: torso (8 elliptical/superelliptical stations), neck, head (skull/cheek/muzzle rings
with free yaw/pitch/roll), two thin ear pinnae, four legs (top joint -> knee/elbow ->
hock/wrist -> paw -> toe) and the tail centre curve; all generalised cylinders.
Camera: look_at (3) + azimuth / elevation / distance about look_at; 85 mm, roll 0.
Gauge: the torso top at the withers station is fixed at 0.25 m.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the folder that holds catlib/
from catlib import paths as _paths  # noqa: E402  (resolves the case: --case / LOWPOLY_CASE / cwd)
ROOT = _paths.ROOT

from catlib import camera as camlib  # noqa: E402
from catlib import raster, skeleton as sk  # noqa: E402
from catlib.metrics import boundary, load_mask, load_ref_landmarks, load_regions  # noqa: E402

PROXY = ROOT / "proxy"
STAGES_DIR = PROXY / "fit_stages"
WITHERS_Z = 0.25
TORSO_S = (0.0, 0.07, 0.20, 0.38, 0.56, 0.72, 0.86, 1.0)
WITHERS_IDX = 6
LEGS = ("FR", "FL", "HR", "HL")

# ---------------------------------------------------------------- parameter specification
# name: (init, scale, lo, hi)
SPEC: dict[str, tuple] = {}


def _p(name, init, scale, lo, hi):
    SPEC[name] = (float(init), float(scale), float(lo), float(hi))


# camera
_p("cam_az", -50.0, 5.0, -85.0, -15.0)
_p("cam_el", 10.0, 3.0, -2.0, 30.0)
_p("cam_dist", 1.7, 0.1, 0.7, 3.5)
_p("cam_lx", 0.02, 0.03, -0.3, 0.3)
_p("cam_ly", 0.0, 0.03, -0.3, 0.3)
_p("cam_lz", 0.14, 0.03, -0.1, 0.4)
# torso
_p("torso_xr", -0.24, 0.02, -0.34, -0.15)
_p("torso_xf", 0.21, 0.02, 0.12, 0.32)
_p("torso_p", 2.2, 0.1, 2.0, 2.4)
_p("torso_cap_r", 0.045, 0.008, 0.03, 0.09)
_p("torso_cap_f", 0.04, 0.008, 0.035, 0.08)
_TOP = (0.22, 0.245, 0.255, 0.255, 0.255, 0.255, WITHERS_Z, 0.235)
_BOT = (0.135, 0.105, 0.088, 0.082, 0.085, 0.092, 0.105, 0.135)
_HW = (0.034, 0.042, 0.05, 0.05, 0.05, 0.051, 0.046, 0.035)
for _i in range(len(TORSO_S)):
    if _i != WITHERS_IDX:
        _p(f"torso_top{_i}", _TOP[_i], 0.01, 0.12, 0.32)
    _p(f"torso_bot{_i}", _BOT[_i], 0.01, 0.03, 0.22)
    _p(f"torso_hw{_i}", _HW[_i], 0.006, 0.032 if _i == 0 else 0.015, 0.11)
# neck (torso anchor + radii)
_p("neck_x0", 0.15, 0.02, 0.0, 0.3)
_p("neck_z0", 0.19, 0.02, 0.1, 0.3)
_p("neck_r0", 0.045, 0.005, 0.03, 0.055)
_p("neck_r1", 0.035, 0.005, 0.015, 0.06)
_p("neck_lift", 0.01, 0.006, -0.02, 0.05)      # nape: mid ring raised above the chord
_p("neck_rmid", 1.0, 0.08, 0.7, 1.5)           # mid ring radius factor
# head
_p("head_x", 0.25, 0.02, 0.1, 0.4)
_p("head_y", -0.01, 0.01, -0.08, 0.08)
_p("head_z", 0.30, 0.02, 0.2, 0.42)
_p("head_yaw", -35.0, 6.0, -80.0, 10.0)
_p("head_pitch", 5.0, 5.0, -30.0, 40.0)
_p("head_roll", 0.0, 4.0, -25.0, 25.0)
_p("head_back", 0.045, 0.005, 0.025, 0.07)
_p("head_front", 0.05, 0.005, 0.03, 0.08)
_p("head_W", 0.047, 0.003, 0.035, 0.06)
_p("head_H", 0.042, 0.003, 0.03, 0.06)
_p("head_cheek", 0.9, 0.05, 0.6, 1.1)
_p("muzzle_W", 0.022, 0.003, 0.012, 0.035)
_p("muzzle_H", 0.018, 0.003, 0.010, 0.03)
_p("muzzle_drop", -0.014, 0.004, -0.035, 0.005)
_p("head_p", 2.1, 0.08, 2.0, 2.3)
_p("eye_b", 0.021, 0.002, 0.012, 0.03)
_p("eye_c", 0.010, 0.003, -0.005, 0.025)
# ears (head-local coords: a fwd, b left, c up)
_p("ear_a", -0.005, 0.005, -0.03, 0.02)
_p("ear_b", 0.026, 0.004, 0.012, 0.042)
_p("ear_c", 0.030, 0.004, 0.015, 0.045)
for _s, _sg in (("R", -1), ("L", 1)):
    _p(f"ear{_s}_ta", -0.002, 0.006, -0.03, 0.03)
    _p(f"ear{_s}_tb", 0.040 * _sg, 0.006, -0.07, 0.07)
    _p(f"ear{_s}_tc", 0.080, 0.006, 0.05, 0.11)
_p("ear_w", 0.019, 0.003, 0.01, 0.03)
_p("ear_t", 0.004, 0.0005, 0.003, 0.0045)
# tail: base on rump + 4 control points
_p("tail_x0", -0.255, 0.01, -0.36, -0.15)
_p("tail_z0", 0.195, 0.01, 0.12, 0.26)
_TAIL = ((-0.32, 0.0, 0.215), (-0.39, 0.0, 0.26), (-0.43, 0.0, 0.33), (-0.45, 0.0, 0.40))
for _k, (_x, _y, _z) in enumerate(_TAIL, 1):
    _p(f"tail_x{_k}", _x, 0.015, -0.7, -0.1)
    _p(f"tail_y{_k}", _y, 0.01, -0.12, 0.12)
    _p(f"tail_z{_k}", _z, 0.015, 0.1, 0.6)
_p("tail_r0", 0.022, 0.003, 0.01, 0.04)
_p("tail_r1", 0.014, 0.002, 0.007, 0.025)
_p("tail_r2", 0.012, 0.002, 0.006, 0.025)
# legs
_p("front_yt", 0.032, 0.005, 0.015, 0.06)
_p("hind_yt", 0.040, 0.005, 0.03, 0.07)
_LEG0 = {  # xt, zt, x1, z1, x2, z2, xp, yp
    "FR": (0.115, 0.165, 0.095, 0.115, 0.115, 0.035, 0.125, -0.045),
    "FL": (0.115, 0.165, 0.095, 0.115, 0.12, 0.035, 0.13, 0.045),
    "HR": (-0.17, 0.165, -0.12, 0.105, -0.215, 0.06, -0.205, -0.045),
    "HL": (-0.16, 0.165, -0.10, 0.105, -0.14, 0.055, -0.115, 0.045),
}
for _L, _v in _LEG0.items():
    for _nm, _x in zip(("xt", "zt", "x1", "z1", "x2", "z2", "xp", "yp"), _v):
        lo, hi, sc = (-0.6, 0.5, 0.012)
        if _nm.startswith("z"):
            lo, hi = (0.01, 0.26)
        if _nm == "yp":
            lo, hi, sc = (-0.12, 0.12, 0.006)
        _p(f"{_L}_{_nm}", _x, sc, lo, hi)
# leg radii (shared left/right): top lateral/sagittal, j1 lat/sag, j2, paw width/height
for _g, _vals in (("front", (0.028, 0.035, 0.019, 0.022, 0.014, 0.017, 0.012)),
                  ("hind", (0.036, 0.055, 0.024, 0.030, 0.012, 0.016, 0.012))):
    for _nm, _x in zip(("rt_l", "rt_s", "r1_l", "r1_s", "r2", "rpw", "rph"), _vals):
        _lo = 0.011 if _nm in ("rph", "r2") else (0.016 if _nm == "rpw" else 0.006)
        if _g == "hind" and _nm == "rt_l":
            _lo = 0.026                      # haunch: thigh at least 5 cm thick laterally
        _p(f"{_g}_{_nm}", max(_x, _lo), max(0.002, 0.12 * _x), _lo, 0.09)
_p("hind_rh", 0.013, 0.002, 0.007, 0.03)    # hock-to-paw (metatarsus) radius

NAMES = list(SPEC)
INIT = np.array([SPEC[n][0] for n in NAMES])
SCALE = np.array([SPEC[n][1] for n in NAMES])
LO = np.array([SPEC[n][2] for n in NAMES])
HI = np.array([SPEC[n][3] for n in NAMES])


def as_dict(vec) -> dict:
    return {n: float(x) for n, x in zip(NAMES, vec)}


def as_vec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, SPEC[n][0])) for n in NAMES])


# ---------------------------------------------------------------- geometry helpers
def _rot(yaw, pitch, roll):
    y, p, r = (math.radians(a) for a in (yaw, pitch, roll))
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    return Rz @ Ry @ Rx


def _chain_rings(pts, ru, rv, p=2.0, ref=(0.0, 1.0, 0.0)):
    """Rings perpendicular to a polyline; u = lateral reference projected, v = t x u."""
    pts = np.asarray(pts, float)
    K = len(pts)
    tng = np.empty_like(pts)
    tng[0] = pts[1] - pts[0]
    tng[-1] = pts[-1] - pts[-2]
    if K > 2:
        d0 = pts[1:-1] - pts[:-2]
        d1 = pts[2:] - pts[1:-1]
        d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
        d1 /= np.linalg.norm(d1, axis=1, keepdims=True)
        tng[1:-1] = d0 + d1
    tng /= np.linalg.norm(tng, axis=1, keepdims=True)
    ref = np.broadcast_to(np.asarray(ref, float), pts.shape)
    u = ref - (ref * tng).sum(1, keepdims=True) * tng
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(tng, u)
    return {"c": pts, "u": u, "v": v, "ru": np.asarray(ru, float) * np.ones(K),
            "rv": np.asarray(rv, float) * np.ones(K), "p": np.full(K, float(p))}


def camera_from(P: dict, size=None) -> camlib.Camera:
    a, e, d = math.radians(P["cam_az"]), math.radians(P["cam_el"]), P["cam_dist"]
    look = np.array([P["cam_lx"], P["cam_ly"], P["cam_lz"]])
    loc = look + d * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    W = 1536
    f_px = 85.0 / 36.0 * W
    cam = camlib.Camera(W, W, f_px, camlib.look_at_matrix(loc, look), name="fit",
                        status="fit", meta={"json": {"location": loc.tolist(),
                                                     "look_at": look.tolist()}})
    return cam.scaled(size) if size else cam


def torso_profile(P):
    xr, xf = P["torso_xr"], P["torso_xf"]
    xs = np.array([xr + s * (xf - xr) for s in TORSO_S])
    top = np.array([WITHERS_Z if i == WITHERS_IDX else P[f"torso_top{i}"] for i in range(len(TORSO_S))])
    bot = np.array([P[f"torso_bot{i}"] for i in range(len(TORSO_S))])
    hw = np.array([P[f"torso_hw{i}"] for i in range(len(TORSO_S))])
    return xs, top, bot, hw


def head_rings_local(P):
    hb, hf, W, H = P["head_back"], P["head_front"], P["head_W"], P["head_H"]
    mW, mH, md, ch = P["muzzle_W"], P["muzzle_H"], P["muzzle_drop"], P["head_cheek"]
    # back of skull, skull, widest (ears/eyes), cheeks, cheek->whisker-pad taper, muzzle, nose
    a = np.array([-hb, -0.55 * hb, 0.0, 0.35 * hf, 0.6 * hf, 0.82 * hf, hf])
    c = np.array([0.004, 0.003, 0.0, -0.004, 0.5 * md - 0.002, md, md + 0.002])
    ry = np.array([0.55 * W, 0.9 * W, W, ch * W, 0.55 * (ch * W + mW), mW, 0.7 * mW])
    rz = np.array([0.6 * H, 0.95 * H, H, 0.85 * H, 0.55 * (0.85 * H + mH), mH, 0.7 * mH])
    return a, c, ry, rz


def head_surface_exit(P, b, c0, a0=0.0):
    """Head-local: march from (a0, b, c0) along +a, return a where the point leaves the head
    cross-section (linear interpolation of the authored rings)."""
    a, c, ry, rz = head_rings_local(P)
    pp = P["head_p"]
    aa = np.linspace(a0, a[-1], 200)
    cc = np.interp(aa, a, c)
    yy = np.interp(aa, a, ry)
    zz = np.interp(aa, a, rz)
    f = np.abs(b / yy) ** pp + np.abs((c0 - cc) / zz) ** pp
    out = np.flatnonzero(f > 1.0)
    return float(aa[out[0]]) if len(out) else float(a[-1])


def build_model(P: dict):
    """-> (parts {name: (rings dict, caps, cap_depth, region)}, landmarks {name: xyz}, aux)."""
    parts = {}
    lm = {}
    aux = {}
    # ---- torso
    xs, top, bot, hw = torso_profile(P)
    K = len(xs)
    c = np.stack([xs, np.zeros(K), 0.5 * (top + bot)], 1)
    parts["torso"] = ({"c": c, "u": np.tile([0.0, 1.0, 0.0], (K, 1)), "v": np.tile([0.0, 0.0, 1.0], (K, 1)),
                       "ru": hw, "rv": 0.5 * (top - bot), "p": np.full(K, P["torso_p"])},
                      {"start": "round", "end": "round"},
                      {"start": P["torso_cap_r"], "end": P["torso_cap_f"]}, "torso")
    # ---- head frame
    R = _rot(P["head_yaw"], P["head_pitch"], P["head_roll"])
    H0 = np.array([P["head_x"], P["head_y"], P["head_z"]])

    def H(local):
        return H0 + np.asarray(local, float) @ R.T

    a, cl, ry, rz = head_rings_local(P)
    hc = H(np.stack([a, np.zeros_like(a), cl], 1))
    Kh = len(a)
    parts["head"] = ({"c": hc, "u": np.tile(R[:, 1], (Kh, 1)), "v": np.tile(R[:, 2], (Kh, 1)),
                      "ru": ry, "rv": rz, "p": np.full(Kh, P["head_p"])},
                     {"start": "round", "end": "round"},
                     {"start": 0.6 * P["head_back"] * 0.5, "end": 0.45 * P["muzzle_H"]}, "head")
    aux["head_R"] = R.tolist()
    aux["head_origin"] = H0.tolist()
    # ---- neck: torso anchor -> under the back of the skull
    n0 = np.array([P["neck_x0"], 0.0, P["neck_z0"]])
    n2 = H([-0.4 * P["head_back"], 0.0, -0.45 * P["head_H"]])
    n1 = 0.5 * (n0 + n2) + np.array([0.0, 0.0, P["neck_lift"]])
    rm = P["neck_rmid"] * 0.5 * (P["neck_r0"] + P["neck_r1"])
    parts["neck"] = (_chain_rings([n0, n1, n2], [P["neck_r0"], rm, P["neck_r1"]],
                                  [1.1 * P["neck_r0"], 1.1 * rm, P["neck_r1"]]),
                     {"start": "round", "end": "round"}, None, "neck")
    # ---- ears
    for s, sg in (("R", -1.0), ("L", 1.0)):
        base_l = np.array([P["ear_a"], sg * P["ear_b"], P["ear_c"]])
        tip_l = np.array([P[f"ear{s}_ta"], P[f"ear{s}_tb"], P[f"ear{s}_tc"]])
        mid_l = base_l + 0.5 * (tip_l - base_l)
        pts = H(np.stack([base_l - 0.25 * (tip_l - base_l), base_l, mid_l, tip_l]))
        rings = _chain_rings(pts, [P["ear_w"], P["ear_w"], 0.6 * P["ear_w"], 0.0025],
                             [P["ear_t"], P["ear_t"], 0.75 * P["ear_t"], 0.0015], 2.0, ref=R[:, 1])
        parts[f"ear_{s}"] = (rings, {"start": "round", "end": "round"}, {"start": None, "end": 0.002}, "ears")
        lm[f"ear_tip_{s}"] = pts[-1] + 0.002 * (pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2])
    # ---- eyes / nose / chin (head-local, on the head surface)
    for s, sg in (("R", -1.0), ("L", 1.0)):
        b, cc = sg * P["eye_b"], P["eye_c"]
        ae = head_surface_exit(P, b, cc)
        lm[f"eye_{s}"] = H([ae, b, cc])
    lm["nose"] = H([a[-1] + 0.45 * P["muzzle_H"] - 0.001, 0.0, cl[-1] + 0.25 * rz[-1]])
    ach = 0.8 * P["head_front"] - 0.004
    c_ch = float(np.interp(ach, a, cl) - np.interp(ach, a, rz))
    lm["chin"] = H([ach, 0.0, c_ch])
    # ---- tail
    tp = [np.array([P["tail_x0"], 0.0, P["tail_z0"]])]
    for k in range(1, 5):
        tp.append(np.array([P[f"tail_x{k}"], P[f"tail_y{k}"], P[f"tail_z{k}"]]))
    tp = np.array(tp)
    tr = np.array([P["tail_r0"], 0.5 * (P["tail_r0"] + P["tail_r1"]), P["tail_r1"],
                   0.5 * (P["tail_r1"] + P["tail_r2"]), P["tail_r2"]])
    parts["tail"] = (_chain_rings(tp, tr, tr), {"start": "round", "end": "round"},
                     {"start": P["tail_r0"], "end": P["tail_r2"]}, "tail")
    # ---- legs
    for L in LEGS:
        front = L[0] == "F"
        g = "front" if front else "hind"
        sg = -1.0 if L[1] == "R" else 1.0
        yt = sg * P[f"{g}_yt"]
        xp, yp = P[f"{L}_xp"], P[f"{L}_yp"]
        zt, z1, z2 = P[f"{L}_zt"], P[f"{L}_z1"], P[f"{L}_z2"]

        def yat(z):
            f = np.clip(z / max(zt, 1e-3), 0, 1)
            return yp + (yt - yp) * f

        top = np.array([P[f"{L}_xt"], yt, zt])
        j1 = np.array([P[f"{L}_x1"], yat(z1), z1])
        j2 = np.array([P[f"{L}_x2"], yat(z2), z2])
        rph, rpw = P[f"{g}_rph"], P[f"{g}_rpw"]
        pm = np.array([xp - 0.007, yp, rph - 0.004])
        toe = np.array([xp + 0.014, yp, 0.8 * rph - 0.004])
        pts = [top, j1, j2]
        rl = [P[f"{g}_rt_l"], P[f"{g}_r1_l"], P[f"{g}_r2"]]
        rs = [P[f"{g}_rt_s"], P[f"{g}_r1_s"], P[f"{g}_r2"] * 1.15]
        if not front:   # metatarsus: extra ring between hock and paw
            mt = j2 + 0.55 * (pm - j2)
            pts.append(mt)
            rl.append(P["hind_rh"])
            rs.append(P["hind_rh"] * 1.1)
        pts += [pm, toe]
        rl += [rpw, 0.85 * rpw]
        rs += [rph, 0.8 * rph]
        parts[f"leg_{L}"] = (_chain_rings(pts, rl, rs), {"start": "round", "end": "round"},
                             {"start": None, "end": 0.6 * rph}, f"leg_{L}")
        lm[f"paw_{L}"] = np.array([xp, yp, 0.0])
        aux[f"leg_{L}_joints"] = np.array(pts).tolist()
        if L == "FR":
            lm["shoulder_R"] = top + np.array([0.0, -P["front_rt_l"], 0.0])
        if L == "HR":
            lm["hip_R"] = top + np.array([0.0, -P["hind_rt_l"], 0.0])
            d = j2 - j1
            post = np.array([-d[2], 0.0, d[0]])                 # sagittal normal
            post = post / np.linalg.norm(post)
            if post[0] > 0:
                post = -post
            lm["hock_HR"] = j2 + P["hind_r2"] * 1.1 * post
    # tail landmarks are taken from the part curve itself (arc length)
    return parts, lm, aux


def make_skeleton(P: dict) -> tuple[sk.Skeleton, dict]:
    parts, lm, aux = build_model(P)
    sp = {}
    for name in sk.PART_ORDER:
        if name in parts:
            rings, caps, cd, region = parts[name]
            sp[name] = sk.Part(name, rings, caps, cd, region)
    tail = sp["tail"]
    lm["tail_base"] = tail.center(0.0)
    lm["tail_mid"] = tail.center(0.5)
    lm["tail_tip"] = tail.center(1.0) + tail.tangent(1.0) * (tail.end_cap_depth("end") - 0.003)
    return sk.Skeleton(sp), {"landmarks": lm, "aux": aux}


FIT_SIDES = {"torso": 24, "neck": 12, "head": 18, "ear_R": 8, "ear_L": 8, "tail": 10,
             "leg_FR": 12, "leg_FL": 12, "leg_HR": 12, "leg_HL": 12}
FIT_SEG = {"torso": 3, "neck": 2, "head": 3, "ear_R": 1, "ear_L": 1, "tail": 4,
           "leg_FR": 3, "leg_FL": 3, "leg_HR": 3, "leg_HL": 3}


def model_mesh(skel, hi=False):
    sides = {k: (v * 2 if hi else v) for k, v in FIT_SIDES.items()}
    seg = {k: (v * 2 if hi else v) for k, v in FIT_SEG.items()}
    return sk.skeleton_mesh(skel, sides, seg)


# ---------------------------------------------------------------- reference data / loss
class Ref:
    def __init__(self, size: int):
        from scipy.ndimage import distance_transform_edt
        d = ROOT / "reference" / "derived"
        full = load_mask(d / "mask.png")
        self.full = full
        f = 1536 // size
        self.size = size
        self.f = f
        self.R = full.reshape(size, f, size, f).mean((1, 3)) >= 0.5
        self.bR = boundary(self.R)
        self.dt_bR = distance_transform_edt(~self.bR)
        self.regions = load_regions(d / "regions.json")
        lg = self.regions["legs"]
        s = size / 1536.0
        self.legs_sl = (slice(int(math.ceil(lg[1] * s - 0.5)), int(math.floor(lg[3] * s - 0.5)) + 1),
                        slice(int(math.ceil(lg[0] * s - 0.5)), int(math.floor(lg[2] * s - 0.5)) + 1))
        raw = json.loads((d / "landmarks.json").read_text())["landmarks"]
        self.lm = {k: (np.array(v["uv"], float), float(v.get("sigma_px", 10.0)), v["class"])
                   for k, v in raw.items()}


def silhouette_terms(M, ref: Ref) -> dict:
    from scipy.ndimage import distance_transform_edt
    R = ref.R
    inter = np.logical_and(R, M).sum()
    uni = np.logical_or(R, M).sum()
    iou = inter / max(uni, 1)
    bM = boundary(M)
    if bM.any():
        d1 = ref.dt_bR[bM].mean()
        d2 = distance_transform_edt(~bM)[ref.bR].mean()
    else:
        d1 = d2 = 100.0
    chamfer = 0.5 * (d1 + d2) * ref.f                      # in full-res px
    a, b = ~R[ref.legs_sl], ~M[ref.legs_sl]
    gap = np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1)
    return {"iou": float(iou), "chamfer_px": float(chamfer), "gap_iou": float(gap)}


def landmark_terms(cam, lm3d: dict, ref: Ref) -> dict:
    names = [n for n in ref.lm if n in lm3d]
    P = np.stack([lm3d[n] for n in names])
    uv = cam.project_uv(P)
    err = {}
    sharp, fuzzy = [], []
    for i, n in enumerate(names):
        r_uv, sig, cls = ref.lm[n]
        e = float(np.hypot(*(uv[i] - r_uv)))
        err[n] = e
        z = (e / (sig + 4.0)) ** 2
        (sharp if cls == "sharp" else fuzzy).append(z)
    return {"lm_sharp": float(np.mean(sharp)), "lm_fuzzy": float(np.mean(fuzzy)), "err": err,
            "uv": {n: uv[i].tolist() for i, n in enumerate(names)}}


def _hinge(x, lo, hi, scale):
    if x < lo:
        return ((lo - x) / scale) ** 2
    if x > hi:
        return ((x - hi) / scale) ** 2
    return 0.0


def priors(P: dict, lm: dict, aux: dict, mode: str | None = None) -> dict:
    """Anatomical priors for what one view cannot see.  Returns {term: penalty}."""
    t = {}
    xs, top, bot, hw = torso_profile(P)
    h = top - bot
    # torso width (2*hw) 55-75 % of its height on the body stations
    t["torso_width_ratio"] = sum(_hinge(2 * hw[i] / h[i], 0.55, 0.75, 0.05) for i in range(1, 8))
    sm = 0.0
    for arr in (top, bot, hw):
        d2 = arr[:-2] - 2 * arr[1:-1] + arr[2:]
        sm += float(((d2[1:] / (0.012 if mode == "touchup" else 0.008)) ** 2).sum())   # station 0 exempt
    t["torso_smooth"] = sm
    # rounded rump: the rear station is not a thin stub behind a much bigger section
    rr = 0.93 if mode == "touchup" else (0.75, 0.8)
    rr = (rr, rr) if mode == "touchup" else rr
    t["rump_round"] = (_hinge(h[0] / h[1], rr[0], 1.2, 0.03) + _hinge(hw[0] / hw[1], rr[1], 1.2, 0.03))
    t["torso_positive"] = sum(_hinge(h[i], 0.05, 1.0, 0.01) for i in range(len(h)))
    # withers is the top of the shoulders: no front-half station much higher
    t["withers_top"] = (_hinge(top[7], 0.0, WITHERS_Z, 0.005)
                        + _hinge(top[5], 0.0, WITHERS_Z + (0.03 if mode == "touchup" else 0.015), 0.005))
    # chest-to-rump length 0.40-0.50 m (including end caps)
    L = (xs[-1] + P["torso_cap_f"]) - (xs[0] - P["torso_cap_r"])
    t["body_length"] = _hinge(L, 0.40, 0.50, 0.01)
    # head width 9-10 cm
    t["head_width"] = _hinge(2 * P["head_W"], 0.09, 0.10, 0.004)
    t["head_depth"] = _hinge(P["head_back"] + P["head_front"], 0.085, 0.11, 0.005)
    # stance width 8-11 cm front and hind; paw centroid near y=0
    for g, a_, b_ in (("front", "FR", "FL"), ("hind", "HR", "HL")):
        w = P[f"{b_}_yp"] - P[f"{a_}_yp"]
        t[f"stance_{g}"] = _hinge(w, 0.08, 0.11, 0.002)
        t[f"stance_center_{g}"] = ((P[f"{b_}_yp"] + P[f"{a_}_yp"]) / 0.01) ** 2
    # paw centroid x near 0 (gauge of the x origin)
    px = np.mean([P[f"{L}_xp"] for L in LEGS])
    t["paw_centroid_x"] = (px / 0.02) ** 2
    # left/right symmetry of limb segment lengths
    sym = 0.0
    for a_, b_ in (("FR", "FL"), ("HR", "HL")):
        ja, jb = np.array(aux[f"leg_{a_}_joints"]), np.array(aux[f"leg_{b_}_joints"])
        la = np.linalg.norm(np.diff(ja, axis=0), axis=1)
        lb = np.linalg.norm(np.diff(jb, axis=0), axis=1)
        sym += float((((la - lb) / 0.006) ** 2).sum())
        sym += ((P[f"{a_}_zt"] - P[f"{b_}_zt"]) / 0.01) ** 2
    t["limb_symmetry"] = sym
    # leg tops inside the torso section at that x
    ins = 0.0
    for L in LEGS:
        x, z = P[f"{L}_xt"], P[f"{L}_zt"]
        tz = np.interp(x, xs, top)
        bz = np.interp(x, xs, bot)
        ins += _hinge(z, bz + 0.02, tz - 0.02, 0.008)
        ins += _hinge(x, xs[0] + 0.02, xs[-1] - 0.01, 0.01)
    t["leg_tops_in_torso"] = ins
    # joints bend the right way: elbow behind shoulder/wrist line; knee in front of hip/hock
    bend = 0.0
    for L in LEGS:
        xt, x1, x2 = P[f"{L}_xt"], P[f"{L}_x1"], P[f"{L}_x2"]
        if L[0] == "F":
            bend += _hinge(x1 - min(xt, x2), -1, 0.01, 0.005)
        else:
            bend += _hinge(x1 - max(xt, x2), 0.0, 1, 0.005)
            bend += _hinge(P[f"{L}_z2"], 0.035, 0.09, 0.006)
        bend += _hinge(P[f"{L}_zt"] - P[f"{L}_z1"], 0.03, 1, 0.01)
        bend += _hinge(P[f"{L}_z1"] - P[f"{L}_z2"], 0.03, 1, 0.01)
    t["joint_order"] = bend
    # tail in or near the sagittal plane
    tp = np.array([[P["tail_x0"], 0, P["tail_z0"]]] + [[P[f"tail_x{k}"], P[f"tail_y{k}"], P[f"tail_z{k}"]]
                                                        for k in range(1, 5)])
    seg = np.diff(tp, axis=0)
    sl = np.linalg.norm(seg, axis=1)
    cosang = (seg[1:] * seg[:-1]).sum(1) / (sl[1:] * sl[:-1])
    t["tail_smooth"] = float(sum(_hinge(c, math.cos(math.radians(45)), 2, 0.05) for c in cosang)
                             + sum(_hinge(x, 0.035, 1, 0.005) for x in sl))
    t["tail_sagittal"] = sum((P[f"tail_y{k}"] / 0.03) ** 2 for k in range(1, 5))
    # tail base on the rump (behind the torso rear station, near its top half)
    t["tail_on_rump"] = (_hinge(P["tail_x0"] - (xs[0] - 0.5 * P["torso_cap_r"]), -0.02, 0.02, 0.006)
                         + _hinge(P["tail_z0"] - 0.5 * (top[0] + bot[0]), -0.01, 0.06, 0.006))
    # tail base centre must lie inside the torso (rear dome or body section): attached tail
    x0, z0 = P["tail_x0"], P["tail_z0"]
    zc0 = 0.5 * (top + bot)
    rv = 0.5 * (top - bot)
    if x0 < xs[0]:
        f = ((x0 - xs[0]) / P["torso_cap_r"]) ** 2 + ((z0 - zc0[0]) / rv[0]) ** 2
    else:
        f = ((z0 - np.interp(x0, xs, zc0)) / np.interp(x0, xs, rv)) ** 2
    t["tail_attached"] = _hinge(math.sqrt(f), 0.0, 0.85, 0.05)
    # ears: mirror symmetry of the tips in the head frame (weak), thin
    t["ear_symmetry"] = (((P["earR_ta"] - P["earL_ta"]) / 0.01) ** 2
                         + ((P["earR_tb"] + P["earL_tb"]) / 0.012) ** 2
                         + ((P["earR_tc"] - P["earL_tc"]) / 0.01) ** 2)
    # neck anchored inside the front torso
    tz = np.interp(P["neck_x0"], xs, top)
    t["neck_anchor"] = _hinge(P["neck_z0"], 0.12, tz - 0.01, 0.008) + _hinge(P["neck_x0"], xs[0] + 0.25, xs[-1], 0.01)
    # head above the chest, not floating: head origin within reach of the neck anchor
    dn = np.linalg.norm(np.array([P["head_x"], P["head_y"], P["head_z"]]) - np.array([P["neck_x0"], 0, P["neck_z0"]]))
    t["neck_length"] = _hinge(dn, 0.06, 0.15, 0.01)
    t["head_y"] = (P["head_y"] / 0.02) ** 2
    # neck no wider than the chest it leaves nor the head it enters (no "collar")
    t["neck_width"] = (_hinge(P["neck_r0"] - np.interp(P["neck_x0"], xs, hw), -1, 0.0, 0.004)
                       + _hinge(P["neck_r1"] - 0.85 * P["head_W"], -1, 0.0, 0.004))
    return t


PRIOR_WEIGHT = 0.05


def objective(vec, active_mask, base_vec, ref: Ref, weights: dict, return_detail=False):
    full = base_vec.copy()
    full[active_mask] = vec
    P = as_dict(full)
    try:
        skel, extra = make_skeleton(P)
        V, T, _, _ = model_mesh(skel)
        cam = camera_from(P)
        cams = cam.scaled(ref.size)
        M = raster.silhouette(V, T, cams)
    except Exception as exc:  # noqa: BLE001
        if return_detail:
            raise
        return 1e3
    s = silhouette_terms(M, ref)
    l = landmark_terms(cam, extra["landmarks"], ref)
    pr = priors(P, extra["landmarks"], extra["aux"], weights.get("mode"))
    prior_sum = float(sum(pr.values()))
    total = (weights["iou"] * (1 - s["iou"]) + weights["chamfer"] * s["chamfer_px"]
             + weights["gap"] * (1 - s["gap_iou"]) + weights["lm"] * l["lm_sharp"]
             + weights["lmf"] * l["lm_fuzzy"] + weights.get("prior", PRIOR_WEIGHT) * prior_sum)
    # out-of-bounds penalty (cma bounds keep us inside, this is a guard)
    if return_detail:
        return total, {"sil": s, "lm": l, "priors": pr, "prior_sum": prior_sum}
    return float(total)


# ---------------------------------------------------------------- workers
_G = {}


def _init_worker(size):
    _G["ref"] = Ref(size)


def _eval(args):
    vec, mask, base, weights = args
    return objective(vec, mask, base, _G["ref"], weights)


# ---------------------------------------------------------------- stages
def group(*prefixes, exclude=()):
    out = []
    for n in NAMES:
        if any(n.startswith(p) for p in prefixes) and not any(n.startswith(e) for e in exclude):
            out.append(n)
    return out


CAMERA = group("cam_")
TORSO = group("torso_", "neck_")
HEAD = group("head_", "muzzle_", "eye_", "ear")
LEGS_P = group("FR_", "FL_", "HR_", "HL_", "front_", "hind_")
TAIL = group("tail_")
GLOBAL_POSE = (CAMERA + ["head_x", "head_y", "head_z", "head_yaw", "head_pitch", "head_roll"]
               + [f"{L}_{k}" for L in LEGS for k in ("xp", "yp")]
               + [f"tail_{a}{k}" for k in range(1, 5) for a in ("x", "z")] + ["tail_x0", "tail_z0"]
               + ["earR_tb", "earR_tc", "earL_tb", "earL_tc"])

W_BASE = {"iou": 4.0, "chamfer": 0.08, "gap": 1.5, "lm": 0.25, "lmf": 0.05, "prior": PRIOR_WEIGHT}
STAGES = {
    1: ("camera + global pose (landmarks, paws)", GLOBAL_POSE, 256, 220, 0.5,
        {**W_BASE, "iou": 2.0, "chamfer": 0.04, "gap": 0.5, "lm": 0.6}),
    2: ("torso + neck (+camera)", TORSO + CAMERA, 256, 260, 0.5, W_BASE),
    3: ("head + ears + eyes", HEAD + CAMERA[3:], 384, 260, 0.5, W_BASE),
    4: ("legs", LEGS_P, 384, 320, 0.5, W_BASE),
    5: ("tail", TAIL, 384, 180, 0.5, W_BASE),
    6: ("refine torso + neck + camera", TORSO + CAMERA, 512, 200, 0.2, W_BASE),
    7: ("refine head + ears", HEAD + CAMERA[3:], 512, 160, 0.2, W_BASE),
    8: ("refine legs", LEGS_P, 512, 200, 0.2, W_BASE),
    9: ("refine tail", TAIL, 512, 100, 0.2, W_BASE),
    10: ("everything", NAMES, 512, 350, 0.08, W_BASE),
    # touch-up with the camera frozen at the LOCKED contract values: higher resolution and
    # boundary weight; dorsal profile less smoothed, withers-adjacent station may rise 30 mm,
    # near-cylindrical rear station so the rear dome closes without a step (no rump collar)
    11: ("touch-up, camera frozen (locked)", [n for n in NAMES if not n.startswith("cam_")], 768, 300, 0.1,
         {**W_BASE, "chamfer": 0.14, "mode": "touchup"}),
}


def run_stage(k, base_vec, pool, trace, quick=False, seed=1):
    import cma
    title, names, size, iters, sigma, weights = STAGES[k]
    if quick:
        iters = max(20, iters // 6)
    mask = np.array([n in names for n in NAMES])
    idx = np.flatnonzero(mask)
    x0 = np.zeros(len(idx))
    lo = (LO[idx] - base_vec[idx]) / SCALE[idx]
    hi = (HI[idx] - base_vec[idx]) / SCALE[idx]
    lo, hi = np.minimum(lo, -1e-6), np.maximum(hi, 1e-6)

    def to_full_active(z):
        return base_vec[idx] + z * SCALE[idx]

    pop = 4 + int(3 * math.log(len(idx))) + 6
    es = cma.CMAEvolutionStrategy(x0, sigma, {"bounds": [lo.tolist(), hi.tolist()], "seed": seed + k,
                                              "popsize": pop, "maxiter": iters, "verbose": -9,
                                              "tolfun": 1e-7, "tolx": 1e-5})
    if pool is None:
        _init_worker(size)
    t0 = time.time()
    f0 = _eval((base_vec[idx], mask, base_vec, weights)) if pool is None else \
        pool.map(_eval, [(base_vec[idx], mask, base_vec, weights)])[0]
    best = (float(f0), base_vec[idx].copy())          # never return worse than the start
    rows = []
    # re-init pool for this resolution
    while not es.stop():
        Z = es.ask()
        args = [(to_full_active(np.asarray(z)), mask, base_vec, weights) for z in Z]
        if pool is not None:
            f = pool.map(_eval, args)
        else:
            f = [_eval(a) for a in args]
        es.tell(Z, f)
        i = int(np.argmin(f))
        if f[i] < best[0]:
            best = (float(f[i]), to_full_active(np.asarray(Z[i])))
        rows.append({"iter": es.countiter, "best": best[0], "gen_min": float(min(f)),
                     "sigma": float(es.sigma), "t": round(time.time() - t0, 2)})
    out = base_vec.copy()
    if best[1] is not None:
        out[idx] = best[1]
    trace.append({"stage": k, "title": title, "start_objective": float(f0), "n_params": int(len(idx)), "popsize": pop,
                  "size": size, "weights": weights, "iters": rows,
                  "evals": int(es.countevals), "seconds": round(time.time() - t0, 1),
                  "stop": {k2: str(v) for k2, v in es.stop().items()}})
    return out


def evaluate_full(vec, size=512) -> dict:
    ref = Ref(size)
    total, det = objective(vec, np.ones(len(NAMES), bool), vec, ref, W_BASE, return_detail=True)
    det["total"] = total
    return det


def save_overlay(vec, path: Path, title: str):
    from catlib.overlay import make_overlay
    P = as_dict(vec)
    skel, extra = make_skeleton(P)
    V, T, _, _ = model_mesh(skel, hi=True)
    cam = camera_from(P)
    M = raster.silhouette(V, T, cam)
    ref_full = load_mask(ROOT / "reference" / "derived" / "mask.png")
    refl = load_ref_landmarks(ROOT / "reference" / "derived" / "landmarks.json")
    uv = cam.project_uv(np.stack([extra["landmarks"][n] for n in refl]))
    lmrows = {n: {"proj_uv": uv[i].tolist(), "ref_uv": refl[n]["uv"]} for i, n in enumerate(refl)}
    inter = np.logical_and(ref_full, M).sum() / np.logical_or(ref_full, M).sum()
    errs = [np.hypot(*(np.array(lmrows[n]["proj_uv"]) - lmrows[n]["ref_uv"])) for n in refl
            if refl[n]["class"] == "sharp"]
    metrics = {"official": False, "iou": float(inter), "landmarks": lmrows,
               "landmark_sharp_mean_px": float(np.mean(errs)), "boundary_mean_px": None,
               "boundary_hd95_px": None, "legs_gap_iou": None}
    make_overlay(path, M, metrics, load_regions(ROOT / "reference" / "derived" / "regions.json"),
                 ref_mask=ref_full, title=title)
    return float(inter), float(np.mean(errs))


def write_outputs(vec, trace, stages_done):
    P = as_dict(vec)
    skel, extra = make_skeleton(P)
    cam = camera_from(P)
    det = evaluate_full(vec, 512)
    meta = {"camera_fit": {"location": cam.location.tolist(),
                           "look_at": [P["cam_lx"], P["cam_ly"], P["cam_lz"]],
                           "azimuth_deg": P["cam_az"], "elevation_deg": P["cam_el"],
                           "distance_m": P["cam_dist"]},
            "landmarks_model": {k: np.asarray(v).tolist() for k, v in extra["landmarks"].items()},
            "generator": "scripts/fit_camera_proxy.py"}
    skel.meta = meta
    sk.save_skeleton(skel, PROXY / "skeleton.json")
    out = {"params": P, "spec": {n: SPEC[n] for n in NAMES},
           "camera": meta["camera_fit"], "landmarks_model": meta["landmarks_model"],
           "aux": extra["aux"], "stages_done": stages_done,
           "final_internal_scores_512": {"sil": det["sil"], "lm_err_px": det["lm"]["err"],
                                         "priors": det["priors"], "total": det["total"]},
           "priors_doc": PRIORS_DOC, "withers_gauge_m": WITHERS_Z}
    (PROXY / "fit_params.json").write_text(json.dumps(out, indent=1))
    old = []
    tp = PROXY / "fit_trace.json"
    if tp.is_file() and trace and stages_done and min(t["stage"] for t in trace) > 1:
        old = [t for t in json.loads(tp.read_text()) if t["stage"] not in {t2["stage"] for t2 in trace}]
    tp.write_text(json.dumps(old + trace))


PRIORS_DOC = {
    "torso_smooth": "second differences of torso top/bottom/half-width over stations 1-7 (8 mm scale)",
    "torso_width_ratio": "torso width (2*half-width) 55-75 % of its height, stations 1-6",
    "withers_top": "withers gauge: torso top at station 6 (above the shoulder joints) fixed at 0.25 m; "
                   "chest station 7 not above it, station 5 at most 15 mm above it; back/rump free",
    "body_length": "chest-to-rump length incl. end caps 0.40-0.50 m",
    "head_width": "head (skull) width 9-10 cm",
    "head_depth": "head back-of-skull to muzzle tip 8.5-11 cm",
    "stance_front/hind": "paw y separation 8-11 cm; paw pair centred on y=0",
    "paw_centroid_x": "x origin gauge: mean paw x near 0",
    "limb_symmetry": "left/right segment lengths and top-joint heights equal (radii shared exactly)",
    "leg_tops_in_torso": "shoulder/hip joints inside the torso section",
    "joint_order": "elbow behind shoulder/wrist, stifle in front of hip/hock, hock 3.5-9 cm high",
    "tail_sagittal": "tail control points near y=0",
    "tail_smooth": "tail control polygon turns <= 45 deg per joint, segments >= 3.5 cm (no hooks)",
    "rump_round": "rear torso station height >= 75 % and half-width >= 80 % of the next station",
    "tail_on_rump": "tail base on the rear of the torso, upper half",
    "tail_attached": "tail base centre inside the torso rear dome / section (<= 0.85 normalised radius)",
    "ear_symmetry": "ear tips mirror-symmetric in the head frame (weak)",
    "neck_anchor/neck_length": "neck anchored in the front torso, head within 6-15 cm of it",
    "head_y": "head origin near the sagittal plane",
    "neck_width": "neck base radius <= chest half-width at the anchor; neck top <= 0.85 head half-width",
    "fixed_by_construction": "paw soles on z=0, body sagittal plane y=0, left/right leg radii shared, "
                             "roll 0, 85 mm, principal point centred",
}


def freeze_locked_camera(vec):
    """Set the camera parameters from the LOCKED contract/camera.json (stage 11 never moves them)."""
    d = json.loads((ROOT / "contract" / "camera.json").read_text())
    if d.get("status") != "locked":
        raise SystemExit("stage 11 needs the locked contract camera")
    loc, look = np.array(d["location"]), np.array(d["look_at"])
    v = loc - look
    dist = float(np.linalg.norm(v))
    P = as_dict(vec)
    P.update(cam_lx=look[0], cam_ly=look[1], cam_lz=look[2], cam_dist=dist,
             cam_el=math.degrees(math.asin(v[2] / dist)), cam_az=math.degrees(math.atan2(v[1], v[0])))
    err = float(np.abs(camera_from(P).location - loc).max())
    assert err < 1e-9, err
    return as_vec(P)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="1,2,3,4,5,6,7,8,9,10,11")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="start from the last stage checkpoint")
    ap.add_argument("--init", default=None, help="start from a params json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--overlay-only", action="store_true")
    args = ap.parse_args(argv)
    STAGES_DIR.mkdir(parents=True, exist_ok=True)
    stages = [int(s) for s in args.stages.split(",") if s]
    vec = INIT.copy()
    done = []
    if args.init:
        vec = as_vec(json.loads(Path(args.init).read_text())["params"])
    elif args.resume:
        ck = sorted(STAGES_DIR.glob("stage*_params.json"))
        ck = sorted(ck, key=lambda c: int(c.stem[5:].split("_")[0]))
        prev = [c for c in ck if int(c.stem[5:].split("_")[0]) < min(stages)]
        if prev:
            d = json.loads(prev[-1].read_text())
            vec = as_vec(d["params"])
            done = d.get("stages_done", [])
            print(f"resumed from {prev[-1].name}")
    vec = np.clip(vec, LO, HI)
    if 11 in stages:
        vec = freeze_locked_camera(vec)
    if args.overlay_only:
        iou, lm = save_overlay(vec, STAGES_DIR / "stage0_overlay.png", "stage 0 (init)")
        print(f"init raster IoU {iou:.4f}, sharp lm mean {lm:.1f}px")
        det = evaluate_full(vec)
        print(json.dumps({k: round(v, 3) for k, v in det["lm"]["err"].items()}))
        print({k: round(v, 3) for k, v in det["priors"].items() if v > 0})
        return
    if not done:
        save_overlay(vec, STAGES_DIR / "stage0_overlay.png", "stage 0 (init)")
    import multiprocessing as mp
    trace = []
    t_all = time.time()
    for k in stages:
        size = STAGES[k][2]
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init_worker, initargs=(size,)) as pool:
            vec = run_stage(k, vec, pool, trace, args.quick)
        done.append(k)
        iou, lm = save_overlay(vec, STAGES_DIR / f"stage{k}_overlay.png",
                               f"stage {k}: {STAGES[k][0]}")
        det = evaluate_full(vec)
        ck = {"stage": k, "params": as_dict(vec), "stages_done": done, "raster_iou_full": iou,
              "sharp_lm_mean_px": lm, "internal": {"sil": det["sil"], "lm_err": det["lm"]["err"],
                                                   "priors": det["priors"], "total": det["total"]}}
        (STAGES_DIR / f"stage{k}_params.json").write_text(json.dumps(ck, indent=1))
        print(f"[stage {k}] {STAGES[k][0]}: total {det['total']:.4f}  IoU(full raster) {iou:.4f}  "
              f"sharp lm {lm:.1f}px  chamfer {det['sil']['chamfer_px']:.1f}px  gap {det['sil']['gap_iou']:.3f}  "
              f"t={trace[-1]['seconds']}s", flush=True)
        print("   lm err:", {n: round(e, 1) for n, e in det["lm"]["err"].items()}, flush=True)
        print("   priors>0:", {n: round(e, 3) for n, e in det["priors"].items() if e > 1e-4}, flush=True)
    write_outputs(vec, trace, done)
    print(f"done in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
