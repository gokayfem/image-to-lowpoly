"""Budget-100 cat: hand-allocated low-poly topology built from the shared proxy skeleton.

    python3 variants/100/build.py --design A --out variants/100/checkpoints/v0_initial/mesh.json
    python3 variants/100/build.py --design B --out variants/100/work/B0.json

Deterministic.  Asserts exactly 100 triangles.  Every part is a generalised-cylinder chain
of polygonal rings resampled from proxy/skeleton.json (catlib.skeleton); ring k uses the
angle convention theta_k = phase + 2*pi*k/n around n = u x v, so consecutive rings of a
chain are already aligned and every band triangle is wound outward by construction.

Designs (see NOTES.md for the allocation tables and the comparison):
  A  continuous torso->neck->head tube (4-sided box sections), muzzle = tip vertex,
     open 3-sided leg tubes buried in the torso, hock bends on both hind legs,
     3-sided tail with 3 segments, ears = open 3-sided pyramids buried in the head.
  B  separate closed head (two 5-sided rings + muzzle/back tips) on a 4-sided torso whose
     neck ring ends open inside the head, straight legs, longer tail.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # the case directory (examples/cat)
import os as _os  # noqa: E402
_os.environ["LOWPOLY_CASE"] = str(ROOT)  # catlib.paths reads the case from here
sys.path.insert(0, _os.environ.get("LOWPOLY_SCRIPTS",
                                   str(ROOT.parents[1] / "skills" / "image-to-lowpoly" / "scripts")))

import numpy as np  # noqa: E402

from catlib import meshio  # noqa: E402
from catlib import proxy as cproxy  # noqa: E402
from catlib.skeleton import load_skeleton  # noqa: E402

BUDGET = 100


def unit(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a)


def ring(c, u, v, ru, rv, n, phase=0.0):
    th = phase + 2 * np.pi * np.arange(n) / n
    return (np.asarray(c, float)[None] + ru * np.cos(th)[:, None] * np.asarray(u, float)[None]
            + rv * np.sin(th)[:, None] * np.asarray(v, float)[None])


def frame_from_axis(n, lateral_hint):
    """u lateral (projected hint), v = n x u, so u x v = n."""
    n = unit(n)
    u = np.asarray(lateral_hint, float)
    u = unit(u - np.dot(u, n) * n)
    v = np.cross(n, u)
    return u, v, n


class Builder:
    def __init__(self):
        self.V: list[np.ndarray] = []
        self.T: list[tuple] = []
        self.reg: list[str] = []
        self.mat: list[str] = []

    def add_verts(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        i0 = len(self.V)
        self.V.extend(list(P))
        return list(range(i0, i0 + len(P)))

    def tri(self, a, b, c, region, material="fur_orange"):
        self.T.append((a, b, c))
        self.reg.append(region)
        self.mat.append(material)
        return len(self.T) - 1

    def chain(self, rings, region, start="open", end="open", start_tip=None, end_tip=None,
              regions=None, mat_fn=None, phases=None):
        """rings: list of (n,3) arrays with the theta convention around the chain direction.
        start/end: 'open' | 'tip' | 'fan'.  regions: optional per-band region list.
        Returns dict of vertex-index rings, tip indices and triangle ids per band."""
        mat_fn = mat_fn or (lambda kind, k, j, pts: "fur_orange")
        idx = [self.add_verts(r) for r in rings]
        out = {"rings": idx, "bands": [], "start_tris": [], "end_tris": []}
        for k in range(len(idx) - 1):
            A, B = idx[k], idx[k + 1]
            n = len(A)
            rg = regions[k] if regions else region
            ids = []
            if len(B) == n:
                for j in range(n):
                    j1 = (j + 1) % n
                    ids.append(self.tri(A[j], A[j1], B[j1], rg, mat_fn("band", k, j, None)))
                    ids.append(self.tri(A[j], B[j1], B[j], rg, mat_fn("band", k, j, None)))
            else:
                # unequal bridge: merge the two rings by their theta (phases[k] + 2 pi i / n)
                m = len(B)
                pa, pb = phases[k], phases[k + 1]
                ta = (pa + 2 * np.pi * np.arange(n) / n) % (2 * np.pi)
                tb = (pb + 2 * np.pi * np.arange(m) / m) % (2 * np.pi)
                j0 = int(np.argmin(np.abs((tb - ta[0] + np.pi) % (2 * np.pi) - np.pi)))
                ua = ta[0] + 2 * np.pi * np.arange(n + 1) / n
                ub = ta[0] + ((tb[(j0 + np.arange(m + 1)) % m] - ta[0] + np.pi) % (2 * np.pi) - np.pi)
                ub = ub[0] + 2 * np.pi * np.arange(m + 1) / m
                i = j = 0
                while i < n or j < m:
                    if j >= m or (i < n and ua[i + 1] <= ub[j + 1]):
                        ids.append(self.tri(A[i % n], A[(i + 1) % n], B[(j0 + j) % m], rg,
                                            mat_fn("band", k, i, None)))
                        i += 1
                    else:
                        ids.append(self.tri(A[i % n], B[(j0 + j + 1) % m], B[(j0 + j) % m], rg,
                                            mat_fn("band", k, i % n, None)))
                        j += 1
            out["bands"].append(ids)
        A = idx[0]
        n = len(A)
        if start == "tip":
            t = self.add_verts(start_tip)[0]
            out["start_tip"] = t
            for j in range(n):
                out["start_tris"].append(self.tri(A[(j + 1) % n], A[j], t, region,
                                                  mat_fn("start", 0, j, None)))
        elif start == "fan":
            for j in range(1, n - 1):
                out["start_tris"].append(self.tri(A[0], A[j + 1], A[j], region,
                                                  mat_fn("start", 0, j, None)))
        B = idx[-1]
        n = len(B)
        if end == "tip":
            t = self.add_verts(end_tip)[0]
            out["end_tip"] = t
            for j in range(n):
                out["end_tris"].append(self.tri(B[j], B[(j + 1) % n], t, region,
                                                mat_fn("end", 0, j, None)))
        elif end == "fan":
            for j in range(1, n - 1):
                out["end_tris"].append(self.tri(B[0], B[j], B[j + 1], region,
                                                mat_fn("end", 0, j, None)))
        return out


# ---------------------------------------------------------------- burying open rims

def winding_number(P, V, T):
    """Generalised winding number of points P (N,3) w.r.t. triangles T (Van Oosterom-Strackee)."""
    P = np.atleast_2d(P)
    A = V[T[:, 0]][None] - P[:, None]
    Bv = V[T[:, 1]][None] - P[:, None]
    C = V[T[:, 2]][None] - P[:, None]
    a, b, c = (np.linalg.norm(X, axis=2) for X in (A, Bv, C))
    det = np.einsum("ijk,ijk->ij", A, np.cross(Bv, C))
    den = a * b * c + np.einsum("ijk,ijk->ij", A, Bv) * c + np.einsum("ijk,ijk->ij", Bv, C) * a \
        + np.einsum("ijk,ijk->ij", C, A) * b
    return (2 * np.arctan2(det, den)).sum(1) / (4 * np.pi)


def bury(V, rim, host_T, anchor, margin=0.004, max_steps=60, shrink=0.96, slide=0.06):
    """Move an open rim as a whole (shrink about its centre by 4 % and slide its centre 6 %
    towards `anchor` per step, which keeps the ring untwisted) until every rim vertex,
    probed `margin` further out, has winding number > 0.5 w.r.t. the host surface."""
    rim = list(rim)
    anchor = np.asarray(anchor, float)
    for step in range(max_steps):
        P = V[rim]
        c = P.mean(0)
        d = P - c
        probe = P + margin * d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
        if (winding_number(probe, V, host_T) > 0.5).all():
            return step
        c2 = c + slide * (anchor - c)
        V[rim] = c2 + shrink * d
    return max_steps


# ---------------------------------------------------------------- part constructors

def skel_ring(part, t, n, phase=0.0, scale=1.0, su=1.0, sv=1.0, shift=(0, 0, 0)):
    c = part.center(t)
    u, v, _ = part.frame(t)
    ru, rv, p = part.radii(t)
    return ring(c + np.asarray(shift, float), u, v, ru * scale * su, rv * scale * sv, n, phase)


def build_leg(B, sk, name, ts_mid, top_t=0.10, n=3, sole_len=(0.020, 0.010), sole_half=0.011,
              top_scale=0.75, top_inset=0.3, mid_scale=1.0, sole_at_landmark=False):
    """3-sided open leg tube from a buried top ring down to a flat triangular sole on z=0
    (sole = the end fan, one triangle).  Rings use n = down the leg, u = cat-left (+Y)."""
    part = sk[name]
    ts = [top_t] + list(ts_mid)
    cs = [part.center(t).copy() for t in ts]
    cs[0][1] *= (1.0 - top_inset)
    pe = part.center(1.0).copy()
    pe[2] = 0.0
    pts_chain = cs + [pe]
    rings = []
    for k, t in enumerate(ts):
        c = cs[k]
        d_out = unit(pts_chain[k + 1] - c)
        d_in = unit(c - pts_chain[k - 1]) if k > 0 else d_out
        u, v, nn = frame_from_axis(unit(d_in + d_out), (0, 1, 0))
        ru, rv, _ = part.radii(t)
        s = top_scale if k == 0 else mid_scale
        r = 0.5 * (ru + min(rv, 1.6 * ru)) * s
        rings.append(ring(c, u, v, r, r, n, phase=np.pi / 2))   # vertex 0 at the front (+X)
    # sole triangle on z = 0: one vertex forward (+X), two back corners; ordering must follow
    # the chain's theta convention seen from n = down.
    pe = part.center(1.0).copy()
    pe[2] = 0.0
    fwd = np.array([1.0, 0, 0])
    lat = np.array([0, 1.0, 0])
    # previous ring order around n(down): theta convention with u=+Y-ish, v = n x u
    prev = rings[-1]
    if sole_at_landmark:
        # centre the sole triangle's centroid on the proxy paw landmark (the paw_* landmark is
        # this triangle's centroid, CONTRACT §12)
        pl = cproxy.landmarks3d()["paw_" + name.split("_")[1]].copy()
        pl[2] = 0.0
        pe = pl - fwd * (sole_len[0] - 2 * sole_len[1]) / 3.0
    sole_pts = np.array([pe + fwd * sole_len[0], pe - fwd * sole_len[1] + lat * sole_half,
                         pe - fwd * sole_len[1] - lat * sole_half])
    # orient the sole CCW around n = down (normal pointing to -Z), then choose the cyclic
    # rotation closest to the previous ring so the last band does not twist.
    if np.cross(sole_pts[1] - sole_pts[0], sole_pts[2] - sole_pts[0])[2] > 0:
        sole_pts = sole_pts[[0, 2, 1]]
    best = None
    for r in range(3):
        o = [(r + k) % 3 for k in range(3)]
        cost = float(np.linalg.norm(prev - sole_pts[o], axis=1).sum())
        if best is None or cost < best[0]:
            best = (cost, o)
    order = best[1]
    rings.append(sole_pts[order])
    region = {"leg_FR": "leg_FR", "leg_FL": "leg_FL", "leg_HR": "leg_HR", "leg_HL": "leg_HL"}[name]

    def mf(kind, k, j, _):
        if kind == "end":
            return "fur_cream"
        return "fur_orange"
    out = B.chain(rings, region, start="open", end="fan", mat_fn=mf)
    return out


def build_tail(B, sk, ts=(0.0, 0.33, 0.66), n=3, inset=0.018, tip_t=1.0, extra_tip=0.006, base_n=3):
    part = sk["tail"]
    rings = []
    for k, t in enumerate(ts):
        c = part.center(t)
        tan = part.tangent(t)
        if k == 0:
            c = c - tan * inset                       # bury the open start inside the rump
        u, v, nn = frame_from_axis(tan, (0, 1, 0))
        ru, rv, _ = part.radii(t)
        r = 0.5 * (ru + rv) * (1.15 if k == 0 else 1.05)
        nk = base_n if k == 0 else n
        rings.append(ring(c, u, v, r, r, nk, phase=np.pi / 2))   # one vertex on top
    tip = part.center(tip_t) + part.tangent(tip_t) * extra_tip

    def mf(kind, k, j, _):
        return "fur_stripe" if (kind == "band" and k == len(ts) - 2) else "fur_orange"
    phases = [np.pi / 2] * len(rings)
    return B.chain(rings, "tail", start="open", end="tip", end_tip=tip, mat_fn=mf, phases=phases)


def build_ear(B, sk, name, inset=0.010, width=1.05, thick=0.008):
    """Open 3-sided pyramid: base triangle buried in the skull, apex = ear tip.
    Base: two corners along the ear's width axis (front edge), one behind (thickness)."""
    part = sk[name]
    c0 = part.center(0.0)
    tip = cproxy.landmarks3d()["ear_tip_R" if name == "ear_R" else "ear_tip_L"]
    axis = unit(tip - c0)
    c = c0 - axis * inset
    u, v, _ = part.frame(0.0)                      # u: ear width, v: ear thickness (front/back)
    ru, rv, _ = part.radii(0.0)
    # ensure v points to the ear's back (away from the face / towards -X roughly)
    if v[0] > 0:
        v = -v
    u = unit(u - np.dot(u, axis) * axis)
    v = unit(v - np.dot(v, axis) * axis - np.dot(v, u) * u)
    pts = np.array([c + u * ru * width, c + v * thick, c - u * ru * width])
    # order so that (pts, theta) is CCW around axis
    nrm = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    if np.dot(nrm, axis) < 0:
        pts = pts[::-1]
    front_pair = None
    out = B.chain([pts], "ears", start="open", end="tip", end_tip=tip)
    # material: inner ear = the face whose base edge is the two width corners (front face)
    R = out["rings"][0]
    for tid in out["end_tris"]:
        a, b, _ = B.T[tid]
        P = np.array(B.V)
        mid = 0.5 * (P[a] + P[b])
        if np.dot(mid - c, v) < 0.25 * thick:      # edge not touching the back corner
            B.mat[tid] = "ear_inner"
    return out


def build_spine_A(B, sk, cfg):
    """Continuous rump-tip -> torso -> neck -> head -> muzzle-tip tube, 4-sided box rings
    (phase pi/4: corners at upper/lower left/right; for the head these are temples/cheeks)."""
    T, N, H = sk["torso"], sk["neck"], sk["head"]
    n = 4
    ph = np.pi / 4
    rings, regs = [], []
    for t, s in zip(cfg["torso_ts"], cfg["torso_scale"]):
        rings.append(skel_ring(T, t, n, ph, su=s[0], sv=s[1]))
    for t, s in zip(cfg["neck_ts"], cfg["neck_scale"]):
        rings.append(skel_ring(N, t, n, ph, su=s[0], sv=s[1]))
    for t, s in zip(cfg["head_ts"], cfg["head_scale"]):
        rings.append(skel_ring(H, t, n, ph, su=s[0], sv=s[1]))
    nt, nn_ = len(cfg["torso_ts"]), len(cfg["neck_ts"])
    for k in range(len(rings) - 1):
        regs.append("torso" if k < nt - 1 or (k == nt - 1 and nn_ > 0) else ("neck" if k < nt + nn_ - 1 else "head"))
    regs = [r if i < nt else r for i, r in enumerate(regs)]
    # label: bands fully inside torso part -> torso; torso->neck -> torso (chest);
    # neck->head and head bands -> head
    lab = []
    for k in range(len(rings) - 1):
        if k < nt - 1:
            lab.append("torso")
        elif nn_ == 0 and k == nt - 1:
            lab.append("neck")
        elif k < nt - 1 + nn_:
            lab.append("torso" if k == nt - 1 else "neck")
        else:
            lab.append("head")
    if nn_ and lab[nt - 1 + nn_ - 1 + 1 - 1] == "neck":
        pass
    rump_tip = T.center(0.0) - T.tangent(0.0) * cfg["rump_depth"]
    muzzle = cproxy.landmarks3d()["nose"] + H.tangent(1.0) * cfg["muzzle_push"]

    head_band0 = nt - 1 + nn_

    def mf(kind, k, j, _):
        # j: quad index between corner j and j+1; corners at theta = pi/4 + j*pi/2
        # -> j=0 top (theta pi/4..3pi/4), j=1 right side, j=2 bottom, j=3 left side
        if kind == "band" and j == 2 and (k == nt - 2 or k == nt - 1):
            return "fur_cream"                       # chest / throat underside
        if kind == "end" and j == 2:
            return "fur_cream"                       # chin / muzzle underside
        if kind == "end" and j == 0:
            return "fur_stripe"
        return "fur_orange"
    out = B.chain(rings, "torso", start=cfg.get("rump_cap", "tip"), start_tip=rump_tip, end="tip", end_tip=muzzle,
                  regions=lab, mat_fn=mf)
    for tid in out["end_tris"]:
        B.reg[tid] = "head"
    out["n_torso"] = nt
    out["n_neck"] = nn_
    return out


def build_spine_B(B, sk, cfg):
    """4-sided torso tube (rump tip) + neck rings, neck end open inside a separate closed head."""
    T, N, H = sk["torso"], sk["neck"], sk["head"]
    ph = np.pi / 4
    rings = [skel_ring(T, t, 4, ph, su=s[0], sv=s[1]) for t, s in zip(cfg["torso_ts"], cfg["torso_scale"])]
    rings += [skel_ring(N, t, 4, ph, su=s[0], sv=s[1]) for t, s in zip(cfg["neck_ts"], cfg["neck_scale"])]
    nt = len(cfg["torso_ts"])
    lab = ["torso"] * (nt - 1) + ["torso"] + ["neck"] * (len(cfg["neck_ts"]) - 1)
    lab = lab[:len(rings) - 1]
    rump_tip = T.center(0.0) - T.tangent(0.0) * cfg["rump_depth"]

    def mf(kind, k, j, _):
        if kind == "band" and j == 2 and k >= nt - 2:
            return "fur_cream"
        return "fur_orange"
    body = B.chain(rings, "torso", start=cfg.get("rump_cap", "tip"), start_tip=rump_tip, end="open", regions=lab, mat_fn=mf)
    # head: back tip, two 5-sided rings, muzzle tip.  Ring 5: top, upper-left/right (temples),
    # lower-left/right (cheeks).  phase pi/2 puts vertex 0 on top.
    hr = [skel_ring(H, t, 5, np.pi / 2, su=s[0], sv=s[1]) for t, s in zip(cfg["head_ts"], cfg["head_scale"])]
    back = H.center(0.0) - H.tangent(0.0) * cfg["head_back"]
    muzzle = cproxy.landmarks3d()["nose"] + H.tangent(1.0) * cfg["muzzle_push"]

    def mfh(kind, k, j, _):
        if kind == "end" and j in (2, 3):
            return "fur_cream"
        return "fur_orange"
    head = B.chain(hr, "head", start="tip", start_tip=back, end="tip", end_tip=muzzle, mat_fn=mfh)
    body["head"] = head
    return body


DESIGNS = {
    "A": dict(
        spine="A",
        torso_ts=[0.0, 0.45, 0.92], torso_scale=[(1.25, 1.2), (1.3, 1.3), (1.35, 1.35)],
        neck_ts=[0.6], neck_scale=[(1.0, 1.0)],
        head_ts=[0.45], head_scale=[(1.3, 1.35)],
        rump_depth=0.045, muzzle_push=0.004,
        legs={"leg_FR": dict(ts_mid=[]), "leg_FL": dict(ts_mid=[]),
              "leg_HR": dict(ts_mid=[0.55]), "leg_HL": dict(ts_mid=[0.55])},
        tail=dict(ts=(0.0, 0.33, 0.66), base_n=4), rump_cap="fan",
    ),
    # A2 = A + sole triangles centred on the proxy paw landmarks (used from v2 on, with the
    # soles frozen during refinement so the refiner cannot spin the paw footprints)
    "A2": None,
    "B": dict(
        spine="B",
        torso_ts=[0.0, 0.95], torso_scale=[(1.25, 1.2), (1.35, 1.35)],
        neck_ts=[0.75], neck_scale=[(0.95, 0.95)],
        head_ts=[0.22, 0.62], head_scale=[(1.1, 1.15), (1.05, 1.1)],
        rump_depth=0.045, head_back=0.010, muzzle_push=0.004,
        legs={"leg_FR": dict(ts_mid=[]), "leg_FL": dict(ts_mid=[]),
              "leg_HR": dict(ts_mid=[0.55]), "leg_HL": dict(ts_mid=[])},
        tail=dict(ts=(0.0, 0.25, 0.5, 0.75), base_n=4), rump_cap="fan",
    ),
}


def bury_all(V, T, reg, parts, sk, cfg):
    """Make every open rim (leg tops, tail root, ear bases, B's neck end) sit inside its host
    so no hole is visible.  Hosts: torso/neck/head triangles of the spine."""
    spine_ids = [i for i, r in enumerate(reg) if r in ("torso", "neck", "head")]
    head_ids = [i for i, r in enumerate(reg) if r == "head"]
    Ts = T[spine_ids]
    Th = T[head_ids] if cfg["spine"] == "B" else Ts
    tor = sk["torso"]
    for leg in ("leg_FR", "leg_FL", "leg_HR", "leg_HL"):
        rim = parts[leg]["rings"][0]
        c = V[rim].mean(0)
        tt = np.clip((c[0] - tor.c[0, 0]) / (tor.c[-1, 0] - tor.c[0, 0]), 0, 1)
        anc = tor.center(tt).copy()
        anc[1] = 0.4 * c[1]
        st = bury(V, rim, Ts, anc)
        print(f"  bury {leg}: {st} steps", file=sys.stderr)
    print("  bury tail:", bury(V, parts["tail"]["rings"][0], Ts, tor.center(0.3)), file=sys.stderr)
    hc = sk["head"].center(0.35)
    if cfg["spine"] == "A":
        # head ring = last ring of the spine; corners theta = pi/4 (+u+v, cat-left top) and
        # 3pi/4 (-u+v, cat-right top).  Anchor each ear just inside its skull corner.
        hr = parts["spine"]["rings"][-1]
        corner = {"ear_L": V[hr[0]], "ear_R": V[hr[1]]}
        hcen = V[hr].mean(0)
        for e in ("ear_R", "ear_L"):
            anc = corner[e] + 0.45 * (hcen - corner[e])
            print(f"  bury {e}:", bury(V, parts[e]["rings"][0], Th, anc, margin=0.002, shrink=0.99,
                                         slide=0.08, max_steps=40), file=sys.stderr)
        return
    for e in ("ear_R", "ear_L"):
        print(f"  bury {e}:", bury(V, parts[e]["rings"][0], Th, hc, margin=0.002, shrink=0.995, slide=0.05),
              file=sys.stderr)
    if cfg["spine"] == "B":
        bury(V, parts["spine"]["rings"][-1], Th, hc)


DESIGNS["A2"] = dict(DESIGNS["A"], sole_at_landmark=True)


def build(design: str) -> dict:
    cfg = DESIGNS[design]
    sk = load_skeleton()
    B = Builder()
    parts = {}
    parts["spine"] = (build_spine_A if cfg["spine"] == "A" else build_spine_B)(B, sk, cfg)
    for leg, lc in cfg["legs"].items():
        parts[leg] = build_leg(B, sk, leg, sole_at_landmark=cfg.get("sole_at_landmark", False), **lc)
    parts["tail"] = build_tail(B, sk, **cfg["tail"])
    parts["ear_R"] = build_ear(B, sk, "ear_R")
    parts["ear_L"] = build_ear(B, sk, "ear_L")

    V = np.array(B.V)
    T = np.array(B.T, np.int64)
    reg, mat = list(B.reg), list(B.mat)
    bury_all(V, T, reg, parts, sk, cfg)
    nT = len(T)
    if nT != BUDGET:
        raise SystemExit(f"design {design}: {nT} triangles != {BUDGET}  "
                         f"(per region: { {r: reg.count(r) for r in sorted(set(reg))} })")

    # ---- landmark attachment (CONTRACT §12) ----
    lm3 = cproxy.landmarks3d()
    tris_of = lambda regions: [i for i, r in enumerate(reg) if r in regions]  # noqa: E731
    att = {}
    att["ear_tip_R"] = {"vertex": int(parts["ear_R"]["end_tip"])}
    att["ear_tip_L"] = {"vertex": int(parts["ear_L"]["end_tip"])}
    att["tail_tip"] = {"vertex": int(parts["tail"]["end_tip"])}
    head_part = parts["spine"]["head"] if "head" in parts["spine"] else parts["spine"]
    att["nose"] = {"vertex": int(head_part["end_tip"])}
    for nm in ("eye_R", "eye_L", "chin"):
        att[nm] = meshio.attach_point(lm3[nm], V, T, tris_of({"head"}))
    for leg, key in (("leg_FR", "paw_FR"), ("leg_FL", "paw_FL"), ("leg_HR", "paw_HR"), ("leg_HL", "paw_HL")):
        sole = parts[leg]["end_tris"][0]
        att[key] = {"tri": int(sole), "bary": [1 / 3, 1 / 3, 1 / 3]}
    tb = parts["tail"]["bands"]
    att["tail_base"] = meshio.attach_point(lm3["tail_base"], V, T, tb[0])
    att["tail_mid"] = meshio.attach_point(lm3["tail_mid"], V, T, [i for b in tb for i in b])
    att["shoulder_R"] = meshio.attach_point(lm3["shoulder_R"], V, T, tris_of({"torso"}))
    att["hip_R"] = meshio.attach_point(lm3["hip_R"], V, T, tris_of({"torso"}))
    att["hock_HR"] = meshio.attach_point(lm3["hock_HR"], V, T, tris_of({"leg_HR"}))
    lms = meshio.landmarks_from_attach(V, T, att)
    rims = [v for leg in ("leg_FR", "leg_FL", "leg_HR", "leg_HL") for v in parts[leg]["rings"][0]]
    rims += list(parts["tail"]["rings"][0]) + list(parts["ear_R"]["rings"][0]) + list(parts["ear_L"]["rings"][0])
    if cfg["spine"] == "B":
        rims += list(parts["spine"]["rings"][-1])
    counts = {r: reg.count(r) for r in sorted(set(reg))}
    d = meshio.to_dict(V, T, reg, mat, lms, BUDGET, name="cat_100",
                       notes=f"budget-100 design {design}; per-region triangles {json.dumps(counts)}",
                       landmark_attach=att)
    d["buried_rim_vertices"] = [int(v) for v in rims]
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", default="A", choices=sorted(DESIGNS))
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    d = build(a.design)
    assert len(d["triangles"]) == BUDGET
    meshio.save(a.out, d)
    # freeze list for scripts/refine_vertices.py --freeze-verts (open rims stay buried)
    a.out.with_suffix(".freeze.json").write_text(json.dumps(d["buried_rim_vertices"]))
    reg = d["tri_region"]
    print(f"wrote {a.out}  triangles={len(d['triangles'])}  vertices={len(d['vertices'])}")
    print({r: reg.count(r) for r in sorted(set(reg))})


if __name__ == "__main__":
    main()
