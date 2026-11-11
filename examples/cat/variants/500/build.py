#!/usr/bin/env python
"""Budget-500 cat: one closed manifold built from hand-allocated ring topology.

    python3 variants/500/build.py --config A --out variants/500/checkpoints/v0_initial/mesh.json

Construction (deterministic, CPU only):
  * main tube  rear apex -> rump dome D -> torso rings R0..R5 -> chest bend C -> neck N0, N1
    (n_body sides, lateral axis +Y, dorsal axis rotates from +Z (torso) to -X (neck)).
  * legs       6-sided tubes welded into 1-column x 2-band holes cut in the lower flanks of
               the torso (true manifold, the limb loop flows into shoulder / haunch);
               paws end in a flat sole quad (z = 0) and a fan-capped toe ring.
  * tail       6-sided tube welded into a hole on the top of the rump, apex tip.
  * head       n_head-sided tube along the proxy head axis (turned towards the camera),
               apex back cap and apex muzzle cap; the neck's last ring is bridged to a
               3-column hole on the head underside; ears are thin 4-sided cones welded into
               single-quad holes (base quad -> narrower mid ring -> apex) with a concave
               inner face.
  * Ring points start on the skeleton superellipses and are pushed radially onto the
    proxy surface (SDF ray march) where the ray finds it within [0.6, 1.3] x radius.
Every triangle gets a CONTRACT region and material; landmarks are attached per §12.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # the case directory (examples/cat)
import os as _os  # noqa: E402
_os.environ["LOWPOLY_CASE"] = str(ROOT)  # catlib.paths reads the case from here
sys.path.insert(0, _os.environ.get("LOWPOLY_SCRIPTS",
                                   str(ROOT.parents[1] / "skills" / "image-to-lowpoly" / "scripts")))
from catlib import meshio, proxy, topo  # noqa: E402
from catlib.skeleton import load_skeleton  # noqa: E402

BUDGET = 500
_LAST_BUILDER = None

CONFIGS = {
    # A: 8-sided torso, 10 body rings, head 8 x 5 rings, legs 6-sided, tail 6 x 6.
    "A": dict(n_body=8, body_phase=22.5,
              body_x=[-0.212, -0.190, -0.150, -0.105, -0.050, 0.020, 0.080],  # D,R0..R5
              neck=True, n_head=8, head_phase=22.5, head_t=[0.0, 0.2, 0.38, 0.58, 0.8],
              head_back="tip", muzzle="tip",
              front_z=[0.075, 0.045, None, None], hind_z=[0.088, None, None, None],
              tail_t=[0.12, 0.32, 0.52, 0.72, 0.9], front_toe="fan", hind_toe="fan",
              neck_spec=[(0.128, 0.190, 40, 0.040, 0.060, "torso"),
                         (0.118, 0.236, 80, 0.039, 0.045, "neck")],
              hind_patch=(2, 4), front_patch=(5, 7), tail_patch=(0, 2)),
    # C: A with a modelled face patch replacing the last head band + muzzle cone.
    "C": None,
    # B: 6-sided torso with more rings following the back/belly line, head 10-sided,
    #    lean tail (6 x 5) and fewer leg rings.
    "B": dict(n_body=6, body_phase=0.0,
              body_x=[-0.212, -0.190, -0.160, -0.120, -0.075, -0.025, 0.025, 0.080],
              neck=True, n_head=10, head_phase=0.0, head_t=[0.0, 0.2, 0.4, 0.6, 0.8],
              head_back="tip", muzzle="tip",
              front_z=[0.075, 0.045, None, None], hind_z=[0.088, None, None, None],
              tail_t=[0.12, 0.32, 0.52, 0.72, 0.9], front_toe="tip", hind_toe="fan",
              neck_spec=[(0.128, 0.190, 40, 0.040, 0.060, "torso"),
                         (0.118, 0.236, 80, 0.039, 0.045, "neck")],
              hind_patch=(2, 4), front_patch=(6, 8), tail_patch=(0, 2)),
}


CONFIGS["C"] = dict(CONFIGS["A"], face="patch", head_t=[0.0, 0.2, 0.38, 0.56])
# D: C + nose-pad triangle (no forehead vertex), face interior frozen during refinement,
#    gentler chest->neck bend and the tail's first ring further from the rump (fewer folds).
CONFIGS["D"] = dict(CONFIGS["C"], rump="fan", tail_t=[0.16, 0.34, 0.52, 0.7, 0.88],
                    neck_spec=[(0.126, 0.195, 35, 0.040, 0.058, "torso"),
                               (0.121, 0.246, 75, 0.039, 0.044, "neck")])


# E (round 2): body prepared for the shared head kit.  Thick 3-ring neck (C, N0, N1) ending
#    in an open 8-ring where neck meets skull; legs re-budgeted (front: no wrist ring; far
#    legs FL/HL 5-sided below the first ring); round 6-sided tail with a root ring and a
#    near-tip ring.  head="own" welds the round-1 head on for testing, head="none" leaves N1
#    open for catlib.headkit.
CONFIGS["E"] = dict(CONFIGS["D"],
                    neck_spec=[(0.128, 0.195, 30, 0.042, 0.060, "torso"),
                               (0.131, 0.222, 48, 0.043, 0.050, "neck"),
                               (0.126, 0.244, 70, 0.042, 0.046, "neck")],
                    tail_root_lift=0.012, tail_patch=(0, 1),
                    legs={"leg_FR": dict(z=[0.075], rings=["z0", 3, 4], sides=[6, 6, 6]),
                          "leg_FL": dict(z=[0.075], rings=["z0", 3, 4], sides=[6, 5, 5]),
                          "leg_HR": dict(z=[0.088], rings=["z0", 2, 4, 5], sides=[6, 6, 6, 6]),
                          "leg_HL": dict(z=[0.088], rings=["z0", 2, 4, 5], sides=[6, 5, 5, 5])},
                    tail_t=[0.05, 0.24, 0.44, 0.64, 0.82, 0.95], tail_tip_depth=0.5,
                    head="own")
CONFIGS["E_body"] = dict(CONFIGS["E"], head="none")
# F (round 2, final): E body + shared head kit L3 (120 tris).  The neck ring N1 is dropped:
#    N0 is rebuilt 15 mm below the kit's neck_loop, parallel to it, and attach_head bridges
#    N0 -> neck_loop (16 tris, replaces the old N0->N1 band).  FR toe ring 5-sided (-2) so
#    the total is exactly 500.  Lower leg rings are moved in the ground plane so each sole
#    centre projects onto its reference paw landmark; the tail root sits lower/forward.
CONFIGS["F"] = dict(CONFIGS["E"],
                    neck_spec=[(0.122, 0.185, 20, 0.042, 0.060, "torso"),
                               (0.131, 0.222, 48, 0.043, 0.050, "neck")],
                    legs=dict(CONFIGS["E"]["legs"],
                              leg_FR=dict(z=[0.075], rings=["z0", 3, 4], sides=[6, 6, 5])),
                    head="kit", kit="head/head_L3.json", kit_gap=0.03,
                    leg_nudge=True, tail_root_vec=(-1.0, 0.0, -0.8))


# G (round 2, final): like F but the chest ring C is bridged straight to the kit's
#    neck_loop (no N0: with the depth-aligned kit the loop sits right above the withers, an
#    extra ring there folded into the back).  The 16 triangles saved go back into the near
#    front leg (wrist ring restored, 6-sided toe: +14) and a 6-sided HL toe (+2).
CONFIGS["G"] = dict(CONFIGS["F"], kit_direct=True,
                    neck_spec=[(0.118, 0.200, 35, 0.042, 0.058, "torso")],
                    legs=dict(CONFIGS["E"]["legs"],
                              leg_FR=dict(z=[0.075, 0.045], rings=["z0", "z1", 3, 4],
                                          sides=[6, 6, 6, 6]),
                              leg_HL=dict(z=[0.088], rings=["z0", 2, 4, 5], sides=[6, 5, 5, 6])))


# ---------------------------------------------------------------------------------------------
class Builder:
    def __init__(self):
        self.V: list[np.ndarray] = []
        self.T: list[list[int]] = []
        self.reg: list[str] = []
        self.mat: list[str] = []
        self.stages: list[tuple] = []

    def snap(self, label):
        self.stages.append((label, [np.asarray(v, float).tolist() for v in self.V],
                            [list(t) for t in self.T], list(self.reg)))

    def add(self, pts) -> list[int]:
        pts = np.atleast_2d(np.asarray(pts, float))
        ids = list(range(len(self.V), len(self.V) + len(pts)))
        self.V.extend(pts)
        return ids

    def tri(self, a, b, c, reg, mat):
        self.T.append([int(a), int(b), int(c)])
        self.reg.append(reg)
        self.mat.append(mat)
        return len(self.T) - 1

    def quad(self, a0, a1, b1, b0, reg, mat):
        V = self.V
        if np.linalg.norm(V[a0] - V[b1]) <= np.linalg.norm(V[a1] - V[b0]):
            return [self.tri(a0, a1, b1, reg, mat), self.tri(a0, b1, b0, reg, mat)]
        return [self.tri(a0, a1, b0, reg, mat), self.tri(a1, b1, b0, reg, mat)]

    def align(self, A, B):
        V = np.asarray(self.V)
        best = None
        for rev in (False, True):
            b = B[::-1] if rev else B
            for s in range(len(b)):
                bb = list(np.roll(b, -s))
                ia = np.linspace(0, len(A), 24, endpoint=False).astype(int) % len(A)
                ib = np.linspace(0, len(bb), 24, endpoint=False).astype(int) % len(bb)
                cost = np.linalg.norm(V[np.asarray(A)[ia]] - V[np.asarray(bb)[ib]], axis=1).sum()
                if best is None or cost < best[0]:
                    best = (cost, bb)
        return best[1]

    def bridge(self, A, B, reg, mat, align=True):
        """Band between loops A and B (equal: quads; unequal: zipper)."""
        if align:
            B = self.align(A, B)
        n, m = len(A), len(B)
        out = []
        if n == m:
            for i in range(n):
                out += self.quad(A[i], A[(i + 1) % n], B[(i + 1) % n], B[i], reg, mat)
            return out
        V = self.V
        i = j = 0
        while i < n or j < m:
            ai, ai1 = A[i % n], A[(i + 1) % n]
            bj, bj1 = B[j % m], B[(j + 1) % m]
            if i >= n:
                adv = False
            elif j >= m:
                adv = True
            else:
                adv = np.linalg.norm(V[ai1] - V[bj]) <= np.linalg.norm(V[ai] - V[bj1])
            if adv:
                out.append(self.tri(ai, ai1, bj, reg, mat)); i += 1
            else:
                out.append(self.tri(ai, bj1, bj, reg, mat)); j += 1
        return out

    def cone(self, ring, apex_pos, reg, mat):
        a = self.add(apex_pos)[0]
        n = len(ring)
        return a, [self.tri(ring[i], ring[(i + 1) % n], a, reg, mat) for i in range(n)]

    def fan(self, ring, reg, mat):
        n = len(ring)
        return [self.tri(ring[0], ring[k], ring[k + 1], reg, mat) for k in range(1, n - 1)]


# ---------------------------------------------------------------------------------------------
def superellipse(c, u, v, ru, rv, p, n, phase_deg):
    th = np.radians(phase_deg) + 2 * np.pi * np.arange(n) / n
    cs, sn = np.cos(th), np.sin(th)
    e = 2.0 / p
    a = np.sign(cs) * np.abs(cs) ** e
    b = np.sign(sn) * np.abs(sn) ** e
    return c + (ru * a)[:, None] * u + (rv * b)[:, None] * v


def project_radial(c, pts, lo=0.6, hi=1.3, steps=60):
    """Move each point along the ray from c to the proxy surface if the crossing lies in
    [lo, hi] x its current radius; otherwise keep it."""
    out = pts.copy()
    for k, q in enumerate(pts):
        d = q - c
        r = np.linalg.norm(d)
        if r < 1e-6:
            continue
        s = np.linspace(lo, hi, steps)
        P = c + s[:, None] * d
        f = proxy.sdf_lookup(P)
        idx = np.flatnonzero((f[:-1] < 0) & (f[1:] >= 0))
        if len(idx):
            i = idx[0]
            t = s[i] + (s[i + 1] - s[i]) * (-f[i]) / (f[i + 1] - f[i])
            out[k] = c + t * d
    return out


def part_t_at_z(part, z):
    ts = np.linspace(0, 1, 801)
    zs = part.center(ts)[:, 2]
    return float(ts[np.argmin(np.abs(zs - z))])


def part_t_at_x(part, x):
    ts = np.linspace(0, 1, 801)
    xs = part.center(ts)[:, 0]
    return float(ts[np.argmin(np.abs(xs - x))])


def nearest_quad_cols(ring_a, ring_b, Vn, target):
    """Column k (vertex k..k+1) between two rings whose quad centroid is nearest target."""
    n = len(ring_a)
    best = None
    for k in range(n):
        q = Vn[[ring_a[k], ring_a[(k + 1) % n], ring_b[k], ring_b[(k + 1) % n]]].mean(0)
        dd = np.linalg.norm(q - target)
        if best is None or dd < best[0]:
            best = (dd, k)
    return best[1]


def col_by_angle(n, phase, target_deg):
    mids = (phase + 360.0 / n * (np.arange(n) + 0.5)) % 360
    diff = np.abs((mids - target_deg + 180) % 360 - 180)
    return int(np.argmin(diff))


def patch_loop(grid, rows, k0, ncols, n):
    """Boundary loop of the patch rows[0]..rows[-1] (ring indices into grid) x columns
    k0..k0+ncols (vertex indices mod n).  Returned as an ordered vertex list."""
    r0, r1 = rows[0], rows[-1]
    ks = [(k0 + j) % n for j in range(ncols + 1)]
    loop = [grid[r0][k] for k in ks]
    loop += [grid[r][ks[-1]] for r in range(r0 + 1, r1 + 1)]
    loop += [grid[r1][k] for k in ks[-2::-1]]
    loop += [grid[r][ks[0]] for r in range(r1 - 1, r0, -1)]
    return loop


def face_patch(B, rim, head, lm3, cfg):
    """Close the head's front ring with a modelled face: forehead vertex, one eye
    triangle per eye (3 own vertices, landmark at its centroid), a protruding nose tip with
    the nose pad above it, and a mouth/chin vertex.  Triangulated by 2D Delaunay in the
    face plane (rim is convex there), so every face triangle is well shaped."""
    from scipy.spatial import Delaunay
    f = head.tangent(0.6)
    lat = np.cross([0, 0, 1.0], f); lat /= np.linalg.norm(lat)        # head's left
    up = np.cross(f, lat)
    Vn = np.asarray(B.V)
    eyes = {s: lm3[f"eye_{s}"] for s in ("R", "L")}
    pts, kinds = [], []
    ew, eh = cfg.get("eye_w", 0.0075), cfg.get("eye_h", 0.0055)
    for s in ("R", "L"):
        c = eyes[s]
        sgn = -1 if s == "R" else 1
        for dl, du in ((-sgn * ew * 0.9, eh * 0.55), (sgn * ew * 1.0, eh * 0.45), (0.0, -eh)):
            pts.append(c + dl * lat + du * up + 0.001 * f); kinds.append(f"eye_{s}")
    nose = lm3["nose"]
    nw, nh_ = cfg.get("nose_w", 0.0075), cfg.get("nose_h", 0.0045)
    for dl, du, df in ((-nw, nh_ * 0.5, -0.002), (nw, nh_ * 0.5, -0.002), (0.0, -nh_, 0.002)):
        pts.append(nose + dl * lat + du * up + df * f); kinds.append("nose_tip")
    mouth = 0.5 * (lm3["nose"] + lm3["chin"]) + f * 0.0
    pts.append(mouth); kinds.append("mouth")
    P = np.asarray(pts)
    # snap non-eye interior points onto the proxy surface
    for i, k in enumerate(kinds):
        if k == "mouth":
            q, _ = proxy.closest_surface(P[i:i + 1])
            P[i] = q[0]
    ids = B.add(P)
    allv = list(rim) + ids
    X = np.asarray(B.V)[allv]
    o = X.mean(0)
    xy = np.stack([(X - o) @ lat, (X - o) @ up], 1)
    tri = Delaunay(xy).simplices
    eye_t = {"R": [], "L": []}
    nose_t = []
    kind_of = {v: k for v, k in zip(ids, kinds)}
    for t in tri:
        vs = [allv[j] for j in t]
        ks = [kind_of.get(v, "rim") for v in vs]
        reg, mat = "head", "fur_orange"
        for s in ("R", "L"):
            if all(k == f"eye_{s}" for k in ks):
                reg, mat = "eyes", "eye"
        if all(k == "nose_tip" for k in ks):
            reg, mat = "nose", "nose"
        elif "nose_tip" in ks or "mouth" in ks:
            mat = "fur_cream"
        ti = B.tri(*vs, reg, mat)
        for s in ("R", "L"):
            if mat == "eye" and ks[0] == f"eye_{s}":
                eye_t[s].append(ti)
        if mat == "nose":
            nose_t.append(ti)
    B.face_ids = ids
    return eye_t, nose_t


# ---------------------------------------------------------------------------------------------
def build(cfg_name: str) -> dict:
    cfg = CONFIGS[cfg_name]
    sk = load_skeleton()
    lm3 = proxy.landmarks3d()
    global _LAST_BUILDER
    B = Builder()
    _LAST_BUILDER = B
    nb, ph = cfg["n_body"], cfg["body_phase"]
    Y = np.array([0.0, 1.0, 0.0])

    # ---------------- main body rings ----------------
    torso = sk["torso"]
    rings_spec = []          # (c, u, v, ru, rv, p, region)
    xs = cfg["body_x"]
    for i, x in enumerate(xs):
        t = part_t_at_x(torso, max(x, torso.center(0.0)[0]))
        c = torso.center(t).copy()
        ru, rv, p = torso.radii(t)
        if x < torso.center(0.0)[0]:                 # rump dome behind the first ring
            f = 1.0 - (torso.center(0.0)[0] - x) / torso.end_cap_depth("start")
            f = max(f, 0.35)
            s = np.sqrt(1 - (1 - f) ** 2) if f < 1 else 1.0
            c = c + np.array([x - c[0], 0, 0.004])
            ru, rv = ru * max(s, 0.55), rv * max(s, 0.55)
        rings_spec.append((c, Y, np.array([0, 0, 1.0]), ru, rv, p, "torso"))
    if cfg["neck"]:
        for (cx, cz, a, ru, rv, reg) in cfg["neck_spec"]:
            ar = np.radians(a)
            v = np.array([-np.sin(ar), 0, np.cos(ar)])
            rings_spec.append((np.array([cx, 0.0, cz]), Y, v, ru, rv, 2.2, reg))
    grid = []
    for j, (c, u, v, ru, rv, p, reg) in enumerate(rings_spec):
        if cfg.get("head") == "kit" and not cfg.get("kit_direct") and j == len(rings_spec) - 1:
            c, u, v, ru, rv = kit_neck_frame(cfg)
            grid.append(B.add(superellipse(c, u, v, ru, rv, 2.0, nb, ph)))
            continue
        pts = superellipse(c, u, v, ru, rv, p, nb, ph)
        pts = project_radial(c, pts)
        grid.append(B.add(pts))
    nrows = len(grid)
    reg_row = [s[6] for s in rings_spec]

    # holes: (rows (start,end), column)
    col_R = col_by_angle(nb, ph, 225.0)
    col_L = col_by_angle(nb, ph, 315.0)
    col_T = col_by_angle(nb, ph, 90.0)
    hp, fp, tp = cfg["hind_patch"], cfg["front_patch"], cfg["tail_patch"]
    holes = {"leg_HR": (hp, col_R), "leg_HL": (hp, col_L), "leg_FR": (fp, col_R),
             "leg_FL": (fp, col_L), "tail": (tp, col_T)}
    skip = set()
    for (r0, r1), k in holes.values():
        for r in range(r0, r1):
            skip.add((r, k))
    for r in range(nrows - 1):
        reg = "neck" if reg_row[r + 1] == "neck" else "torso"
        for k in range(nb):
            if (r, k) in skip:
                continue
            a, b = grid[r], grid[r + 1]
            B.quad(a[k], a[(k + 1) % nb], b[(k + 1) % nb], b[k], reg, "fur_orange")
    B.snap("torso + neck rings")
    # rump apex
    c0 = np.asarray(B.V)[grid[0]].mean(0)
    rear = c0 + np.array([-0.014, 0, 0.0])
    rear = project_radial(c0, rear[None], 0.5, 3.0)[0]
    if cfg.get("rump", "tip") == "tip":
        B.cone(grid[0][::-1], rear, "torso", "fur_orange")
    else:                                            # small flat octagon on the rump dome
        B.fan(grid[0][::-1], "torso", "fur_orange")

    # ---------------- legs ----------------
    sole_quads = {}
    leg_rings = {}
    leg_loops = {}
    for leg in ("leg_FR", "leg_FL", "leg_HR", "leg_HL"):
        part = sk[leg]
        (r0, r1), k = holes[leg]
        loop = patch_loop(grid, list(range(r0, r1 + 1)), k, 1, nb)
        leg_loops[leg] = loop
        tr = part.t_of_ring()
        zs = cfg["front_z"] if leg[4] == "F" else cfg["hind_z"]
        if "legs" in cfg:
            ls = cfg["legs"][leg]
            ts = [part_t_at_z(part, ls["z"][int(r[1:])]) if isinstance(r, str) else tr[r]
                  for r in ls["rings"]]
            sides = ls["sides"]
        elif leg[4] == "F":
            ts = [part_t_at_z(part, zs[0]), part_t_at_z(part, zs[1]), tr[3], tr[4]]
            sides = [6] * 4
        else:
            ts = [part_t_at_z(part, zs[0]), tr[2], tr[4], tr[5]]
            sides = [6] * 4
        prev = loop
        rings = []
        for j, t in enumerate(ts):
            pts = part.ring(t, sides[j], 0.0)
            rid = B.add(pts)
            B.bridge(prev, rid, leg, "fur_orange" if j < 2 else "fur_cream")
            prev = rid
            rings.append(rid)
        # flat sole: two lowest vertices of the last two rings to z = 0
        Vn = np.asarray(B.V)
        sole = []
        for rid in rings[-2:]:
            low = sorted(rid, key=lambda i: Vn[i][2])[:2]
            for i in low:
                B.V[i] = np.array([B.V[i][0], B.V[i][1], 0.0])
            sole += low
        if cfg["front_toe" if leg[4] == "F" else "hind_toe"] == "tip":
            Vn = np.asarray(B.V)
            cc = Vn[rings[-1]].mean(0)
            nrm = part.tangent(1.0)
            tipp = cc + nrm * 0.006
            tipp[2] = max(tipp[2], 0.004)
            B.cone(rings[-1], tipp, leg, "fur_cream")
        else:
            B.fan(rings[-1], leg, "fur_cream")
        if cfg.get("leg_nudge"):
            nudge_leg(B, leg, rings, sole)
        sole_quads[leg] = sole
        leg_rings[leg] = rings
        B.snap(f"{leg}")

    # ---------------- tail ----------------
    tail = sk["tail"]
    (r0, r1), k = holes["tail"]
    loop = patch_loop(grid, list(range(r0, r1 + 1)), k, 1, nb)
    prev = loop
    tail_rings = []
    lift = cfg.get("tail_root_lift")
    hole_c = np.asarray(B.V)[loop].mean(0)
    for j, t in enumerate(cfg["tail_t"]):
        pts = tail.ring(t, 6, 0.0)
        if lift and t < 0.2:
            # root ring sits just outside the rump hole (up and back), so the root band
            # leaves the body instead of dipping into it
            vec = np.asarray(cfg.get("tail_root_vec", (-1.0, 0.0, -0.6)), float)
            pts = pts - tail.center(t) + hole_c + vec * lift
        rid = B.add(pts)
        B.bridge(prev, rid, "tail", "fur_stripe" if j % 2 else "fur_orange")
        prev = rid
        tail_rings.append(rid)
    tip = tail.center(1.0) + tail.tangent(1.0) * tail.end_cap_depth("end") * cfg.get("tail_tip_depth", 0.8)
    tail_tip, _ = B.cone(prev, tip, "tail", "fur_stripe")
    B.snap("tail")
    B.groups = {"tail_rings": [list(map(int, r)) for r in tail_rings], "tail_tip": int(tail_tip),
                "neck_ring": [int(i) for i in grid[-1]],
                "neck_rings": [[int(i) for i in g_] for g_ in grid[-3:]],
                "leg_loops": {k: [int(i) for i in v] for k, v in leg_loops.items()},
                "leg_first_rings": {k: [int(i) for i in v[0]] for k, v in leg_rings.items()},
                "tail_root_loop": [int(i) for i in patch_loop(grid, list(range(r0, r1 + 1)), k, 1, nb)]}
    if cfg.get("head", "own") in ("none", "kit"):
        d = finish_body(B, cfg, cfg_name, lm3, sole_quads, tail_rings, tail_tip, grid)
        if cfg["head"] == "kit":
            from catlib import headkit
            d, frozen = headkit.attach_head(d, str(ROOT / cfg["kit"]), [int(i) for i in grid[-1]])
            d.pop("freeze_vertices", None)
            hk = d.pop("head_kit")
            d["_freeze"] = [int(i) for i in frozen]
            B.groups["kit_neck_loop"] = [int(i) for i in hk["neck_loop"]]
            B.groups["kit_vertex_offset"] = int(hk["vertex_offset"])
            B.stages.append(("head kit L3 weld", [list(map(float, v)) for v in d["vertices"]],
                             [list(map(int, t)) for t in d["triangles"]], list(d["tri_region"])))
            d["notes"] = (f"budget-500 config {cfg_name}: body (8-sided torso, welded legs/tail) + "
                          f"shared head kit L3 bridged from neck ring N0; single closed manifold")
        return d

    # ---------------- head ----------------
    head = sk["head"]
    nh, hph = cfg["n_head"], cfg["head_phase"]
    hgrid = []
    hcent = []
    for t in cfg["head_t"]:
        c = head.center(t)
        u, v, nrm = head.frame(t)
        ru, rv, p = head.radii(t)
        pts = superellipse(c, u, v, ru, rv, p, nh, hph)
        pts = project_radial(c, pts, 0.5, 1.4)
        hgrid.append(B.add(pts))
        hcent.append(c)
    Vn = np.asarray(B.V)
    # neck hole: 3 bottom columns between rings 0 and 1
    kb = col_by_angle(nh, hph, 270.0)
    neck_cols = [(kb - 1) % nh, kb, (kb + 1) % nh]
    # ears: one quad each, nearest to the proxy ear base
    ear_q = {}
    for side in ("R", "L"):
        base = sk[f"ear_{side}"].center(0.0)
        best = None
        for r in range(len(hgrid) - 1):
            kk = nearest_quad_cols(hgrid[r], hgrid[r + 1], Vn, base)
            q = Vn[[hgrid[r][kk], hgrid[r][(kk + 1) % nh], hgrid[r + 1][kk],
                    hgrid[r + 1][(kk + 1) % nh]]].mean(0)
            dd = np.linalg.norm(q - base)
            if best is None or dd < best[0]:
                best = (dd, r, kk)
        ear_q[side] = best[1:]
    face = cfg.get("face", "cone")
    eye_q = {}
    if face == "cone":
        # eyes: quad nearest the proxy eye landmark
        for side in ("R", "L"):
            best = None
            for r in range(len(hgrid) - 1):
                kk = nearest_quad_cols(hgrid[r], hgrid[r + 1], Vn, lm3[f"eye_{side}"])
                q = Vn[[hgrid[r][kk], hgrid[r][(kk + 1) % nh], hgrid[r + 1][kk],
                        hgrid[r + 1][(kk + 1) % nh]]].mean(0)
                dd = np.linalg.norm(q - lm3[f"eye_{side}"])
                if best is None or dd < best[0]:
                    best = (dd, r, kk)
            eye_q[side] = best[1:]
    else:
        eye_q = {"R": None, "L": None}
    hskip = {(0, k) for k in neck_cols} | {ear_q["R"], ear_q["L"]}
    eye_tris = {"R": [], "L": []}
    for r in range(len(hgrid) - 1):
        for k in range(nh):
            if (r, k) in hskip:
                continue
            a, b = hgrid[r], hgrid[r + 1]
            mat = "fur_orange"
            reg = "head"
            side = [s for s in ("R", "L") if eye_q[s] == (r, k)]
            if side:
                mat, reg = "eye", "eyes"
            ids = B.quad(a[k], a[(k + 1) % nh], b[(k + 1) % nh], b[k], reg, mat)
            if side:
                eye_tris[side[0]] = ids
    back = head.center(0.0) - head.tangent(0.0) * head.end_cap_depth("start")
    back = project_radial(head.center(0.1), back[None], 0.5, 2.0)[0]
    B.cone(hgrid[0][::-1], back, "head", "fur_orange")
    if face == "cone":
        front = head.center(1.0) + head.tangent(1.0) * head.end_cap_depth("end")
        front = project_radial(head.center(0.85), front[None], 0.5, 2.5)[0]
        _, muzzle_tris = B.cone(hgrid[-1], front, "head", "fur_cream")
        Vn = np.asarray(B.V)
        cen = [Vn[B.T[i]].mean(0) for i in muzzle_tris]
        order = np.argsort([np.linalg.norm(c - lm3["nose"]) for c in cen])
        nose_tris = [muzzle_tris[i] for i in order[:2]]
        for i in nose_tris:
            B.mat[i], B.reg[i] = "nose", "nose"
    else:
        eye_tris, nose_tris = face_patch(B, hgrid[-1], head, lm3, cfg)
    B.snap("head + face")
    # neck -> head bridge
    hl = patch_loop(hgrid, [0, 1], neck_cols[0], 3, nh)
    B.bridge(grid[-1], hl, "neck", "fur_cream")
    B.groups["head_neck_loop"] = [int(i) for i in hl]
    B.snap("neck weld")
    # ears
    ear_apex = {}
    for side in ("R", "L"):
        r, kk = ear_q[side]
        loop = patch_loop(hgrid, [r, r + 1], kk, 1, nh)
        Vn = np.asarray(B.V)
        base_c = Vn[loop].mean(0)
        apex = lm3[f"ear_tip_{side}"]
        hf = head.tangent(0.5)                       # head forward
        mid = []
        for i in loop:
            q = base_c + 0.55 * (Vn[i] - base_c) + 0.5 * (apex - base_c)
            if np.dot(Vn[i] - base_c, hf) > 0:       # front (inner) side -> cup
                q = q - hf * 0.004
            mid.append(q)
        mid_ids = B.add(np.asarray(mid))
        tris = B.bridge(loop, mid_ids, "ears", "fur_orange", align=False)
        a, tt = B.cone(mid_ids, apex, "ears", "fur_orange")
        ear_apex[side] = a
        Vn = np.asarray(B.V)
        for i in tris + tt:
            n_ = np.cross(Vn[B.T[i][1]] - Vn[B.T[i][0]], Vn[B.T[i][2]] - Vn[B.T[i][0]])
            cc = Vn[B.T[i]].mean(0) - Vn[[*loop, *mid_ids]].mean(0)
            if np.dot(cc, hf) > 0.0005:
                B.mat[i] = "ear_inner"

    B.snap("ears")
    # ---------------- assemble + materials ----------------
    V = np.asarray(B.V)
    T = np.asarray(B.T)
    d = meshio.to_dict(V, T, B.reg, B.mat, {k: v for k, v in lm3.items()}, BUDGET,
                       notes=f"budget-500 config {cfg_name}")
    d = topo.orient_outward(d)
    V = np.asarray(d["vertices"]); T = np.asarray(d["triangles"])
    cen = V[T].mean(1)
    for i, (r, m) in enumerate(zip(d["tri_region"], d["tri_material"])):
        if r == "torso" and cen[i][2] < 0.125 and cen[i][0] > 0.03:
            d["tri_material"][i] = "fur_cream"          # chest underside
        if r == "head" and m == "fur_orange":
            hf = head.tangent(0.5)
            rel = cen[i] - head.center(0.5)
            if np.dot(rel, hf) > 0.012 and rel[2] < -0.004:
                d["tri_material"][i] = "fur_cream"      # muzzle / chin
    # landmarks (§12)
    T = np.asarray(d["triangles"])
    att = {}
    att["ear_tip_R"] = {"vertex": ear_apex["R"]}
    att["ear_tip_L"] = {"vertex": ear_apex["L"]}
    att["tail_tip"] = {"vertex": tail_tip}
    for side in ("R", "L"):
        ids = eye_tris[side]
        q = V[T[ids]].reshape(-1, 3).mean(0)
        att[f"eye_{side}"] = meshio.attach_point(q, V, T, ids)
    q = V[T[nose_tris]].reshape(-1, 3).mean(0)
    att["nose"] = meshio.attach_point(q, V, T, nose_tris)
    if cfg.get("legs"):                       # round-2 configs: shared body attachments
        att.update(_body_attach(d, V, T, lm3, sole_quads, tail_rings, tail_tip))
    for leg in ("FR", "FL", "HR", "HL"):
        if cfg.get("legs"):
            break
        q = V[sole_quads[f"leg_{leg}"]].mean(0)
        cand = [i for i in range(len(T)) if d["tri_region"][i] == f"leg_{leg}"]
        att[f"paw_{leg}"] = meshio.attach_point(q, V, T, cand)
    tail_ids = [i for i in range(len(T)) if d["tri_region"][i] == "tail"]
    q = V[tail_rings[0]].mean(0)
    # tail_base: the first tail ring, on the camera-facing (-Y) side
    if not cfg.get("legs"):
        att["tail_base"] = meshio.attach_point(q + np.array([0, -0.03, 0]), V, T, tail_ids)
    att["tail_mid"] = meshio.attach_point(lm3["tail_mid"], V, T, tail_ids)
    torso_ids = [i for i in range(len(T)) if d["tri_region"][i] in ("torso", "leg_FR", "leg_HR")]
    att["shoulder_R"] = meshio.attach_point(lm3["shoulder_R"], V, T, torso_ids)
    att["hip_R"] = meshio.attach_point(lm3["hip_R"], V, T, torso_ids)
    hr = [i for i in range(len(T)) if d["tri_region"][i] == "leg_HR"]
    att["hock_HR"] = meshio.attach_point(lm3["hock_HR"], V, T, hr)
    hd = [i for i in range(len(T)) if d["tri_region"][i] == "head"]
    att["chin"] = meshio.attach_point(lm3["chin"], V, T, hd)
    d["landmark_attach"] = att
    meshio.sync_landmarks(d)
    if getattr(B, "face_ids", None):
        d["_freeze"] = [int(i) for i in B.face_ids]
    d["notes"] = (f"budget-500 config {cfg_name}: single closed manifold; main tube {nb}-sided, "
                  f"head {nh}-sided, legs/tail 6-sided welded into torso holes")
    return d


def kit_neck_frame(cfg):
    """Ring frame `kit_gap` below the head kit's neck_loop, parallel to it: centre, u (cat's
    left), v (dorsal, towards the nape), ru, rv (loop half-extents x 1.05)."""
    h = json.loads((ROOT / cfg["kit"]).read_text())
    L = np.asarray(h["vertices"], float)[h["neck_loop"]]
    c = L.mean(0)
    n = np.linalg.svd(L - c)[2][2]
    if n[2] < 0:
        n = -n
    nape = L[4]                                   # loop order: throat, right, nape, left
    v = (nape - c) - ((nape - c) @ n) * n
    v /= np.linalg.norm(v)
    u = np.cross(v, n)
    ru = np.abs((L - c) @ u).max() * 1.05
    rv = np.abs((L - c) @ v).max() * 1.05
    return c - cfg["kit_gap"] * n, u, v, ru, rv


def nudge_leg(B, leg, rings, sole):
    """Translate the lower rings of a leg in the ground plane so its sole centre projects
    onto the reference paw landmark (ray through the reference pixel intersected with
    z = 0).  Last two rings move fully, the ring above by half, capped at 15 mm."""
    from catlib import metrics, paths
    from catlib.camera import Camera
    cam = Camera.from_json()
    ref = metrics.load_ref_landmarks(paths.DERIVED / "landmarks.json")["paw_" + leg[4:]]["uv"]
    o, d = cam.pixel_rays(np.asarray([ref]))
    tgt = o[0] + d[0] * (-o[0][2] / d[0][2])
    q = np.asarray(B.V)[sole].mean(0)
    delta = np.array([tgt[0] - q[0], tgt[1] - q[1], 0.0])
    nrm = np.linalg.norm(delta)
    if nrm > 0.015:
        delta *= 0.015 / nrm
    w = [0.5] * (len(rings) - 2) + [1.0, 1.0]
    w[0] = 0.0 if len(rings) > 3 else 0.5
    for rid, wt in zip(rings, w):
        for i in rid:
            B.V[i] = np.asarray(B.V[i], float) + wt * delta
    B.nudges = getattr(B, "nudges", {}); B.nudges[leg] = delta.round(4).tolist()


def finish_body(B, cfg, cfg_name, lm3, sole_quads, tail_rings, tail_tip, grid):
    """Body without head: open 8-ring at the neck top (boundary loop for the head kit).
    Head landmarks are left at the proxy positions (unattached) until the kit is welded."""
    V = np.asarray(B.V)
    T = np.asarray(B.T)
    d = meshio.to_dict(V, T, B.reg, B.mat, {k: v for k, v in lm3.items()}, BUDGET)
    d = _orient_open(d, grid[-1])
    V = np.asarray(d["vertices"]); T = np.asarray(d["triangles"])
    cen = V[T].mean(1)
    for i, r in enumerate(d["tri_region"]):
        if r == "torso" and cen[i][2] < 0.125 and cen[i][0] > 0.03:
            d["tri_material"][i] = "fur_cream"
    att = _body_attach(d, V, T, lm3, sole_quads, tail_rings, tail_tip)
    d["landmark_attach"] = att
    meshio.sync_landmarks(d)
    d["notes"] = f"budget-500 config {cfg_name}: BODY ONLY, open neck ring for the head kit"
    return d


def _orient_open(d, ring):
    """orient_outward needs a closed surface for the volume sign: close the neck ring with a
    temporary fan, orient, then drop the fan."""
    n0 = len(d["triangles"])
    R = [int(i) for i in ring]
    tmp = dict(d)
    tmp["triangles"] = d["triangles"] + [[R[0], R[k], R[k + 1]] for k in range(1, len(R) - 1)]
    tmp["tri_region"] = d["tri_region"] + ["neck"] * (len(R) - 2)
    tmp["tri_material"] = d["tri_material"] + ["fur_orange"] * (len(R) - 2)
    tmp.pop("landmark_attach", None)
    o = topo.orient_outward(tmp)
    for k in ("triangles", "tri_region", "tri_material"):
        d[k] = o[k][:n0]
    return d


def _body_attach(d, V, T, lm3, sole_quads, tail_rings, tail_tip):
    att = {"tail_tip": {"vertex": int(tail_tip)}}
    reg = d["tri_region"]
    for leg in ("FR", "FL", "HR", "HL"):
        q = V[sole_quads[f"leg_{leg}"]].mean(0)
        cand = [i for i in range(len(T)) if reg[i] == f"leg_{leg}"]
        att[f"paw_{leg}"] = meshio.attach_point(q, V, T, cand)
    tail_ids = [i for i in range(len(T)) if reg[i] == "tail"]
    # tail_base: on the root band (triangles touching the first tail ring), nearest the
    # proxy's tail/rump junction point
    r0 = [int(i) for i in tail_rings[0]]
    best = None
    for j in range(len(r0)):                      # closest point ON the first ring's polyline
        A, Bp = V[r0[j]], V[r0[(j + 1) % len(r0)]]
        t = np.clip(np.dot(lm3["tail_base"] - A, Bp - A) / np.dot(Bp - A, Bp - A), 0, 1)
        q = A + t * (Bp - A)
        dd = np.linalg.norm(q - lm3["tail_base"])
        if best is None or dd < best[0]:
            best = (dd, q, {r0[j], r0[(j + 1) % len(r0)]})
    edge_tris = [i for i in tail_ids if best[2] <= set(int(x) for x in T[i])]
    att["tail_base"] = meshio.attach_point(best[1], V, T, edge_tris)
    att["tail_mid"] = meshio.attach_point(lm3["tail_mid"], V, T, tail_ids)
    torso_ids = [i for i in range(len(T)) if reg[i] in ("torso", "leg_FR", "leg_HR")]
    att["shoulder_R"] = meshio.attach_point(lm3["shoulder_R"], V, T, torso_ids)
    att["hip_R"] = meshio.attach_point(lm3["hip_R"], V, T, torso_ids)
    hr = [i for i in range(len(T)) if reg[i] == "leg_HR"]
    att["hock_HR"] = meshio.attach_point(lm3["hock_HR"], V, T, hr)
    return att


def region_counts(d):
    out = {}
    for r in d["tri_region"]:
        out[r] = out.get(r, 0) + 1
    return dict(sorted(out.items()))


def topology_report(d):
    T = np.asarray(d["triangles"])
    from collections import Counter
    e = Counter()
    for t in T:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            e[(min(a, b), max(a, b))] += 1
    return {"boundary_edges": sum(1 for v in e.values() if v == 1),
            "nonmanifold_edges": sum(1 for v in e.values() if v > 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="D", choices=sorted(CONFIGS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-under", action="store_true", help="do not assert == 500 (exploration)")
    ap.add_argument("--stages-dir", help="CONTRACT §13: write construction stages (build order) here")
    a = ap.parse_args()
    d = build(a.config)
    if a.stages_dir:
        sd = Path(a.stages_dir); sd.mkdir(parents=True, exist_ok=True)
        names = []
        for i, (label, V, T, R) in enumerate(_LAST_BUILDER.stages):
            fn = f"{i + 1:02d}_{label.replace(' ', '_').replace('+', 'and')}.json"
            (sd / fn).write_text(json.dumps({"label": label, "vertices": V, "triangles": T,
                                             "tri_region": R}))
            names.append({"label": label, "kind": "construct", "mesh": fn,
                          "triangles": len(T)})
        (sd / "index.json").write_text(json.dumps(names, indent=1))
        print("wrote", len(names), "stages to", sd)
    n = len(d["triangles"])
    print("triangles", n, region_counts(d), topology_report(d),
          "nudges", getattr(_LAST_BUILDER, "nudges", None))
    if not a.allow_under:
        assert n == BUDGET, f"{n} != {BUDGET}"
    if n > BUDGET: print("WARNING over budget (exploration only)")
    fr = d.pop("_freeze", None)
    if CONFIGS[a.config].get("head", "own") == "none":
        errs = [e for e in meshio.validate_dict(d) if not e.startswith("over_budget")]
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(d))          # body-only: not an evaluable catmesh
        print("wrote BODY ONLY", a.out, "schema:", errs or "ok")
    else:
        meshio.save(a.out, d)
        print("wrote", a.out)
    g = getattr(_LAST_BUILDER, "groups", None)
    if g:
        Path(a.out).with_name("groups.json").write_text(json.dumps(g))
    if fr:
        fp = Path(a.out).with_name("freeze_verts.json")
        fp.write_text(json.dumps(fr))
        print("wrote", fp, "(face interior vertices, pass to refine_vertices --freeze-verts)")


if __name__ == "__main__":
    main()
