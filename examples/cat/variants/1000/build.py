#!/usr/bin/env python
"""Budget-1000 cat: designed loft topology on the shared proxy skeleton.

    python3 variants/1000/build.py --design head_rich --out variants/1000/checkpoints/v0_initial/mesh.json

Every part is a closed "loft": a list of cross-section rings (each with its own side
count), consecutive rings bridged (equal counts: quads split on the short diagonal;
unequal counts: shortest-diagonal zipper), both ends closed by a cone to a tip vertex.
A closed loft with ring sizes n_i has exactly 2*sum(n_i) triangles, so the budget is
allocated as ring vertices: sum over all parts of sum(n_i) == 500 -> 1000 triangles.

Ring placement along each part is chosen by --spacing: "curv" puts rings where the
centre line bends or the radius changes (cumulative density 1 + k*change), "uniform"
uses equal arc length.  Parts are separate closed components that interpenetrate
(legs, neck, ears and tail pushed into torso/head); disclosed in NOTES.md.

Deterministic: no randomness.  Asserts the triangle count equals the budget.
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

from catlib import meshio, proxy  # noqa: E402
from catlib.skeleton import load_skeleton  # noqa: E402

BUDGET = 1000

# ---- designs: per part (t0, t1, ring side counts, start-tip depth, end-tip depth) --------
# t0/t1 = arc-length range used for the body rings; tip depth in metres along the end
# tangent (start: backwards).  Ring counts listed from t0 to t1.
DESIGNS = {
    # head-rich: head 106 ring verts (212 tris), torso 120 (240)
    "head_rich": {
        "torso":  dict(t=(0.0, 1.0), n=[12] * 10, tips=(0.035, 0.030)),
        "neck":   dict(t=(0.0, 1.0), n=[8, 8], tips=(0.010, 0.010)),
        "head":   dict(t=(0.0, 1.0), n=[10, 14, 16, 16, 16, 14, 12, 8], tips=(0.020, 0.008)),
        "ear_R":  dict(t=(0.0, 0.88), n=[6, 6, 4], tips=(0.006, None)),
        "ear_L":  dict(t=(0.0, 0.88), n=[6, 6, 4], tips=(0.006, None)),
        "leg_FR": dict(t=(0.12, 1.0), n=[8, 8, 7, 6, 6, 6], tips=(0.010, 0.012)),
        "leg_FL": dict(t=(0.12, 1.0), n=[8, 8, 7, 6, 6, 6], tips=(0.010, 0.012)),
        "leg_HR": dict(t=(0.05, 1.0), n=[8, 8, 8, 6, 6, 6, 6], tips=(0.015, 0.014)),
        "leg_HL": dict(t=(0.05, 1.0), n=[8, 8, 8, 6, 6, 6, 6], tips=(0.015, 0.014)),
        "tail":   dict(t=(0.0, 0.97), n=[6] * 8, tips=(0.010, 0.012)),
    },
    # body-rich: torso 14-sided x 11 rings (154 verts, 308 tris), head 76 (152)
    "body_rich": {
        "torso":  dict(t=(0.0, 1.0), n=[14] * 11, tips=(0.035, 0.030)),
        "neck":   dict(t=(0.0, 1.0), n=[8, 8], tips=(0.010, 0.010)),
        "head":   dict(t=(0.0, 1.0), n=[8, 10, 12, 12, 12, 10, 8, 4], tips=(0.020, 0.008)),
        "ear_R":  dict(t=(0.0, 0.88), n=[5, 5, 4], tips=(0.006, None)),
        "ear_L":  dict(t=(0.0, 0.88), n=[5, 5, 4], tips=(0.006, None)),
        "leg_FR": dict(t=(0.12, 1.0), n=[8, 8, 7, 6, 6, 6], tips=(0.010, 0.012)),
        "leg_FL": dict(t=(0.12, 1.0), n=[8, 8, 7, 6, 6, 6], tips=(0.010, 0.012)),
        "leg_HR": dict(t=(0.05, 1.0), n=[8, 8, 8, 6, 6, 6, 6], tips=(0.015, 0.014)),
        "leg_HL": dict(t=(0.05, 1.0), n=[8, 8, 8, 6, 6, 6, 6], tips=(0.015, 0.014)),
        "tail":   dict(t=(0.0, 0.97), n=[6] * 8, tips=(0.010, 0.012)),
    },
    # v1: head-rich, plus tapered roots so legs/neck flow into the torso instead of
    # ending in caps that poke out of the flank (v0 stumps); torso 9 rings, deeper chest.
    "v1": {
        "torso":  dict(t=(0.0, 1.0), n=[8] + [12] * 6 + [8], tips=(0.035, 0.045)),
        "neck":   dict(t=(0.1, 1.0), n=[10, 10, 10], tips=(0.008, 0.010),
                       root=dict(n=8, scale=0.65, pull=0.5, target=[0.07, 0.0, 0.19])),
        "head":   dict(t=(0.0, 1.0), n=[8, 14, 16, 14, 14, 14, 12, 8], tips=(0.020, 0.008)),
        "ear_R":  dict(t=(0.0, 0.88), n=[5, 5, 4], tips=(0.006, None)),
        "ear_L":  dict(t=(0.0, 0.88), n=[5, 5, 4], tips=(0.006, None)),
        "leg_FR": dict(t=(0.0, 1.0), n=[8, 7, 7, 6, 6, 6], tips=(0.006, 0.012),
                       root=dict(n=6, scale=0.7, pull=0.6, target=None)),
        "leg_FL": dict(t=(0.0, 1.0), n=[8, 7, 7, 6, 6, 6], tips=(0.006, 0.012),
                       root=dict(n=6, scale=0.7, pull=0.6, target=None)),
        "leg_HR": dict(t=(0.0, 1.0), n=[8, 8, 7, 6, 6, 6, 6], tips=(0.006, 0.014),
                       root=dict(n=6, scale=0.7, pull=0.6, target=None)),
        "leg_HL": dict(t=(0.0, 1.0), n=[8, 8, 7, 6, 6, 6, 6], tips=(0.006, 0.014),
                       root=dict(n=6, scale=0.7, pull=0.6, target=None)),
        "tail":   dict(t=(0.03, 0.97), n=[6] * 7, tips=(0.006, 0.012),
                       root=dict(n=6, scale=0.7, pull=0.5, target=[-0.15, 0.0, 0.2])),
    },
    # v2 = v1 with shallower leg roots (scale 0.85, pull 0.4: the first band slopes into
    # the flank instead of forming a shelf) and the neck ending at t=0.85 inside the head
    # (removes the collar ledge under the chin).  Same ring counts as v1.
    "v2": {
        "torso":  dict(t=(0.0, 1.0), n=[8] + [12] * 6 + [8], tips=(0.035, 0.045)),
        "neck":   dict(t=(0.1, 0.85), n=[10, 10, 10], tips=(0.008, 0.004),
                       root=dict(n=8, scale=0.65, pull=0.5, target=[0.07, 0.0, 0.19])),
        "head":   dict(t=(0.0, 1.0), n=[8, 14, 16, 14, 14, 14, 12, 8], tips=(0.020, 0.008)),
        "ear_R":  dict(t=(0.0, 0.88), n=[5, 5, 4], tips=(0.006, None)),
        "ear_L":  dict(t=(0.0, 0.88), n=[5, 5, 4], tips=(0.006, None)),
        "leg_FR": dict(t=(0.0, 1.0), n=[8, 7, 7, 6, 6, 6], tips=(0.006, 0.012),
                       root=dict(n=6, scale=0.85, pull=0.4, target=None)),
        "leg_FL": dict(t=(0.0, 1.0), n=[8, 7, 7, 6, 6, 6], tips=(0.006, 0.012),
                       root=dict(n=6, scale=0.85, pull=0.4, target=None)),
        "leg_HR": dict(t=(0.0, 1.0), n=[8, 8, 7, 6, 6, 6, 6], tips=(0.006, 0.014),
                       root=dict(n=6, scale=0.85, pull=0.4, target=None)),
        "leg_HL": dict(t=(0.0, 1.0), n=[8, 8, 7, 6, 6, 6, 6], tips=(0.006, 0.014),
                       root=dict(n=6, scale=0.85, pull=0.4, target=None)),
        "tail":   dict(t=(0.03, 0.97), n=[6] * 7, tips=(0.006, 0.012),
                       root=dict(n=6, scale=0.7, pull=0.5, target=[-0.15, 0.0, 0.2])),
    },
    # f1: lofts for the fused manifold (variants/1000/fuse.py).  No root rings: parts
    # simply overlap their host and are boolean-unioned; buried geometry is discarded by
    # the union, so the loft total is ABOVE the budget and fuse.py brings it to 1000.
    "f1": {
        "torso":  dict(t=(0.0, 1.0), n=[8, 12, 14, 14, 14, 14, 14, 12, 8], tips=(0.035, 0.045)),
        "neck":   dict(t=(0.0, 0.9), n=[10, 12, 12, 10], tips=(0.006, 0.006)),
        "head":   dict(t=(0.0, 1.0), n=[8, 14, 16, 16, 16, 14, 12, 8], tips=(0.020, 0.008)),
        "ear_R":  dict(t=(0.0, 0.88), n=[6, 6, 4], tips=(0.006, None)),
        "ear_L":  dict(t=(0.0, 0.88), n=[6, 6, 4], tips=(0.006, None)),
        "leg_FR": dict(t=(0.0, 1.0), n=[8, 8, 8, 7, 6, 6, 6], tips=(0.006, 0.012)),
        "leg_FL": dict(t=(0.0, 1.0), n=[8, 8, 8, 7, 6, 6, 6], tips=(0.006, 0.012)),
        "leg_HR": dict(t=(0.0, 1.0), n=[10, 10, 8, 8, 6, 6, 6, 6], tips=(0.006, 0.014)),
        "leg_HL": dict(t=(0.0, 1.0), n=[10, 10, 8, 8, 6, 6, 6, 6], tips=(0.006, 0.014)),
        "tail":   dict(t=(0.0, 0.97), n=[6] * 9, tips=(0.006, 0.012)),
    },
    # f2: f1 with ~20 tris moved off the flat torso flanks (14 -> 12 sides) to the head
    # (a 18-sided ring at the cheeks/whisker pads) and one more ring vertex on each leg
    # at the elbow/hock.
    "f2": {
        "torso":  dict(t=(0.0, 1.0), n=[8, 12, 12, 12, 12, 12, 12, 12, 8], tips=(0.035, 0.045)),
        "neck":   dict(t=(0.0, 0.9), n=[10, 12, 12, 10], tips=(0.006, 0.006)),
        "head":   dict(t=(0.0, 1.0), n=[8, 14, 16, 18, 16, 16, 12, 8], tips=(0.020, 0.008)),
        "ear_R":  dict(t=(0.0, 0.88), n=[6, 6, 4], tips=(0.006, None)),
        "ear_L":  dict(t=(0.0, 0.88), n=[6, 6, 4], tips=(0.006, None)),
        "leg_FR": dict(t=(0.0, 1.0), n=[8, 8, 8, 7, 7, 6, 6], tips=(0.006, 0.012)),
        "leg_FL": dict(t=(0.0, 1.0), n=[8, 8, 8, 7, 7, 6, 6], tips=(0.006, 0.012)),
        "leg_HR": dict(t=(0.0, 1.0), n=[10, 10, 8, 8, 7, 6, 6, 6], tips=(0.006, 0.014)),
        "leg_HL": dict(t=(0.0, 1.0), n=[10, 10, 8, 8, 7, 6, 6, 6], tips=(0.006, 0.014)),
        "tail":   dict(t=(0.0, 0.97), n=[6] * 9, tips=(0.006, 0.012)),
    },
    # f3 (round 3 body): f1 + tail start tip 30 mm deep inside the rump (no notch at the
    # tail root), tail tip rounded by a 0.65-scaled dome ring before the apex.
    "f3": {
        "torso":  dict(t=(0.0, 1.0), n=[8, 12, 14, 14, 14, 14, 14, 12, 8], tips=(0.035, 0.045)),
        "neck":   dict(t=(0.0, 0.9), n=[10, 12, 12, 10], tips=(0.006, 0.006)),
        "head":   dict(t=(0.0, 1.0), n=[8, 14, 16, 16, 16, 14, 12, 8], tips=(0.020, 0.008)),
        "ear_R":  dict(t=(0.0, 0.88), n=[6, 6, 4], tips=(0.006, None)),
        "ear_L":  dict(t=(0.0, 0.88), n=[6, 6, 4], tips=(0.006, None)),
        "leg_FR": dict(t=(0.0, 1.0), n=[8, 8, 8, 7, 6, 6, 6], tips=(0.006, 0.012)),
        "leg_FL": dict(t=(0.0, 1.0), n=[8, 8, 8, 7, 6, 6, 6], tips=(0.006, 0.012)),
        "leg_HR": dict(t=(0.0, 1.0), n=[10, 10, 8, 8, 6, 6, 6, 6], tips=(0.006, 0.014)),
        "leg_HL": dict(t=(0.0, 1.0), n=[10, 10, 8, 8, 6, 6, 6, 6], tips=(0.006, 0.014)),
        "tail":   dict(t=(0.0, 0.95), n=[6] * 9, tips=(0.030, 0.014), dome=(0.65, 0.55)),
    },
}

REGION_OF_PART = {"torso": "torso", "neck": "neck", "head": "head", "ear_R": "ears",
                  "ear_L": "ears", "leg_FR": "leg_FR", "leg_FL": "leg_FL", "leg_HR": "leg_HR",
                  "leg_HL": "leg_HL", "tail": "tail"}


def ring_ts(part, t0, t1, m, spacing, k=3.0):
    if m == 1:
        return np.array([t0])
    if spacing == "uniform":
        return np.linspace(t0, t1, m)
    t = np.linspace(t0, t1, 400)
    c = part.center(t)
    ru, rv, _ = part.radii(t)
    r = 0.5 * (ru + rv)
    tan = np.gradient(c, t, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    bend = np.linalg.norm(np.gradient(tan, t, axis=0), axis=1) * r          # radians * radius
    dr = np.abs(np.gradient(r, t)) / max(part.length, 1e-6)
    ch = bend / max(part.length, 1e-6) + dr
    ch = ch / max(ch.mean(), 1e-9)
    dens = 1.0 + k * ch / (1.0 + ch)
    cum = np.concatenate([[0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(t))])
    cum /= cum[-1]
    return np.interp(np.linspace(0, 1, m), cum, t)


def loft(part, ts, ns, tip_start, tip_end, root=None, dome=None):
    """Rings + 2 tips -> (V, T, ring index list).  Outward CCW winding.
    root = {"n", "scale", "pull", "target"}: an extra first ring, a scaled copy of the
    t0 ring pulled towards `target` (inside the host part), so the part tapers into its
    host instead of ending in a cap that pokes out of the host's surface."""
    rings = [part.ring(float(t), int(n)) for t, n in zip(ts, ns)]
    root_c = None
    if root:
        c0 = part.center(float(ts[0]))
        cr = c0 + root["pull"] * (np.asarray(root["target"], float) - c0)
        R0 = part.ring(float(ts[0]), int(root["n"]))
        rings.insert(0, cr + root["scale"] * (R0 - c0))
        root_c = (cr, cr - c0)
    if dome:   # rounded end: extra ring = last ring scaled by dome[0], moved dome[1]*tip depth
        ce = part.center(float(ts[-1])); te = part.tangent(float(ts[-1]))
        rings.append(ce + te * dome[1] * tip_end + dome[0] * (rings[-1] - ce))
    V, idx, base = [], [], 0
    for R in rings:
        V.append(R)
        idx.append(np.arange(base, base + len(R)))
        base += len(R)
    T = []
    for A, B in zip(idx[:-1], idx[1:]):
        Va = np.concatenate(V)
        n, m = len(A), len(B)
        if n == m:
            for i in range(n):
                a0, a1, b0, b1 = A[i], A[(i + 1) % n], B[i], B[(i + 1) % n]
                if np.linalg.norm(Va[a0] - Va[b1]) <= np.linalg.norm(Va[a1] - Va[b0]):
                    T += [[a0, a1, b1], [a0, b1, b0]]
                else:
                    T += [[a0, a1, b0], [a1, b1, b0]]
        else:   # zipper in normalised angle (both rings start at theta=0)
            i = j = 0
            while i < n or j < m:
                if i >= n:
                    adv_a = False
                elif j >= m:
                    adv_a = True
                else:
                    adv_a = (i + 1) / n <= (j + 1) / m
                if adv_a:
                    T.append([A[i % n], A[(i + 1) % n], B[j % m]]); i += 1
                else:
                    T.append([A[i % n], B[(j + 1) % m], B[j % m]]); j += 1
    Vc = np.concatenate(V)
    tips = []
    for which, t, dep in (("start", ts[0], tip_start), ("end", ts[-1], tip_end)):
        c = part.center(float(t))
        tan = part.tangent(float(t))
        s = -1.0 if which == "start" else 1.0
        if which == "start" and root_c is not None:
            off = root_c[1] / max(np.linalg.norm(root_c[1]), 1e-9)
            tips.append(root_c[0] + off * dep)
        else:
            tips.append(c + s * tan * dep)
    Vc = np.vstack([Vc, tips[0], tips[1]])
    i0, i1 = len(Vc) - 2, len(Vc) - 1
    A, B = idx[0], idx[-1]
    for i in range(len(A)):
        T.append([A[(i + 1) % len(A)], A[i], i0])
    for i in range(len(B)):
        T.append([B[i], B[(i + 1) % len(B)], i1])
    T = np.asarray(T, np.int64)
    # orientation: positive signed volume
    cen = Vc.mean(0)
    vol = ((Vc[T[:, 0]] - cen) * np.cross(Vc[T[:, 1]] - cen, Vc[T[:, 2]] - cen)).sum()
    if vol < 0:
        T = T[:, ::-1].copy()
    return Vc, T, idx, (i0, i1)


def tri_normals(V, T):
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def build(design: str, spacing: str, features: str = "v0") -> dict:
    skel = load_skeleton()
    spec = DESIGNS[design]
    Vs, Ts, reg, part_of_tri, tips_of, base = [], [], [], [], {}, 0
    rings_of = {}
    for name, sp in spec.items():
        part = skel[name]
        t0, t1 = sp["t"]
        ts = ring_ts(part, t0, t1, len(sp["n"]), spacing)
        ds, de = sp["tips"]
        if de is None:   # ear: apex at the skeleton's own tip
            de = float(np.linalg.norm(part.center(1.0) - part.center(t1))) + 0.002
        root = sp.get("root")
        if root and root.get("target") is None:   # host = torso axis at this x
            c0 = part.center(float(ts[0]))
            tz = float(skel["torso"].center(np.clip((c0[0] + 0.184) / 0.31, 0, 1))[2])
            root = {**root, "target": [c0[0], 0.0, tz]}
        V, T, idx, tips = loft(part, ts, sp["n"], ds, de, root, sp.get("dome"))
        Vs.append(V); Ts.append(T + base)
        reg += [REGION_OF_PART[name]] * len(T)
        part_of_tri += [name] * len(T)
        tips_of[name] = (tips[0] + base, tips[1] + base)
        rings_of[name] = [r + base for r in idx]
        base += len(V)
    V = np.concatenate(Vs)
    T = np.concatenate(Ts)
    # flat soles: everything at or below 1 mm goes onto the ground plane (pinned by refiner)
    V[V[:, 2] < 0.001, 2] = 0.0
    if not design.startswith("f"):
        assert len(T) == BUDGET, f"{design}: {len(T)} triangles != {BUDGET}"
    mat = ["fur_orange"] * len(T)
    part_of_tri = np.array(part_of_tri)
    reg = np.array(reg, dtype=object)
    C = V[T].mean(1)
    N = tri_normals(V, T)
    lm3 = proxy.landmarks3d()

    # --- anatomical materials -------------------------------------------------------
    head = skel["head"]
    hfwd = head.tangent(1.0)
    for i in range(len(T)):
        p = part_of_tri[i]
        c = C[i]
        if p.startswith("leg") and c[2] < 0.028:
            mat[i] = "fur_cream"                                   # paws
        if p == "torso" and c[0] > 0.07 and c[2] < 0.20 and N[i, 0] > 0.2:
            mat[i] = "fur_cream"                                   # chest bib
        if p == "neck" and N[i, 0] > 0.3 and c[2] < 0.24:
            mat[i] = "fur_cream"
        if p == "head" and np.linalg.norm(c - lm3["chin"]) < 0.020 and N[i, 2] < 0.3:
            mat[i] = "fur_cream"                                   # muzzle / chin
        if p.startswith("ear") and N[i] @ hfwd > 0.25:
            mat[i] = "ear_inner"
    head_ids = np.flatnonzero(part_of_tri == "head")
    att = {}
    if features == "v0":
        # nose: head faces nearest the proxy nose landmark (muzzle tip)
        nose_att = meshio.attach_point(lm3["nose"], V, T, head_ids)
        nt = nose_att["tri"]
        nose_tris = [j for j in head_ids if np.linalg.norm(C[j] - C[nt]) < 0.009 and N[j] @ hfwd > 0.2]
        for j in nose_tris:
            mat[j] = "nose"; reg[j] = "nose"
        att["nose"] = {"tri": int(nt), "bary": [1 / 3, 1 / 3, 1 / 3]}
        # eyes: pupil = face under the proxy eye landmark, eye = its neighbours within 10 mm
        for side in ("R", "L"):
            a = meshio.attach_point(lm3[f"eye_{side}"], V, T, head_ids)
            pt = a["tri"]
            for j in head_ids:
                if mat[j] in ("nose",):
                    continue
                if np.linalg.norm(C[j] - C[pt]) < 0.010 and len(set(T[j]) & set(T[pt])) > 0:
                    mat[j] = "eye"; reg[j] = "eyes"
            mat[pt] = "pupil"; reg[pt] = "eyes"
            att[f"eye_{side}"] = {"tri": int(pt), "bary": [1 / 3, 1 / 3, 1 / 3]}
    else:
        # v1 features.  nose = upper half of the muzzle-tip fan (tip vertex = landmark);
        # eyes = the full fan of the head vertex nearest each proxy eye landmark, that
        # vertex pushed 2.5 mm into the skull (a shallow socket), pupil = the fan face
        # under the landmark; eye landmark = the socket vertex (centroid of the eye fan).
        tip = int(tips_of["head"][1])
        for j in head_ids:
            if tip in T[j] and N[j, 2] > -0.35:
                mat[j] = "nose"; reg[j] = "nose"
        att["nose"] = {"vertex": tip}
        hv = np.unique(T[head_ids])
        for side in ("R", "L"):
            q = lm3[f"eye_{side}"]
            vi = int(hv[np.argmin(np.linalg.norm(V[hv] - q, axis=1))])
            fan = [j for j in head_ids if vi in T[j]]
            vn = N[fan].mean(0); vn /= np.linalg.norm(vn)
            V[vi] = V[vi] - 0.0025 * vn
            for j in fan:
                mat[j] = "eye"; reg[j] = "eyes"
            a = meshio.attach_point(q, V, T, fan)
            mat[a["tri"]] = "pupil"
            att[f"eye_{side}"] = {"vertex": vi}
        C = V[T].mean(1)
    # --- landmark attachments (CONTRACT §12) ---------------------------------------------
    att["ear_tip_R"] = {"vertex": int(tips_of["ear_R"][1])}
    att["ear_tip_L"] = {"vertex": int(tips_of["ear_L"][1])}
    att["tail_tip"] = {"vertex": int(tips_of["tail"][1])}
    tail_ids = np.flatnonzero(part_of_tri == "tail")
    r0 = rings_of["tail"][0]
    first_band = [j for j in tail_ids if len(set(T[j]) & set(r0)) >= 1 and tips_of["tail"][0] not in T[j]]
    att["tail_base"] = meshio.attach_point(lm3["tail_base"], V, T, first_band)
    mid = rings_of["tail"][len(rings_of["tail"]) // 2]
    mid_band = [j for j in tail_ids if len(set(T[j]) & set(mid)) >= 2]
    att["tail_mid"] = meshio.attach_point(lm3["tail_mid"], V, T, mid_band)
    for leg in ("FR", "FL", "HR", "HL"):
        ids = np.flatnonzero(part_of_tri == f"leg_{leg}")
        low = [j for j in ids if N[j, 2] < -0.3]                   # sole faces
        att[f"paw_{leg}"] = meshio.attach_point(lm3[f"paw_{leg}"], V, T, low or ids)
    att["shoulder_R"] = meshio.attach_point(lm3["shoulder_R"], V, T, np.flatnonzero(part_of_tri == "torso"))
    att["hip_R"] = meshio.attach_point(lm3["hip_R"], V, T, np.flatnonzero(
        (part_of_tri == "torso") | (part_of_tri == "leg_HR")))
    att["hock_HR"] = meshio.attach_point(lm3["hock_HR"], V, T, np.flatnonzero(part_of_tri == "leg_HR"))
    att["chin"] = meshio.attach_point(lm3["chin"], V, T, head_ids)

    d = meshio.to_dict(V, T, list(reg), mat, {k: v for k, v in meshio.landmarks_from_attach(V, T, att).items()},
                       BUDGET, name="cat_1000",
                       notes=f"budget-1000 designed loft topology, design={design}, spacing={spacing}, features={features}; "
                             "closed interpenetrating part components (see variants/1000/NOTES.md)",
                       landmark_attach=att)
    # rings/tips for later targeted edits
    d["build_meta"] = {"design": design, "spacing": spacing,
                       "tips": {k: [int(a), int(b)] for k, (a, b) in tips_of.items()},
                       "rings": {k: [[int(x) for x in r] for r in v] for k, v in rings_of.items()}}
    # rigid-move groups for scripts/refine_vertices.py: one per non-torso part
    d["groups"] = {k: sorted({int(x) for r in v for x in r} | {int(tips_of[k][0]), int(tips_of[k][1])})
                   for k, v in rings_of.items() if k != "torso"}
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", default="head_rich", choices=sorted(DESIGNS))
    ap.add_argument("--spacing", default="curv", choices=["curv", "uniform"])
    ap.add_argument("--features", default="v0", choices=["v0", "v1"],
                    help="v0: eye/nose = face labels only; v1: eye sockets + muzzle-tip nose fan")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = build(a.design, a.spacing, a.features)
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    meshio.save(out, d)
    alloc = {}
    for r in d["tri_region"]:
        alloc[r] = alloc.get(r, 0) + 1
    print(json.dumps({"out": str(out), "triangles": len(d["triangles"]),
                      "vertices": len(d["vertices"]), "by_region": alloc}))


if __name__ == "__main__":
    main()
