"""Head kit: one bilaterally symmetric low-poly cat head at five levels of detail.

Owned by the HEAD KIT stage (head/, scripts/build_head_kit.py, scripts/fit_head_kit.py).
numpy + scipy (ConvexHull) only; CPU only.

Head-local frame (millimetres while authoring; metres in every saved file):
    X = forward (nose), Y = cat's left, Z = up.  Origin = skull centre at pupil height.
    Exact mirror symmetry about Y = 0: every vertex is on the midline (y == 0) or one of
    a mirror pair, and the triangle list is mirror-symmetric too.

Construction (identical machinery for every LOD; only the feature list differs):
  1. A LOD is a list of named feature points on the cat's RIGHT half (y < 0) and on the
     midline.  Each point is either projected radially from the head centre c0 onto an
     analytic head surface (smooth union of ellipsoids: skull, cheeks, jaw, whisker pads,
     nose, brow, chin, neck root) or placed explicitly (neck loop, ear apex).
  2. Connectivity of the half head = convex hull of the half set of unit directions
     (template parameters), minus the facet lying in the y = 0 plane; mirrored to the
     left.  Hull facets spanned only by a ring's points are removed (neck loop -> open
     boundary; ear bases -> holes that the ear solids close) or relabelled (eyes, pupils,
     nose pad).  Topology is fixed per LOD; the fit only moves vertices.
  3. Ears are explicit solids built on the ear-base ring: pyramid (L0-L2), cupped cone
     with an ear_inner front cup (L3), cupped cone with a fur rim around the cup (L4).

`attach_head(body_mesh, head_lod, body_neck_ring)` welds a fitted (world-space) LOD onto
a budget worker's body: bridge their open neck ring to the kit's neck loop.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

LODS = (0, 1, 2, 3, 4)
NECK_SIZES = {0: 4, 1: 6, 2: 6, 3: 8, 4: 10}

# ---------------------------------------------------------------------------- parameters
# All lengths in mm in the head-local frame (before the uniform pose scale).
DEFAULT_SHAPE = {
    # skull ellipsoid
    "skull_d": 39.0, "skull_w": 37.0, "skull_h": 32.0, "skull_cx": -5.0, "skull_cz": 4.0,
    # cheeks (pair)
    "cheek_w": 22.0, "cheek_y": 25.0, "cheek_z": -13.0, "cheek_x": 3.0,
    # muzzle: nose tip x/z, whisker pads
    "nose_x": 45.0, "nose_z": -20.0, "pad_y": 11.0,
    # chin (lowest point of the pale chin)
    "chin_x": 25.0, "chin_z": -45.0,
    # eyes (pupil centre, right eye at -eye_y)
    "eye_x": 31.0, "eye_y": 19.0, "eye_z": 0.0,
    # ears: base inner / outer corner (targets, projected onto the skull), thickness,
    # yaw of the ear's front normal (outward), apex position
    "ear_in_y": 13.0, "ear_in_z": 36.0, "ear_out_y": 39.0, "ear_out_z": 15.0, "ear_base_x": 4.0,
    "ear_d": 12.0, "ear_yaw": 18.0,
    "ear_tip_x": 3.0, "ear_tip_y": 46.0, "ear_tip_z": 60.0,
    # neck loop: throat (front, midline), nape (back, midline), half width
    "throat_x": 6.0, "throat_z": -54.0, "nape_x": -40.0, "nape_z": -14.0, "neck_w": 38.0,
}
# Radial projection centre (mm, local).
C0 = np.array([0.0, 0.0, -8.0])


# ---------------------------------------------------------------------------- analytic surface
def _ellipsoids(S: dict):
    """(centres (K,3), radii (K,3)) of the analytic head, right+left+midline."""
    E = [
        ((S["skull_cx"], 0, S["skull_cz"]), (S["skull_d"], S["skull_w"], S["skull_h"])),
        ((S["cheek_x"], -S["cheek_y"], S["cheek_z"]), (26.0, S["cheek_w"], 22.0)),
        ((S["cheek_x"], S["cheek_y"], S["cheek_z"]), (26.0, S["cheek_w"], 22.0)),
        ((5.0, 0, -30.0), (24.0, 27.0, 17.0)),                                       # jaw
        ((S["nose_x"] - 16, -S["pad_y"], S["nose_z"] - 4), (14.0, 13.0, 11.0)),      # pads
        ((S["nose_x"] - 16, S["pad_y"], S["nose_z"] - 4), (14.0, 13.0, 11.0)),
        ((S["nose_x"] - 9, 0, S["nose_z"] + 2), (9.0, 9.0, 7.5)),                    # nose
        ((S["nose_x"] - 18, 0, S["nose_z"] + 15), (13.0, 13.0, 12.0)),               # bridge/brow
        ((S["chin_x"] - 3, 0, S["chin_z"] + 8), (11.0, 12.0, 8.5)),                  # chin
        ((-14.0, 0, -46.0), (30.0, S["neck_w"] - 1.0, 26.0)),                        # neck root
    ]
    C = np.array([e[0] for e in E], float)
    R = np.array([e[1] for e in E], float)
    return C, R


def head_sdf(S: dict, p: np.ndarray, k: float = 5.0) -> np.ndarray:
    """Approximate signed distance (mm) of the analytic head (smooth union, k mm)."""
    C, R = _ellipsoids(S)
    q = (p[..., None, :] - C) / R                                   # (...,K,3)
    k0 = np.linalg.norm(q, axis=-1)
    k1 = np.linalg.norm(q / R, axis=-1)
    d = k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)                      # (...,K)
    # smooth min over K (polynomial, pairwise-associative via log-sum-exp style)
    return -k * np.log(np.exp(-d / k).sum(-1))


def project_radial(S: dict, dirs: np.ndarray, c0=C0, rmax: float = 110.0, step: float = 0.25):
    """Outermost surface crossing along each unit direction from c0."""
    t = np.arange(0.0, rmax, step)
    P = c0 + dirs[:, None, :] * t[None, :, None]                    # (N,M,3)
    d = head_sdf(S, P)
    inside = d < 0
    last = inside.shape[1] - 1 - np.argmax(inside[:, ::-1], axis=1)  # last inside sample
    i0 = np.clip(last, 0, len(t) - 2)
    d0 = d[np.arange(len(dirs)), i0]
    d1 = d[np.arange(len(dirs)), i0 + 1]
    frac = np.clip(d0 / np.where(np.abs(d0 - d1) < 1e-12, -1e-12, d0 - d1), 0, 1)
    r = t[i0] + frac * step
    return c0 + dirs * r[:, None]


# ---------------------------------------------------------------------------- feature lists
def _ear_base_targets(S, k):
    yaw = math.radians(S["ear_yaw"])
    f = np.array([math.cos(yaw), -math.sin(yaw), 0.0])              # right ear front normal
    ein = np.array([S["ear_base_x"] + 2.0, -S["ear_in_y"], S["ear_in_z"]])
    eout = np.array([S["ear_base_x"] - 2.0, -S["ear_out_y"], S["ear_out_z"]])
    h = 0.5 * S["ear_d"] * f
    if k == 3:
        return {"ear_fi": ein + h, "ear_fo": eout + h, "ear_b": 0.5 * (ein + eout) - h}
    return {"ear_fi": ein + h, "ear_fo": eout + h, "ear_bo": eout - h, "ear_bi": ein - h}


def _neck_targets(S, n):
    T = np.array([S["throat_x"], 0, S["throat_z"]])
    N = np.array([S["nape_x"], 0, S["nape_z"]])
    # L0 (n == 4): the two side vertices also have to carry the cheeks -> wider + higher
    wide = 1.3 if n == 4 else 1.0
    c, a1, a2 = 0.5 * (T + N), 0.5 * (T - N), np.array([0, S["neck_w"] * wide, 0.0])
    if n == 4:
        c = c + np.array([4.0, 0, 6.0])
    out = {}
    for i in range(n):
        th = 2 * math.pi * i / n
        if math.sin(th) > 1e-9:
            continue                                                 # left half: mirrored
        name = "throat" if i == 0 else ("nape" if 2 * i == n else f"neck{i}")
        p = c + math.cos(th) * a1 + math.sin(th) * a2
        if abs(math.sin(th)) < 1e-9:
            p = 0.5 * (T + N) + math.cos(th) * a1
            p[1] = 0.0
        out[name] = p
    return out


def lod_features(lod: int, S: dict) -> list[dict]:
    """Right-half + midline feature points of one LOD.
    Each: {name, p (target, mm), mode 'surf'|'explicit', group, inset (mm, radial)}."""
    nx, nz, ex, ey, ez = S["nose_x"], S["nose_z"], S["eye_x"], S["eye_y"], S["eye_z"]
    cx, cz = S["chin_x"], S["chin_z"]
    F = []

    def add(name, p, mode="surf", group=None, inset=0.0):
        F.append({"name": name, "p": np.asarray(p, float), "mode": mode, "group": group, "inset": inset})

    for n, p in _neck_targets(S, NECK_SIZES[lod]).items():
        add(n, p, "explicit", "neck")
    k_ear = 3 if lod <= 2 else 4
    for n, p in _ear_base_targets(S, k_ear).items():
        add(n, p, "surf", "ear_base")
    sk_top = S["skull_cz"] + S["skull_h"]
    back = S["skull_cx"] - S["skull_d"]

    # --- midline
    add("nose", (nx, 0, nz)) if lod <= 1 else None
    add("crown", (2.0, 0, sk_top + 4))
    if lod >= 1:
        add("brow", (nx - 12, 0, 12))
        add("chin", (cx, 0, cz))
    if lod >= 2:
        add("nose_top", (nx - 1, 0, nz + 4), group="nose")
        add("nose_bot", (nx - 2, 0, nz - 4), group="nose")
    if lod >= 3:
        add("occiput", (back, 0, 6))
        add("mouth", (nx - 8, 0, nz - 13))
        add("bridge", (nx - 6, 0, 2))
        add("forehead", (nx - 22, 0, 26))
    if lod >= 4:
        add("lip_low", (cx + 6, 0, cz + 5))
        add("philtrum", (nx - 3, 0, nz - 8))
        add("crown_back", (-14.0, 0, sk_top + 1))
        add("occ_low", (back + 4, 0, -12))

    # --- right half (y < 0)
    if lod == 0:
        pass
    if lod >= 1:
        add("cheek", (4, -S["cheek_y"] - S["cheek_w"] + 2, -12))
    if lod >= 2:
        add("nose_side", (nx - 3, -6.0, nz + 3.5), group="nose")
    if lod >= 3:
        add("pad", (nx - 10, -14, nz - 6), )
        add("temple", (19, -33, 3))
    if lod >= 2:
        r = 6.5
        if lod == 2:
            for i, a in enumerate((90, 210, 330)):
                t = math.radians(a)
                add(f"eye{i}", (ex - 1, -ey - r * math.cos(t), ez + r * math.sin(t)), group="eye")
    if lod >= 4:
        add("skull_side", (-22, -32, 12))
        add("back_top", (back + 18, 0, sk_top - 4))
        add("cheekbone", (ex - 6, -ey - 6, ez - 10))
    if lod >= 3:
        add("chin_side", (cx - 4, -11, cz + 6))
        add("jaw", (2, -30, -36))
        add("brow_side", (ex - 3, -ey + 1, ez + 11))
        add("mouth_corner", (nx - 14, -13, nz - 13))
        add("back_side", (-36, -22, -6))
        r = 7.0
        n_eye = 4 if lod == 3 else 6
        for i in range(n_eye):
            t = 2 * math.pi * i / n_eye + (math.pi / 2 if lod == 3 else math.pi / 6)
            add(f"eye{i}", (ex - 1, -ey - r * math.cos(t), ez + r * 0.85 * math.sin(t)), group="eye")
        if lod == 3:
            add("eye_c", (ex, -ey, ez), group="eye_centre", inset=1.5)
        else:
            pr, ph = 1.6, 5.2                         # vertical slit pupil (half width, half height)
            add("pup_t", (ex + 0.3, -ey, ez + ph), group="pupil", inset=0.6)
            add("pup_b", (ex + 0.3, -ey, ez - ph), group="pupil", inset=0.6)
            add("pup_i", (ex + 0.6, -ey + pr, ez), group="pupil", inset=0.6)
            add("pup_o", (ex + 0.0, -ey - pr, ez), group="pupil", inset=0.6)
    if lod >= 4:
        add("pad_low", (nx - 9, -9, nz - 11))
        add("pad_out", (nx - 16, -20, nz - 3))
        add("cheek_low", (-2, -40, -26))
        add("cheek_hi", (12, -40, 0))
        add("skull_hi", (-16, -26, 24))
        add("skull_back_hi", (-30, -16, 26))
        add("back_low", (-30, -30, -20))
        add("brow_out", (ex - 9, -ey - 10, ez + 9))
        add("forehead_side", (ex - 12, -12, 25))
        add("jaw_front", (16, -22, -38))
        add("temple_low", (8, -42, -12))
    return F


# ---------------------------------------------------------------------------- ears
def _ear_geometry(lod, base: dict, apex: np.ndarray, S: dict):
    """Ear vertices (dict name -> pos) and triangles (list of (names, material)) for the
    RIGHT ear given its base-ring positions (on the skull) and apex."""
    yaw = math.radians(S["ear_yaw"])
    f = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    P = {"apex": apex}
    tris = []
    if lod <= 2:                     # 3-base pyramid: front face + two back faces
        fi, fo, b = "ear_fi", "ear_fo", "ear_b"
        tris = [((fi, fo, "apex"), "ear_inner"), ((fo, b, "apex"), "fur_orange"),
                ((b, fi, "apex"), "fur_orange")]
        return P, tris
    fi, fo, bo, bi = "ear_fi", "ear_fo", "ear_bo", "ear_bi"
    if lod == 2:                     # 4-base pyramid: separate front (inner) face
        tris = [((fi, fo, "apex"), "ear_inner"), ((fo, bo, "apex"), "fur_orange"),
                ((bo, bi, "apex"), "fur_orange"), ((bi, fi, "apex"), "fur_orange")]
        return P, tris
    t1 = 0.42
    for n in (fi, fo, bo, bi):
        P["r1" + n] = base[n] + t1 * (apex - base[n])
    # convex (slightly bulged) back
    bc = 0.5 * (P["r1" + bo] + P["r1" + bi])
    for n in (bo, bi):
        P["r1" + n] = P["r1" + n] - 0.6 * f
    side = [((fo, bo), "fur_orange"), ((bo, bi), "fur_orange"), ((bi, fi), "fur_orange")]
    for (a, b), m in side:
        tris += [((a, b, "r1" + b), m), ((a, "r1" + b, "r1" + a), m)]
        tris += [(("r1" + a, "r1" + b, "apex"), m)]
    outer = [fi, fo, "r1" + fo, "apex", "r1" + fi]          # front loop, CCW seen from front
    cen = np.mean([base[fi], base[fo], P["r1" + fo], P["r1" + fi], apex], axis=0)
    thick_front_back = np.linalg.norm(base[fi] - base[bi])
    cup = 0.45 * thick_front_back
    pos = lambda n: base[n] if n in base else P[n]
    if lod == 3:
        P["cup"] = cen - cup * f
        for i in range(5):
            tris.append(((outer[i], outer[(i + 1) % 5], "cup"), "ear_inner"))
        return P, tris
    # lod 4: fur rim around an inset cup
    inner = []
    for i, n in enumerate(outer):
        q = pos(n) + (0.30 if n != "apex" else 0.22) * (cen - pos(n)) - 0.15 * cup * f
        P["in" + str(i)] = q
        inner.append("in" + str(i))
    for i in range(5):
        a, b, c, d = outer[i], outer[(i + 1) % 5], inner[(i + 1) % 5], inner[i]
        tris += [((a, b, c), "fur_orange"), ((a, c, d), "fur_orange")]
    P["cup"] = cen - cup * f
    for i in range(5):
        tris.append(((inner[i], inner[(i + 1) % 5], "cup"), "ear_inner"))
    return P, tris


def ear_apex(S):
    return np.array([S["ear_tip_x"], -S["ear_tip_y"], S["ear_tip_z"]])


# ---------------------------------------------------------------------------- topology
def _half_hull(dirs: np.ndarray, mid: np.ndarray):
    from scipy.spatial import ConvexHull
    hull = ConvexHull(dirs)
    tris = []
    for s in hull.simplices:
        if mid[s].all():
            continue                                    # the y = 0 facet
        a, b, c = dirs[s]
        n = np.cross(b - a, c - a)
        if n @ (a + b + c) < 0:
            s = s[[0, 2, 1]]
        tris.append([int(x) for x in s])
    return tris


def _regularise_rings(dirs, groups, mid):
    """Topology-only directions: every ring group (neck, ear base, eye, nose) is moved onto
    a circle around its centroid direction (same angular order, mean radius), so that its
    hull facet is empty and convex.  Positions are NOT affected."""
    D = dirs.copy()
    for g in ("neck", "ear_base", "eye", "nose"):
        ids = [i for i, x in enumerate(groups) if x == g]
        if len(ids) < 3:
            continue
        full = D[ids]
        if g in ("neck", "nose"):          # rings straddling the midline: use full ring
            mirror = full[~mid[ids]] * [1, -1, 1]
            allp = np.vstack([full, mirror])
        else:
            allp = full
        c = allp.mean(0)
        c /= np.linalg.norm(c)
        e1 = np.cross(c, [0, 0, 1.0]) if abs(c[2]) < 0.9 else np.cross(c, [1.0, 0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(c, e1)
        rel = allp - np.outer(allp @ c, c)
        rad = np.linalg.norm(rel, axis=1).mean()
        for i in ids:
            r = D[i] - (D[i] @ c) * c
            ang = math.atan2(r @ e2, r @ e1)
            q = c * math.sqrt(max(1 - rad * rad, 0.0)) + rad * (math.cos(ang) * e1 + math.sin(ang) * e2)
            if mid[i]:
                q[1] = 0.0
                q /= np.linalg.norm(q)
            D[i] = q
    return D


def _orient_bfs(tris, n_fixed):
    """Make winding consistent, propagating from the first n_fixed (outward) triangles."""
    from collections import defaultdict, deque
    T = [list(t) for t in tris]
    et = defaultdict(list)
    for i, t in enumerate(T):
        for k in range(3):
            et[frozenset((t[k], t[(k + 1) % 3]))].append(i)
    seen = [False] * len(T)
    q = deque(range(n_fixed))
    for i in range(n_fixed):
        seen[i] = True
    while q:
        i = q.popleft()
        t = T[i]
        for k in range(3):
            a, b = t[k], t[(k + 1) % 3]
            for j in et[frozenset((a, b))]:
                if seen[j]:
                    continue
                tj = T[j]
                if any((tj[m], tj[(m + 1) % 3]) == (a, b) for m in range(3)):
                    T[j] = [tj[0], tj[2], tj[1]]
                seen[j] = True
                q.append(j)
    if not all(seen):
        raise RuntimeError("orientation: disconnected triangles")
    return T


def build_template(lod: int, S: dict | None = None) -> dict:
    """Connectivity of one LOD (fixed), built from template proportions S."""
    S = {**DEFAULT_SHAPE, **(S or {})}
    F = lod_features(lod, S)
    names = [f["name"] for f in F]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate feature names in LOD {lod}")
    P = _feature_positions(F, S)
    dirs = P - C0
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    mid = np.array([abs(f["p"][1]) < 1e-9 for f in F])
    group = {f["name"]: f["group"] for f in F}
    dirs = _regularise_rings(dirs, [f["group"] for f in F], mid)
    tris = _half_hull(dirs, mid)
    idx = {n: i for i, n in enumerate(names)}

    def all_in(t, g):
        return all(group[names[i]] == g for i in t)

    # holes: neck cap (half) and ear base cap
    half, removed = [], {"neck": 0, "ear_base": 0}
    labels = []
    for t in tris:
        if all_in(t, "neck"):
            removed["neck"] += 1
            continue
        if all_in(t, "ear_base"):
            removed["ear_base"] += 1
            continue
        g = {group[names[i]] for i in t}
        if g == {"pupil"}:
            lab = ("eyes", "pupil")
        elif g <= {"eye", "pupil", "eye_centre"}:
            lab = ("eyes", "eye")
        elif g == {"nose"}:
            lab = ("nose", "nose")
        else:
            lab = ("head", None)                     # material decided from geometry
        half.append(t)
        labels.append(lab)
    # check expected cap sizes
    n_neck = NECK_SIZES[lod]
    exp_neck = (n_neck - 2) // 2
    k_ear = 3 if lod <= 2 else 4
    if removed["neck"] != exp_neck or removed["ear_base"] != k_ear - 2:
        raise RuntimeError(f"LOD {lod}: unexpected caps removed {removed} "
                           f"(want neck {exp_neck}, ear {k_ear - 2})")
    # ears (right), appended as extra named vertices
    base_names = list(_ear_base_targets(S, k_ear).keys())
    Pe, ear_tris = _ear_geometry(lod, {n: P[idx[n]] for n in base_names}, ear_apex(S), S)
    ear_names = list(Pe.keys())
    all_names = names + ear_names
    aidx = {n: i for i, n in enumerate(all_names)}
    for (a, b, c), m in ear_tris:
        half.append([aidx[a], aidx[b], aidx[c]])
        labels.append(("ears", m))
    half = _orient_bfs(half, n_fixed=len(half) - len(ear_tris))
    tpl = {"lod": lod, "names": all_names, "n_feat": len(names), "half_tris": half,
           "half_labels": labels, "features": [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                for k, v in f.items()} for f in F],
           "ear_names": ear_names, "k_ear": k_ear}
    tpl.update(_mirror(tpl))
    return tpl


def _mirror(tpl):
    """Full-head index maps: right/mid vertices first (same order as names), then the
    mirror images of the right vertices."""
    names = tpl["names"]
    feats = {f["name"]: f for f in tpl["features"]}
    is_mid = [(n in feats and abs(feats[n]["p"][1]) < 1e-9) for n in names]
    n0 = len(names)
    mirror_of = {}
    k = n0
    for i, m in enumerate(is_mid):
        if not m:
            mirror_of[i] = k
            k += 1
    full = []
    reg, mat = [], []
    for t, lab in zip(tpl["half_tris"], tpl["half_labels"]):
        full.append(list(t))
        reg.append(lab[0]); mat.append(lab[1])
    for t, lab in zip(tpl["half_tris"], tpl["half_labels"]):
        mt = [mirror_of.get(i, i) for i in t]
        full.append([mt[0], mt[2], mt[1]])
        reg.append(lab[0]); mat.append(lab[1])
    return {"n_vert": k, "is_mid": is_mid, "mirror_of": {int(a): int(b) for a, b in mirror_of.items()},
            "triangles": full, "tri_region": reg, "tri_material_fixed": mat}


# ---------------------------------------------------------------------------- positions
def _feature_positions(F, S):
    P = np.zeros((len(F), 3))
    surf = [i for i, f in enumerate(F) if f["mode"] == "surf"]
    if surf:
        tg = np.array([F[i]["p"] for i in surf])
        d = tg - C0
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        q = project_radial(S, d)
        ins = np.array([F[i]["inset"] for i in surf])
        q = q - d * ins[:, None]
        P[surf] = q
    for i, f in enumerate(F):
        if f["mode"] == "explicit":
            P[i] = f["p"]
    mid = np.array([abs(f["p"][1]) < 1e-9 for f in F])
    P[mid, 1] = 0.0
    return P


def local_vertices(tpl: dict, S: dict, offsets: np.ndarray | None = None) -> np.ndarray:
    """Full-head local vertices (mm) for shape S.  `offsets` (n_half, 3) optional
    per-vertex symmetric offsets on the right/mid vertices (y of mid vertices ignored)."""
    S = {**DEFAULT_SHAPE, **S}
    F = lod_features(tpl["lod"], S)
    P = _feature_positions(F, S)
    idx = {f["name"]: i for i, f in enumerate(F)}
    base = {n: P[idx[n]] for n in ("ear_fi", "ear_fo", "ear_bo", "ear_bi", "ear_b") if n in idx}
    Pe, _ = _ear_geometry(tpl["lod"], base, ear_apex(S), S)
    half = np.vstack([P, np.array([Pe[n] for n in tpl["ear_names"]])])
    if offsets is not None:
        half = half + offsets
    half[np.array(tpl["is_mid"]), 1] = 0.0
    V = np.zeros((tpl["n_vert"], 3))
    V[:len(half)] = half
    for a, b in tpl["mirror_of"].items():
        V[b] = half[a] * np.array([1, -1, 1])
    return V


def tri_materials(tpl: dict, V: np.ndarray, S: dict) -> list[str]:
    """Fixed labels + fur_cream for muzzle/chin/throat (by local position)."""
    S = {**DEFAULT_SHAPE, **S}
    T = np.asarray(tpl["triangles"])
    cen = V[T].mean(1)
    out = []
    for m, c in zip(tpl["tri_material_fixed"], cen):
        if m is not None:
            out.append(m)
            continue
        x, y, z = c
        cream = ((x > S["nose_x"] - 30 and z < S["nose_z"] + 1 and abs(y) < 20) or
                 (x > -2 and z < S["chin_z"] + 6 and abs(y) < 22))
        out.append("fur_cream" if cream else "fur_orange")
    return out


# ---------------------------------------------------------------------------- pose
def rot_matrix(yaw, pitch, roll):
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll), degrees.  Positive pitch tips the nose DOWN."""
    y, p, r = (math.radians(a) for a in (yaw, pitch, roll))
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    return Rz @ Ry @ Rx


def to_world(V_mm: np.ndarray, pose: dict) -> np.ndarray:
    R = rot_matrix(pose["yaw_deg"], pose["pitch_deg"], pose["roll_deg"])
    return np.asarray(pose["t"], float) + pose["scale"] * 1e-3 * (V_mm @ R.T)


# ---------------------------------------------------------------------------- export
def neck_loop(tpl: dict) -> list[int]:
    """Ordered neck loop (full-mesh vertex indices), starting at the throat, going
    round the cat's right side first."""
    names = tpl["names"]
    n = NECK_SIZES[tpl["lod"]]
    right = []
    for i in range(n):
        th = 2 * math.pi * i / n
        if math.sin(th) > 1e-9:
            continue
        nm = "throat" if i == 0 else ("nape" if 2 * i == n else f"neck{i}")
        right.append((i, names.index(nm)))
    order = {}
    for i, vi in right:
        order[i] = vi
        if i not in (0, n // 2):
            order[n - i] = tpl["mirror_of"][vi]          # mirror angle 2pi - th
    # angles decreasing sin<0 first: i = n-1 ... is left; we want throat -> right -> nape -> left
    seq = [order[0]] + [order[i] for i in range(n - 1, n // 2, -1)] + [order[n // 2]] + \
          [order[i] for i in range(n // 2 - 1, 0, -1)]
    # right side vertices have sin<0 -> indices i in (n/2, n); listed above after throat
    return [int(v) for v in seq]


def landmark_attach(tpl: dict, V: np.ndarray, S: dict) -> dict:
    """§12 attachments for the six head landmarks."""
    from .meshio import attach_point
    S = {**DEFAULT_SHAPE, **S}
    names, T = tpl["names"], np.asarray(tpl["triangles"])
    reg, mat = tpl["tri_region"], tpl["tri_material_fixed"]
    ii = {n: i for i, n in enumerate(names)}
    mir = tpl["mirror_of"]
    att = {"ear_tip_R": {"vertex": ii["apex"]}, "ear_tip_L": {"vertex": mir[ii["apex"]]}}
    # eyes
    for side, sgn in (("R", -1), ("L", 1)):
        tgt = np.array([S["eye_x"], sgn * S["eye_y"], S["eye_z"]])
        if "eye_c" in ii:
            v = ii["eye_c"] if side == "R" else mir[ii["eye_c"]]
            att[f"eye_{side}"] = {"vertex": int(v)}
            continue
        if "pup_t" in ii:
            cand = [i for i, m in enumerate(mat) if m == "pupil"]
        elif "eye0" in ii:
            cand = [i for i, r in enumerate(reg) if r == "eyes"]
        else:
            cand = [i for i, r in enumerate(reg) if r == "head"]
        cand = [i for i in cand if np.sign(V[T[i]].mean(0)[1]) == sgn]
        if "eye0" in ii and "pup_t" not in ii:          # L2: centroid of the eye facet
            att[f"eye_{side}"] = {"tri": int(cand[0]), "bary": [1 / 3, 1 / 3, 1 - 2 / 3]}
        else:
            # project the target radially onto the surface first, then closest point
            d = (tgt - C0) / np.linalg.norm(tgt - C0)
            p = project_radial(S, d[None])[0]
            att[f"eye_{side}"] = attach_point(p, V, T, cand)
    if "nose" in ii:
        att["nose"] = {"vertex": ii["nose"]}
    else:
        cand = [i for i, m in enumerate(mat) if m == "nose"]
        tgt = V[[ii["nose_top"], ii["nose_bot"], ii["nose_side"], mir[ii["nose_side"]]]].mean(0)
        att["nose"] = attach_point(tgt, V, T, cand)
    if "chin" in ii:
        att["chin"] = {"vertex": ii["chin"]}
    else:
        tgt = np.array([S["chin_x"], 0, S["chin_z"]])
        cand = [i for i, r in enumerate(reg) if r == "head"]
        att["chin"] = attach_point(tgt, V, T, cand)
    return att


def export_lod(tpl: dict, S: dict, pose: dict | None = None, offsets=None) -> dict:
    """Head-kit LOD dict (local metres if pose is None, else world metres)."""
    S = {**DEFAULT_SHAPE, **S}
    Vmm = local_vertices(tpl, S, offsets)
    att = landmark_attach(tpl, Vmm, S)
    mats = tri_materials(tpl, Vmm, S)
    V = to_world(Vmm, pose) if pose else Vmm * 1e-3
    from .meshio import landmarks_from_attach
    lm = landmarks_from_attach(V, tpl["triangles"], att)
    pairs = [[int(a), int(b)] for a, b in tpl["mirror_of"].items()]
    return {
        "format": "headkit-1", "lod": tpl["lod"], "frame": "world" if pose else "head-local",
        "units": "m",
        "vertices": [[float(c) for c in v] for v in V],
        "triangles": [[int(c) for c in t] for t in tpl["triangles"]],
        "tri_region": list(tpl["tri_region"]), "tri_material": mats,
        "neck_loop": neck_loop(tpl),
        "mirror_pairs": pairs,
        "midline_vertices": [i for i, m in enumerate(tpl["is_mid"]) if m],
        "landmark_attach": att,
        "landmarks": {k: [float(c) for c in v] for k, v in lm.items()},
        "vertex_names": tpl["names"] + [tpl["names"][a] + "_L" for a in tpl["mirror_of"]],
        "n_triangles": len(tpl["triangles"]),
    }


# ---------------------------------------------------------------------------- checks
def check_lod(d: dict) -> dict:
    """Manifold / boundary / degeneracy / symmetry / self-intersection checks."""
    from collections import Counter
    from .geom import topology_report
    V = np.asarray(d["vertices"], float)
    T = np.asarray(d["triangles"], np.int64)
    rep = topology_report(V, T)
    ec = Counter()
    de = Counter()
    for t in T:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            ec[(min(a, b), max(a, b))] += 1
            de[(a, b)] += 1
    bnd = {e for e, c in ec.items() if c == 1}
    loop = d["neck_loop"]
    loop_edges = {(min(loop[i], loop[(i + 1) % len(loop)]), max(loop[i], loop[(i + 1) % len(loop)]))
                  for i in range(len(loop))}
    out = {
        "triangles": int(len(T)),
        "boundary_is_neck_loop": bnd == loop_edges,
        "boundary_edges": len(bnd),
        "non_manifold_edges": int(sum(1 for c in ec.values() if c > 2)),
        "inconsistent_winding_edges": int(sum(1 for c in de.values() if c > 1)),
        "degenerate_triangles": len(rep["degenerate_triangles"]),
        "min_triangle_area_mm2": rep["min_triangle_area_m2"] * 1e6,
        "min_edge_mm": rep["min_edge_m"] * 1e3,
        "duplicates": len(rep["duplicate_triangles_same_indices"]) +
                      len(rep["duplicate_triangles_coincident_positions"]),
        "components": rep["components"],
        "self_intersecting_pairs": self_intersections(V, T),
    }
    if d.get("frame") == "head-local":
        pairs = np.asarray(d["mirror_pairs"])
        mv = np.asarray(d["midline_vertices"])
        err = np.abs(V[pairs[:, 0]] * [1, -1, 1] - V[pairs[:, 1]]).max()
        out["mirror_max_err_m"] = float(max(err, np.abs(V[mv, 1]).max() if len(mv) else 0))
        mp = {int(a): int(b) for a, b in pairs}
        mp.update({b: a for a, b in list(mp.items())})
        tset = {tuple(sorted(t)) for t in T.tolist()}
        out["triangulation_symmetric"] = all(tuple(sorted(mp.get(i, i) for i in t)) in tset for t in T.tolist())
    out["ok"] = (out["boundary_is_neck_loop"] and out["non_manifold_edges"] == 0 and
                 out["inconsistent_winding_edges"] == 0 and out["degenerate_triangles"] == 0 and
                 out["duplicates"] == 0 and out["self_intersecting_pairs"] == 0 and
                 out["components"] == 1)
    return out


def _tri_tri_intersect(p, q):
    """Möller-style test via edge-triangle segment intersection (non-adjacent tris)."""
    def seg_tri(a, b, t):
        e1, e2 = t[1] - t[0], t[2] - t[0]
        d = b - a
        h = np.cross(d, e2)
        det = e1 @ h
        if abs(det) < 1e-14:
            return False
        f = 1.0 / det
        s = a - t[0]
        u = f * (s @ h)
        if u < 0 or u > 1:
            return False
        qv = np.cross(s, e1)
        v = f * (d @ qv)
        if v < 0 or u + v > 1:
            return False
        tt = f * (e2 @ qv)
        return 0 <= tt <= 1
    for i in range(3):
        if seg_tri(p[i], p[(i + 1) % 3], q) or seg_tri(q[i], q[(i + 1) % 3], p):
            return True
    return False


def self_intersections(V, T) -> int:
    V = np.asarray(V, float)
    T = np.asarray(T, np.int64)
    lo, hi = V[T].min(1), V[T].max(1)
    n = 0
    for i in range(len(T)):
        ov = np.flatnonzero((lo[i + 1:] <= hi[i]).all(1) & (hi[i + 1:] >= lo[i]).all(1)) + i + 1
        si = set(T[i].tolist())
        for j in ov:
            if si & set(T[j].tolist()):
                continue
            if _tri_tri_intersect(V[T[i]], V[T[j]]):
                n += 1
    return n


# ---------------------------------------------------------------------------- weld
def bridge_cost(lod: int, body_ring: int) -> int:
    return NECK_SIZES[lod] + body_ring if NECK_SIZES[lod] != body_ring else 2 * body_ring


def attach_head(body_mesh: dict, lod, body_neck_ring, region: str = "neck",
                material: str = "fur_orange"):
    """Weld a fitted head-kit LOD onto a catmesh-1 body.

    body_mesh: catmesh-1 dict whose open neck ring is `body_neck_ring` (ordered indices).
    lod: head-kit dict (world frame) or a path to head/head_L*.json.
    Returns (combined catmesh-1 dict, frozen vertex indices).  The combined mesh carries
    the head's triangles (regions/materials), the bridge band (region `neck`), the head's
    six landmark attachments (re-indexed) plus the body's other attachments, and
    `freeze_vertices` = every kit vertex except the neck loop."""
    from . import topo, meshio
    if not isinstance(lod, dict):
        lod = json.loads(Path(lod).read_text())
    if lod.get("frame") != "world":
        raise ValueError("attach_head needs a fitted world-frame head (head/head_L*.json)")
    body = copy.deepcopy(body_mesh)
    nb = len(body["vertices"])
    ntb = len(body["triangles"])
    Vh = lod["vertices"]
    Th = [[a + nb, b + nb, c + nb] for a, b, c in lod["triangles"]]
    d = copy.deepcopy(body)
    d["vertices"] = list(body["vertices"]) + [list(v) for v in Vh]
    d["triangles"] = list(body["triangles"]) + Th
    d["tri_region"] = list(body["tri_region"]) + list(lod["tri_region"])
    d["tri_material"] = list(body["tri_material"]) + list(lod["tri_material"])
    att = {k: v for k, v in (body.get("landmark_attach") or {}).items() if k not in lod["landmark_attach"]}
    for k, a in lod["landmark_attach"].items():
        att[k] = {"vertex": a["vertex"] + nb} if "vertex" in a else {"tri": a["tri"] + ntb, "bary": a["bary"]}
    d["landmark_attach"] = att
    if "landmarks" in body:
        d["landmarks"] = dict(body["landmarks"])
    head_loop = [i + nb for i in lod["neck_loop"]]
    d = topo.bridge_rings(d, head_loop, list(body_neck_ring), region=region, material=material)
    loop_set = set(head_loop)
    frozen = [i + nb for i in range(len(Vh)) if i + nb not in loop_set]
    d["freeze_vertices"] = frozen
    d["head_kit"] = {"lod": lod["lod"], "vertex_offset": nb, "triangle_offset": ntb,
                     "neck_loop": head_loop, "bridge_triangles": bridge_cost(lod["lod"], len(body_neck_ring))}
    meshio.sync_landmarks(d)
    return d, frozen
