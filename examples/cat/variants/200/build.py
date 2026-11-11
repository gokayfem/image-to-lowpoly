#!/usr/bin/env python
"""Budget-200 cat: explicit low-poly topology built from the shared proxy skeleton.

    python3 variants/200/build.py --alloc A --out variants/200/checkpoints/v0_initial/mesh.json

Deterministic.  Components (all triangles counted, CONTRACT §5/§6):
  torso_neck : one tube rump -> chest -> neck, closed at the rump by a tip cap, the neck end
               is an open rim buried inside the head (allowed with disclosure, brief).
  head       : closed: back cap (fan or tip) + rings + muzzle cone to a nose-tip vertex.
  ear_R/L    : thin pyramids (k-sided base, open base buried inside the head).
  leg_*      : open top rim buried in the torso, bend ring(s), ground sole polygon (fan) with
               a forward toe vertex; sole vertices at z = 0.
  tail       : open start rim buried in the rump, rings, tip vertex.
Allocations (see NOTES.md):
  A: 6-sided torso (5 body rings + neck ring), 7-sided head, 4-sided ears/legs/tail.
  B: 5-sided torso (6 body rings + neck ring), 8-sided head, 3-sided ears/legs (2 bend rings),
     3-sided tail.
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

from catlib import meshio, topo  # noqa: E402
from catlib.skeleton import load_skeleton  # noqa: E402

BUDGET = 200
PI = np.pi

ALLOCS = {
    "A": dict(
        torso_n=6, torso_phase=PI / 2, torso_ts=[0.0, 0.30, 0.62, 1.0], neck_ts=[0.45, 0.80],
        rump_cap="tip",
        head_n=7, head_phase=PI / 2, head_ts=[0.12, 0.50], head_back="fan",
        ear_n=4,
        leg_n=4, leg_mid={"leg_FR": [0.68], "leg_FL": [0.673], "leg_HR": [0.65], "leg_HL": [0.648]},
        tail_n=4, tail_ts=[0.0, 0.30, 0.60, 0.86],
    ),
    "B": dict(
        torso_n=5, torso_phase=PI / 2, torso_ts=[0.0, 0.20, 0.45, 0.70, 1.0], neck_ts=[0.45, 0.80],
        rump_cap="tip",
        head_n=8, head_phase=0.0, head_ts=[0.22, 0.52], head_back="tip",
        ear_n=3,
        leg_n=3, leg_mid={"leg_FR": [0.369, 0.68], "leg_FL": [0.367, 0.673],
                          "leg_HR": [0.285, 0.65], "leg_HL": [0.285, 0.648]},
        tail_n=3, tail_ts=[0.0, 0.30, 0.58, 0.84],
    ),
    # C (v2 edit of B): one torso ring fewer (torso IoU already 0.99), flat head back;
    # spent on 4-sided wider ears, a tail ring at the rump exit, 4-corner front soles.
    "C": dict(
        torso_n=5, torso_phase=PI / 2, torso_ts=[0.0, 0.30, 0.65, 1.0], neck_ts=[0.45, 0.80],
        rump_cap="tip",
        head_n=8, head_phase=0.0, head_ts=[0.12, 0.52], head_back="fan",
        ear_n=4, ear_scale=1.25,
        leg_n=3, leg_mid={"leg_FR": [0.369, 0.68], "leg_FL": [0.367, 0.673],
                          "leg_HR": [0.285, 0.65], "leg_HL": [0.285, 0.648]},
        sole_n={"leg_FR": 4, "leg_FL": 4},
        tail_n=3, tail_ts=[0.0, 0.08, 0.32, 0.58, 0.84], tail_base_ring=1,
    ),
}
# D (v3): allocation C, tail_base attached to the surface around the rump-exit ring instead of
# one of its three vertices (a ring vertex sits ~60 px off the tail centreline).
ALLOCS["D"] = dict(ALLOCS["C"], tail_base_mode="surface")

PAW_LM = {"leg_FR": "paw_FR", "leg_FL": "paw_FL", "leg_HR": "paw_HR", "leg_HL": "paw_HL"}


class Builder:
    def __init__(self):
        self.V: list[np.ndarray] = []
        self.T: list[list[int]] = []
        self.reg: list[str] = []
        self.mat: list[str] = []
        self.comp: list[str] = []
        self.open_rims: dict[str, tuple[list[int], str]] = {}   # comp -> (rim vids, burying comp)

    def add_verts(self, P) -> list[int]:
        P = np.atleast_2d(np.asarray(P, float))
        base = len(self.V)
        self.V.extend(list(P))
        return list(range(base, base + len(P)))

    def tri(self, a, b, c, region, material, comp):
        self.T.append([a, b, c]); self.reg.append(region); self.mat.append(material); self.comp.append(comp)

    def band(self, A, B, region, comp, material="fur_orange"):
        n = len(A)
        assert n == len(B)
        V = np.asarray(self.V)
        for i in range(n):
            a0, a1, b0, b1 = A[i], A[(i + 1) % n], B[i], B[(i + 1) % n]
            if np.linalg.norm(V[a0] - V[b1]) <= np.linalg.norm(V[a1] - V[b0]):
                self.tri(a0, a1, b1, region, material, comp); self.tri(a0, b1, b0, region, material, comp)
            else:
                self.tri(a0, a1, b0, region, material, comp); self.tri(a1, b1, b0, region, material, comp)

    def band_uneq(self, A, B, region, comp, material="fur_orange"):
        """Zipper between loops of different sizes (len(A)+len(B) triangles); B is rotated
        so that B[0] is the vertex nearest A[0]; both loops must run in the same sense."""
        V = np.asarray(self.V)
        j0 = int(np.argmin([np.linalg.norm(V[A[0]] - V[b]) for b in B]))
        B = list(B[j0:]) + list(B[:j0])
        n, m = len(A), len(B)
        i = j = 0
        seen = {(0, 0)}
        while i < n or j < m:
            ai, ai1 = A[i % n], A[(i + 1) % n]
            bj, bj1 = B[j % m], B[(j + 1) % m]
            if i >= n:
                adv_a = False
            elif j >= m:
                adv_a = True
            else:
                adv_a = np.linalg.norm(V[ai1] - V[bj]) <= np.linalg.norm(V[ai] - V[bj1])
                # never step onto an already used (A, B) vertex pair before the closing step
                # (that would wrap one ring around a single vertex of the other: non-manifold).
                # Valid zippers never do this, so it only changes otherwise-broken outputs.
                nxt = ((i + 1) % n, j % m) if adv_a else (i % n, (j + 1) % m)
                last = (i + 1 == n and j == m) or (i == n and j + 1 == m)
                if nxt in seen and not last:
                    adv_a = not adv_a
            seen.add(((i + 1) % n, j % m) if adv_a else (i % n, (j + 1) % m))
            if adv_a:
                self.tri(ai, ai1, bj, region, material, comp); i += 1
            else:
                self.tri(ai, bj1, bj, region, material, comp); j += 1

    def cone(self, R, apex, region, comp, material="fur_orange"):
        for i in range(len(R)):
            self.tri(R[i], R[(i + 1) % len(R)], apex, region, material, comp)

    def fan(self, R, region, comp, material="fur_orange"):
        for k in range(1, len(R) - 1):
            self.tri(R[0], R[k], R[k + 1], region, material, comp)


def ring_pts(part, t, n, phase, shift=None, scale=1.0):
    P = part.ring(t, n, phase)
    c = part.center(t)
    P = c + scale * (P - c)
    if shift is not None:
        P = P + np.asarray(shift, float)
    return P


def winding_number(p, V, T):
    """Generalised winding number of point p w.r.t. a closed triangle set."""
    A, B, C = V[T[:, 0]] - p, V[T[:, 1]] - p, V[T[:, 2]] - p
    la, lb, lc = (np.linalg.norm(X, axis=1) for X in (A, B, C))
    num = (A * np.cross(B, C)).sum(1)
    den = la * lb * lc + (A * B).sum(1) * lc + (B * C).sum(1) * la + (C * A).sum(1) * lb
    return float(np.sum(2 * np.arctan2(num, den)) / (4 * PI))


def build(alloc: str) -> dict:
    cfg = ALLOCS[alloc]
    sk = load_skeleton()
    lm3 = json.loads((ROOT / "proxy" / "landmarks3d.json").read_text())["landmarks"]
    b = Builder()

    # ---- torso + neck --------------------------------------------------------------------
    torso, neck, head = sk["torso"], sk["neck"], sk["head"]
    n = cfg["torso_n"]
    rings = []
    for t in cfg["torso_ts"]:   # the t = 1 ring is the chest front, pushed into the end-cap volume
        sh = torso.tangent(1.0) * 0.6 * torso.end_cap_depth("end") if t >= 1.0 else None
        rings.append(b.add_verts(ring_pts(torso, t, n, cfg["torso_phase"], shift=sh,
                                          scale=0.85 if t >= 1.0 else 1.0)))
    hcen = head.center(0.42)
    for k, t in enumerate(cfg["neck_ts"]):
        if k < len(cfg["neck_ts"]) - 1:     # full neck ring
            rings.append(b.add_verts(ring_pts(neck, t, n, cfg["torso_phase"])))
        else:                               # neck end: pulled into the head, open rim buried
            nc = neck.center(t)
            rings.append(b.add_verts(ring_pts(neck, t, n, cfg["torso_phase"], shift=(hcen - nc) * 1.0,
                                              scale=0.5)))
    nb = len(cfg["torso_ts"])
    for i in range(len(rings) - 1):
        region = "torso" if i < nb - 1 else "neck"
        b.band(rings[i], rings[i + 1], region, "torso_neck")
    # rump cap
    t0 = cfg["torso_ts"][0]
    axis = -torso.tangent(t0)
    if cfg["rump_cap"] == "tip":
        apex = b.add_verts(torso.center(t0) + axis * torso.end_cap_depth("start"))[0]
        b.cone(rings[0][::-1], apex, "torso", "torso_neck")
    else:
        b.fan(rings[0][::-1], "torso", "torso_neck")
    b.open_rims["torso_neck"] = (rings[-1], "head")

    # ---- head ----------------------------------------------------------------------------
    n = cfg["head_n"]
    hr = [b.add_verts(ring_pts(head, t, n, cfg["head_phase"])) for t in cfg["head_ts"]]
    b.band(hr[0], hr[1], "head", "head")
    if cfg["head_back"] == "tip":
        back = b.add_verts(head.center(0.0) - head.tangent(0.0) * head.end_cap_depth("start"))[0]
        b.cone(hr[0][::-1], back, "head", "head")
    else:
        b.fan(hr[0][::-1], "head", "head")
    nose_v = b.add_verts(np.asarray(lm3["nose"]) + head.tangent(1.0) * 0.004)[0]
    b.cone(hr[1], nose_v, "head", "head", "fur_cream")

    # ---- ears ----------------------------------------------------------------------------
    hc = head.center(0.3)
    ear_tip_v = {}
    for side in ("R", "L"):
        ear = sk[f"ear_{side}"]
        base_c = ear.center(0.0)
        shift = (hc - base_c) * 0.4
        base = b.add_verts(ring_pts(ear, 0.0, cfg["ear_n"], 0.0, shift=shift,
                                    scale=cfg.get("ear_scale", 1.0)))
        tip = b.add_verts(np.asarray(lm3[f"ear_tip_{side}"]))[0]
        ear_tip_v[side] = tip
        b.cone(base, tip, "ears", f"ear_{side}")
        b.open_rims[f"ear_{side}"] = (base, "head")

    # ---- legs ----------------------------------------------------------------------------
    n = cfg["leg_n"]
    leg_phase = 0.0 if n == 4 else PI / 2          # 4: +side/front/-side/back; 3: front vertex
    sole_tris = {}
    for name in ("leg_FR", "leg_FL", "leg_HR", "leg_HL"):
        leg = sk[name]
        c0 = leg.center(0.0)
        top = b.add_verts(ring_pts(leg, 0.0, n, leg_phase, shift=[-0.02 if name.startswith('leg_F') else 0.0, -0.5 * c0[1], 0.03], scale=0.7))
        mids = [b.add_verts(ring_pts(leg, t, n, leg_phase)) for t in cfg["leg_mid"][name]]
        # sole polygon on z = 0, same angular order as the rings (u ~ +y, v ~ +x)
        paw = np.asarray(lm3[PAW_LM[name]], float)
        ns = cfg.get("sole_n", {}).get(name, n)
        th = (leg_phase if ns == n else PI / 4) + 2 * PI * np.arange(ns) / ns
        hind = name.startswith("leg_H")
        half_w = 0.015 if not hind else 0.014
        toe, heel = (0.026, 0.016) if not hind else (0.024, 0.018)
        pts = []
        for a in th:
            cu, cv = np.cos(a), np.sin(a)
            x = paw[0] + (toe if cv > 0 else heel) * (cv if ns == 3 or ns == n else np.sign(cv) * 0.8)
            y = paw[1] + half_w * (cu if ns == n else np.sign(cu) * (0.9 if cv > 0 else 0.75))
            pts.append([x, y, 0.0])
        sole = b.add_verts(np.asarray(pts))
        chain = [top] + mids + [sole]
        for i in range(len(chain) - 1):
            if len(chain[i]) == len(chain[i + 1]):
                b.band(chain[i], chain[i + 1], name, name)
            else:
                b.band_uneq(chain[i], chain[i + 1], name, name)
        k0 = len(b.T)
        b.fan(sole[::-1], name, name, "fur_cream")
        sole_tris[name] = list(range(k0, len(b.T)))
        b.open_rims[name] = (top, "torso_neck")

    # ---- tail ----------------------------------------------------------------------------
    tail = sk["tail"]
    n = cfg["tail_n"]
    ts = cfg["tail_ts"]
    tr = []
    for t in ts:
        shift = None
        if t == 0.0:   # push the open start rim into the rump
            shift = -tail.tangent(0.0) * 0.045 + np.array([0, 0, -0.01])
        tr.append(b.add_verts(ring_pts(tail, t, n, 0.0, shift=shift, scale=0.9 if t == 0 else 1.0)))
    for i in range(len(tr) - 1):
        b.band(tr[i], tr[i + 1], "tail", "tail")
    tail_tip = b.add_verts(np.asarray(lm3["tail_tip"]))[0]
    b.cone(tr[-1], tail_tip, "tail", "tail")
    b.open_rims["tail"] = (tr[0], "torso_neck")

    V = np.asarray(b.V)
    T = np.asarray(b.T)
    d = meshio.to_dict(V, T, b.reg, b.mat, {k: v for k, v in lm3.items()}, BUDGET,
                       name="cat_200", notes=f"variants/200/build.py alloc {alloc}")
    d = topo.orient_outward(d)
    T = np.asarray(d["triangles"])

    # ---- materials: belly/chest underside cream, ear front faces pink ----------------------
    comp = np.asarray(b.comp)
    Vt = V[T]
    nrm = np.cross(Vt[:, 1] - Vt[:, 0], Vt[:, 2] - Vt[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    mat = list(d["tri_material"])
    hax = head.tangent(1.0)
    for i in range(len(T)):
        if comp[i] == "torso_neck" and nrm[i, 2] < -0.5:
            mat[i] = "fur_cream"
        if comp[i] in ("ear_R", "ear_L") and np.dot(nrm[i], hax) > 0.3:
            mat[i] = "ear_inner"
    d["tri_material"] = mat

    # ---- landmark attachment (CONTRACT §12) ----------------------------------------------
    def tris_of(*comps):
        return [i for i in range(len(T)) if comp[i] in comps]
    att = {
        "ear_tip_R": {"vertex": int(ear_tip_v["R"])},
        "ear_tip_L": {"vertex": int(ear_tip_v["L"])},
        "nose": {"vertex": int(nose_v)},
        "tail_tip": {"vertex": int(tail_tip)},
    }
    for name, lmn in PAW_LM.items():
        att[lmn] = meshio.attach_point(np.asarray(lm3[lmn]), V, T, sole_tris[name])
    if "tail_base_ring" in cfg:     # first visible tail ring (where the tail leaves the rump)
        rv = tr[cfg["tail_base_ring"]]
        p = np.asarray(lm3["tail_base"], float)
        if cfg.get("tail_base_mode") == "surface":   # nearest point on faces around that ring
            ids = [i for i in range(len(T)) if comp[i] == "tail" and set(T[i]) & set(rv)]
            att["tail_base"] = meshio.attach_point(p, V, T, ids)
        else:
            att["tail_base"] = {"vertex": int(rv[int(np.argmin([np.linalg.norm(V[v] - p) for v in rv]))])}
    for lmn, comps in (("eye_R", ("head",)), ("eye_L", ("head",)), ("chin", ("head",)),
                       *((("tail_base", ("tail",)),) if "tail_base_ring" not in cfg else ()), ("tail_mid", ("tail",)),
                       ("shoulder_R", ("torso_neck", "leg_FR")), ("hip_R", ("torso_neck", "leg_HR")),
                       ("hock_HR", ("leg_HR",))):
        p = np.asarray(lm3[lmn], float)
        ids = tris_of(*comps)
        att[lmn] = meshio.attach_point(p, V, T, ids)
    d["landmark_attach"] = att
    meshio.sync_landmarks(d)

    # ---- checks ----------------------------------------------------------------------------
    assert len(T) == BUDGET, f"triangle count {len(T)} != {BUDGET}"
    burial = {}
    for c, (rim, host) in b.open_rims.items():
        hostT = T[comp != c]
        wn = [winding_number(V[v], V, hostT) for v in rim]
        burial[c] = min(wn)
        if min(wn) < 0.5:
            print(f"WARNING: open rim of {c} not buried in {host}: winding {np.round(wn, 2)}")
    d["notes"] += (f"; components {sorted(set(comp))}; open rims buried "
                   f"{ {k: round(v, 2) for k, v in burial.items()} }")
    d["components"] = {c: int((comp == c).sum()) for c in sorted(set(comp))}
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alloc", default="W", choices=sorted(ALLOCS) + ["W", "K1", "K2", "K3"],
                    help="A/B/C/D: round-1 part-based builds; W: welded single manifold (round 2)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--freeze-out", type=Path, default=None,
                    help="(K1/K2) write the head-kit frozen vertex list (for refine --freeze-verts)")
    ap.add_argument("--stages-dir", type=Path, default=None,
                    help="(W only) also write construction stages in build order (timelapse)")
    a = ap.parse_args(argv)
    if a.alloc in ("K1", "K2", "K3"):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from kitbody import CFG_K, build_kit
        stages = [] if a.stages_dir else None
        d = build_kit(dict(CFG_K[a.alloc], name=a.alloc), Builder, ring_pts, BUDGET, stages)
        if a.freeze_out:
            a.freeze_out.parent.mkdir(parents=True, exist_ok=True)
            a.freeze_out.write_text(json.dumps(d["freeze_vertices"]))
        if stages is not None:
            a.stages_dir.mkdir(parents=True, exist_ok=True)
            for i, (label, sd) in enumerate(stages, 1):
                fn = a.stages_dir / f"{i:02d}_{label.replace(' ', '_')}.json"
                fn.write_text(json.dumps({"label": label, **sd}))
            print(f"  wrote {len(stages)} construction stages to {a.stages_dir}")
    elif a.alloc == "W":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from welded import CFG_W, build_welded
        stages = [] if a.stages_dir else None
        d = build_welded(CFG_W, Builder, ring_pts, BUDGET, stages)
        if stages is not None:
            a.stages_dir.mkdir(parents=True, exist_ok=True)
            for i, (label, sd) in enumerate(stages, 1):
                fn = a.stages_dir / f"{i:02d}_{label.replace(' ', '_')}.json"
                fn.write_text(json.dumps({"label": label, **sd}))
            print(f"  wrote {len(stages)} construction stages to {a.stages_dir}")
    else:
        d = build(a.alloc)
    meshio.save(a.out, d)
    reg = {}
    for r in d["tri_region"]:
        reg[r] = reg.get(r, 0) + 1
    print(f"alloc {a.alloc}: {len(d['triangles'])} triangles, {len(d['vertices'])} vertices -> {a.out}")
    print("  by region:", reg)
    print("  by component:", d["components"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
