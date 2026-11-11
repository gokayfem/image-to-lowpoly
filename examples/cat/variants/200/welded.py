"""Budget-200 welded single-manifold construction (allocation "W", round 2).

One closed genus-0 surface, so T = 2V - 4: 200 triangles <=> 102 vertices.  Built by
cut-and-bridge: the torso/neck/head tube is generated with the limb- and tail-root quads
left out; each such 4-vertex hole loop is bridged directly to the limb's first ring.
Ears are pyramids standing on a head quad (they share the head ring vertices).
Two edge splits (chin, brisket) bring V from 100 to 102.

Called from build.py (``--alloc W``); see NOTES.md for the allocation table.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from catlib import meshio, topo
from catlib.skeleton import load_skeleton

PI = np.pi
ROOT = Path(__file__).resolve().parents[2]

CFG_W = dict(
    torso_n=6,
    # (t, forward shift in units of the end-cap depth, radial scale)
    torso_rings=[("Rc", 0.0, -0.55, 0.62), ("R0", 0.03, 0.0, 1.0), ("R1", 0.34, 0.0, 1.0),
                 ("R2", 0.74, 0.0, 1.0), ("R3", 1.0, 0.6, 0.85)],
    neck_t=0.5,
    # custom tilted rings for a gradual chest -> neck -> skull bend (None = skeleton ring)
    chest_ring=dict(c=[0.115, 0.0, 0.175], n=[0.82, 0.0, 0.57], ru=0.042, rv=0.060),
    neck_ring=dict(c=[0.125, 0.004, 0.238], n=[0.45, -0.18, 0.87], ru=0.038, rv=0.042),
    skull_ring=dict(t=0.20, tilt_up=0.55),
    tail_on_cap=True,
    head_n=8, head_ts=[0.22, 0.55],
    leg_mid={"leg_FR": 0.68, "leg_FL": 0.673, "leg_HR": 0.65, "leg_HL": 0.648},
    tail_ts=[0.12, 0.42, 0.72],
)

PAW_LM = {"leg_FR": "paw_FR", "leg_FL": "paw_FL", "leg_HR": "paw_HR", "leg_HL": "paw_HL"}


def custom_ring(c, nrm, ru, rv, n, phase=0.0, u_hint=(0.0, 1.0, 0.0)):
    """Ellipse ring: c + ru cos(th) u + rv sin(th) v, u ~ u_hint projected off nrm, v = n x u."""
    nrm = np.asarray(nrm, float); nrm /= np.linalg.norm(nrm)
    u = np.asarray(u_hint, float); u = u - u.dot(nrm) * nrm; u /= np.linalg.norm(u)
    v = np.cross(nrm, u)
    th = phase + 2 * PI * np.arange(n) / n
    return np.asarray(c, float) + ru * np.cos(th)[:, None] * u + rv * np.sin(th)[:, None] * v


def build_welded(cfg, Builder, ring_pts, budget=200, stages=None):
    """-> catmesh-1 dict.  `stages`: optional list that receives (label, partial dict)."""
    sk = load_skeleton()
    lm3 = json.loads((ROOT / "proxy" / "landmarks3d.json").read_text())["landmarks"]
    b = Builder()
    torso, neck, head = sk["torso"], sk["neck"], sk["head"]

    def snap(label):
        if stages is not None:
            stages.append((label, {"vertices": [list(map(float, v)) for v in b.V],
                                   "triangles": [list(map(int, t)) for t in b.T],
                                   "tri_region": list(b.reg)}))

    def bridge(A, B, region, material="fur_orange"):
        V = np.asarray(b.V)
        Bal = list(topo._align_ring(np.asarray(A), np.asarray(B), V, V))
        if len(A) == len(Bal):
            b.band(list(A), Bal, region, "cat", material)
        else:
            b.band_uneq(list(A), Bal, region, "cat", material)

    # ---- torso tube (6 sides, phase 0: k*60 deg; k=0 cat's left, k=3 cat's right) ---------
    n = cfg["torso_n"]
    R = {}
    for name, t, fwd, sc in cfg["torso_rings"]:
        sh = torso.tangent(t) * fwd * torso.end_cap_depth("end" if t >= 0.5 else "start")
        if name == "R3" and cfg.get("chest_ring"):
            cr = cfg["chest_ring"]
            R[name] = b.add_verts(custom_ring(cr["c"], cr["n"], cr["ru"], cr["rv"], n))
        else:
            R[name] = b.add_verts(ring_pts(torso, t, n, 0.0, shift=sh, scale=sc))
    # holes (quads left out): tail on top of the rump, legs on the lower flanks
    holes = {("R0", "R1"): [3, 5], ("R2", "R3"): [3, 5]}
    if not cfg.get("tail_on_cap"):
        holes[("Rc", "R0")] = [1]
    hole_loops = {}
    order = [r[0] for r in cfg["torso_rings"]]
    for a, c in zip(order[:-1], order[1:]):
        skip = holes.get((a, c), [])
        A, C = R[a], R[c]
        for i in range(n):
            if i in skip:
                continue
            _quad(b, A[i], A[(i + 1) % n], C[(i + 1) % n], C[i], "torso")
    if cfg.get("tail_on_cap"):      # upper half of the rump end cap is the tail socket
        hole_loops["tail"] = [R["Rc"][0], R["Rc"][1], R["Rc"][2], R["Rc"][3]]
    else:
        hole_loops["tail"] = [R["Rc"][1], R["Rc"][2], R["R0"][2], R["R0"][1]]
    hole_loops["leg_HR"] = [R["R0"][3], R["R0"][4], R["R1"][4], R["R1"][3]]
    hole_loops["leg_HL"] = [R["R0"][5], R["R0"][0], R["R1"][0], R["R1"][5]]
    hole_loops["leg_FR"] = [R["R2"][3], R["R2"][4], R["R3"][4], R["R3"][3]]
    hole_loops["leg_FL"] = [R["R2"][5], R["R2"][0], R["R3"][0], R["R3"][5]]
    if cfg.get("tail_on_cap"):                               # lower half of the rump cap
        b.fan([R["Rc"][3], R["Rc"][4], R["Rc"][5], R["Rc"][0]][::-1], "torso", "cat")
    else:
        b.fan(R["Rc"][::-1], "torso", "cat")                 # rump cap
    snap("torso rings with limb and tail sockets")

    # ---- neck and head ------------------------------------------------------------------
    if cfg.get("neck_ring"):
        nr = cfg["neck_ring"]
        N1 = b.add_verts(custom_ring(nr["c"], nr["n"], nr["ru"], nr["rv"], n))
    else:
        N1 = b.add_verts(ring_pts(neck, cfg["neck_t"], n, 0.0))
    bridge(R["R3"], N1, "neck")
    hn = cfg["head_n"]
    if cfg.get("skull_ring"):       # head ring tilted part-way towards the neck direction
        sr = cfg["skull_ring"]
        hu, hv, hnrm = head.frame(sr["t"])
        ru_, rv_, _ = head.radii(sr["t"])
        nrm = hnrm + sr["tilt_up"] * np.array([0.0, 0.0, 1.0])
        H1 = b.add_verts(custom_ring(head.center(sr["t"]), nrm, ru_, rv_, hn, PI / hn, u_hint=hu))
    else:
        H1 = b.add_verts(ring_pts(head, cfg["head_ts"][0], hn, PI / hn))
    H2 = b.add_verts(ring_pts(head, cfg["head_ts"][1], hn, PI / hn))
    bridge(N1, H1, "neck")
    ear_quads = {"L": 0, "R": 2}          # head ring sides k..k+1 under each ear
    for i in range(hn):
        if i in ear_quads.values():
            continue
        _quad(b, H1[i], H1[(i + 1) % hn], H2[(i + 1) % hn], H2[i], "head")
    nose_v = b.add_verts(np.asarray(lm3["nose"]) + head.tangent(1.0) * 0.004)[0]
    b.cone(H2, nose_v, "head", "cat", "fur_cream")
    snap("neck and head")

    # ---- ears: pyramids on head quads ---------------------------------------------------
    ear_tip_v = {}
    for side, k in ear_quads.items():
        loop = [H1[k], H1[k + 1], H2[k + 1], H2[k]]
        tip = b.add_verts(np.asarray(lm3[f"ear_tip_{side}"]))[0]
        ear_tip_v[side] = tip
        b.cone(loop, tip, "ears", "cat")
    snap("ears")

    # ---- legs ---------------------------------------------------------------------------
    sole_tris = {}
    for name in ("leg_HR", "leg_HL", "leg_FR", "leg_FL"):
        leg = sk[name]
        mid = b.add_verts(ring_pts(leg, cfg["leg_mid"][name], 4, 0.0))
        bridge(hole_loops[name], mid, name)
        paw = np.asarray(lm3[PAW_LM[name]], float)
        hind = name.startswith("leg_H")
        half_w = 0.014 if hind else 0.015
        toe, heel = (0.024, 0.018) if hind else (0.026, 0.016)
        pts = []
        for a in 2 * PI * np.arange(4) / 4:              # same angular order as `mid`
            cu, cv = np.cos(a), np.sin(a)
            pts.append([paw[0] + (toe if cv > 0 else heel) * cv, paw[1] + half_w * cu, 0.0])
        sole = b.add_verts(np.asarray(pts))
        b.band(mid, sole, name, "cat")
        k0 = len(b.T)
        b.fan(sole[::-1], name, "cat", "fur_cream")
        sole_tris[name] = list(range(k0, len(b.T)))
        snap(f"{name}")

    # ---- tail ---------------------------------------------------------------------------
    tail = sk["tail"]
    tr = [b.add_verts(ring_pts(tail, t, 4, PI / 4)) for t in cfg["tail_ts"]]
    bridge(hole_loops["tail"], tr[0], "tail")
    for i in range(len(tr) - 1):
        bridge(tr[i], tr[i + 1], "tail")
    tail_tip = b.add_verts(np.asarray(lm3["tail_tip"]))[0]
    b.cone(tr[-1], tail_tip, "tail", "cat")
    snap("tail")

    V = np.asarray(b.V)
    T = np.asarray(b.T)
    d = meshio.to_dict(V, T, b.reg, b.mat, dict(lm3), budget, name=f"cat_{budget}",
                       notes="variants/200/build.py alloc W (welded single manifold)")
    d = topo.orient_outward(d)

    # ---- two edge splits: chin (jaw point) and brisket ------------------------------------
    hb = (5, 6)                                               # bottom edge of the head rings
    d = topo.edge_split(d, H2[hb[0]], H2[hb[1]], pos=np.asarray(lm3["chin"], float))
    chin_v = len(d["vertices"]) - 1
    r3 = R["R3"]
    p = 0.5 * (np.asarray(d["vertices"][r3[4]]) + np.asarray(d["vertices"][r3[5]]))
    d = topo.edge_split(d, r3[4], r3[5], pos=p + np.array([0.012, 0.0, -0.006]))
    snap_d = d
    if stages is not None:
        stages.append(("jaw and brisket splits", {k: snap_d[k] for k in ("vertices", "triangles", "tri_region")}))
    V = np.asarray(d["vertices"], float)
    T = np.asarray(d["triangles"], int)
    reg = np.asarray(d["tri_region"])

    # ---- materials -----------------------------------------------------------------------
    Vt = V[T]
    nrm = np.cross(Vt[:, 1] - Vt[:, 0], Vt[:, 2] - Vt[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    mat = list(d["tri_material"])
    hax = head.tangent(1.0)
    for i in range(len(T)):
        if reg[i] in ("torso", "neck") and nrm[i, 2] < -0.5:
            mat[i] = "fur_cream"
        if reg[i] == "ears" and np.dot(nrm[i], hax) > 0.3:
            mat[i] = "ear_inner"
    d["tri_material"] = mat

    # ---- landmarks (CONTRACT §12) ------------------------------------------------------
    def tris_of(*regions):
        return [i for i in range(len(T)) if reg[i] in regions]
    att = {"ear_tip_R": {"vertex": int(ear_tip_v["R"])}, "ear_tip_L": {"vertex": int(ear_tip_v["L"])},
           "nose": {"vertex": int(nose_v)}, "tail_tip": {"vertex": int(tail_tip)},
           "chin": {"vertex": int(chin_v)}}
    for name, lmn in PAW_LM.items():
        ids = [i for i in tris_of(name) if np.allclose(V[T[i], 2], 0.0)]
        att[lmn] = meshio.attach_point(np.asarray(lm3[lmn]), V, T, ids)
    t1 = set(tr[0])
    att["tail_base"] = meshio.attach_point(np.asarray(lm3["tail_base"]), V, T,
                                           [i for i in tris_of("tail") if set(T[i]) & t1])
    for lmn, regions in (("eye_R", ("head",)), ("eye_L", ("head",)), ("tail_mid", ("tail",)),
                         ("shoulder_R", ("torso", "leg_FR")), ("hip_R", ("torso", "leg_HR")),
                         ("hock_HR", ("leg_HR",))):
        att[lmn] = meshio.attach_point(np.asarray(lm3[lmn], float), V, T, tris_of(*regions))
    d["landmark_attach"] = att
    meshio.sync_landmarks(d)

    # ---- checks: one closed manifold, exact budget ------------------------------------------
    ec = Counter()
    for t in T:
        for a, c in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            ec[(min(a, c), max(a, c))] += 1
    bad = {e: c for e, c in ec.items() if c != 2}
    assert not bad, f"not a closed manifold: {len(bad)} bad edges, e.g. {list(bad.items())[:4]}"
    assert len(T) == budget, f"triangle count {len(T)} != {budget}"
    assert len(V) == budget // 2 + 2, f"vertex count {len(V)}"
    d["components"] = {"cat": int(len(T))}
    return d


def _quad(b, a0, a1, c1, c0, region, material="fur_orange"):
    """Quad a0-a1-c1-c0 as two triangles, split along the shorter diagonal."""
    V = np.asarray(b.V)
    if np.linalg.norm(V[a0] - V[c1]) <= np.linalg.norm(V[a1] - V[c0]):
        b.tri(a0, a1, c1, region, material, "cat"); b.tri(a0, c1, c0, region, material, "cat")
    else:
        b.tri(a0, a1, c0, region, material, "cat"); b.tri(a1, c1, c0, region, material, "cat")
