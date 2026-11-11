"""Budget-100 cat, WELDED topology (round 2, checkpoint v3_welded): one closed 2-manifold built
by box-modelling.  Every limb, the tail and both ears are extruded from faces of one closed
spine tube, so there is no buried geometry and no open rim.

    python3 variants/100/build_welded.py --out variants/100/work/W0.json \
        [--stages-dir variants/100/timelapse/construct]

Deterministic; asserts exactly 100 triangles.

Allocation (see NOTES.md):
  spine  40 = rump tip fan 4 + 4 bands x 8 (4-sided box rings: rump, waist, chest, neck, head)
              + muzzle tip fan 4
  legs   30 = FR +6, FL +6 (torso triangle -> sole ring, 1-triangle sole), HR +12 (hock ring),
              HL +6 (straight; it is the far leg, angled forward)
  tail   14 = rump-fan triangle -> ring -> ring -> tip  (6 + 6 + 3 - 1)
  ears   12 = per ear: 2 edge splits on the skull top (+4) cut a small corner triangle at the
              skull corner, replaced by a 3-triangle pyramid to the ear-tip landmark (+2)
  chin    2 = 1 edge split on the neck ring's front edge (throat/chest break)
  belly   2 = 1 edge split on the chest ring's bottom edge (sternum / chest-front break)
  total 100
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
HERE = Path(__file__).resolve().parent
for p in (str(HERE),):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from catlib import meshio  # noqa: E402
from catlib import proxy as cproxy  # noqa: E402
from catlib.skeleton import load_skeleton  # noqa: E402
from build import Builder, frame_from_axis, ring, skel_ring, unit  # noqa: E402

BUDGET = 100
CFG = dict(
    torso_ts=[0.0, 0.40, 1.0], torso_scale=[(1.25, 1.2), (1.3, 1.3), (1.35, 1.3)],
    neck_t=0.55, neck_scale=(1.0, 1.0),
    head_t=0.42, head_scale=(1.3, 1.0), head_drop=0.008,
    rump_depth=0.045, muzzle_push=0.004,
    ear_split=0.3,
)


class Mesh:
    def __init__(self, V, T, reg, mat):
        self.V = [np.asarray(v, float) for v in V]
        self.T = [tuple(int(i) for i in t) for t in T]
        self.reg = list(reg)
        self.mat = list(mat)

    def add_v(self, p):
        self.V.append(np.asarray(p, float))
        return len(self.V) - 1

    def add_t(self, t, r, m="fur_orange"):
        self.T.append(tuple(int(i) for i in t))
        self.reg.append(r)
        self.mat.append(m)
        return len(self.T) - 1

    def remove_t(self, i):
        """Tombstone (indices of other triangles stay valid); compacted in arrays()."""
        t = self.T[i]
        self.T[i] = None
        return t

    def live(self):
        return [i for i, t in enumerate(self.T) if t is not None]

    def new_index(self):
        m, k = {}, 0
        for i, t in enumerate(self.T):
            if t is not None:
                m[i] = k
                k += 1
        return m

    def find_tri(self, verts):
        s = set(verts)
        out = [i for i, t in enumerate(self.T) if t is not None and set(t) == s]
        assert len(out) == 1, (verts, out)
        return out[0]

    def tris_with_edge(self, a, b):
        return [i for i, t in enumerate(self.T) if t is not None and a in t and b in t]

    def edge_split(self, a, b, t=0.5, pos=None):
        """+2 triangles (interior edge); new vertex returned.  Same rule as catlib.topo."""
        p = np.asarray(pos, float) if pos is not None else self.V[a] + t * (self.V[b] - self.V[a])
        m = self.add_v(p)
        for ti in self.tris_with_edge(a, b):
            tri = list(self.T[ti])
            k = [i for i in range(3) if tri[i] in (a, b) and tri[(i + 1) % 3] in (a, b)][0]
            x, y, c = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
            self.T[ti] = (x, m, c)
            self.add_t((m, y, c), self.reg[ti], self.mat[ti])
        return m

    def extrude(self, face, rings, region, end="fan", tip=None, mats=None):
        """Delete triangle `face`; bridge its 3 edges to rings[0], ring to ring, then close
        with a 1-triangle fan ('fan') or a tip vertex ('tip').  Each ring (3,3) is reordered
        to keep the face's orientation along the extrusion direction and rotated for the
        least twist.  Returns {'rings': [[ids]...], 'end_tris': [...], 'bands': [[...]...], 'tip': id}."""
        a, b, c = self.T[face]
        self.remove_t(face)
        prev = [a, b, c]
        out = {"rings": [], "bands": [], "end_tris": []}
        for k, R in enumerate(rings):
            R0 = np.asarray(R, float)
            P = np.array([self.V[i] for i in prev])
            # choose among the 6 vertex assignments the one whose walls (and end cap) face
            # outward best; every assignment is a valid oriented manifold, only the geometry
            # differs (a twisted assignment folds the prism)
            ctr = 0.5 * (P.mean(0) + R0.mean(0))
            d = unit(R0.mean(0) - P.mean(0))
            best, best_s = None, -1e9
            for perm in ([0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1], [2, 1, 0], [1, 0, 2]):
                Rp = R0[perm]
                sc = 0.0
                for jj in range(3):
                    j1 = (jj + 1) % 3
                    for tri in ((P[jj], P[j1], Rp[j1]), (P[jj], Rp[j1], Rp[jj])):
                        nrm = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                        nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
                        sc += float(np.dot(nrm, unit(np.mean(tri, 0) - ctr)))
                ncap = np.cross(Rp[1] - Rp[0], Rp[2] - Rp[0])
                sc += 2.0 * float(np.dot(unit(ncap), d))
                if sc > best_s:
                    best, best_s = Rp, sc
            R = best
            ids = [self.add_v(p) for p in R]
            band = []
            for j in range(3):
                j1 = (j + 1) % 3
                m = mats(k, j) if mats else "fur_orange"
                band.append(self.add_t((prev[j], prev[j1], ids[j1]), region, m))
                band.append(self.add_t((prev[j], ids[j1], ids[j]), region, m))
            out["bands"].append(band)
            out["rings"].append(ids)
            prev = ids
        if end == "fan":
            out["end_tris"].append(self.add_t(tuple(prev), region, "fur_cream"))
        else:
            t = self.add_v(tip)
            out["tip"] = t
            for j in range(3):
                out["end_tris"].append(self.add_t((prev[j], prev[(j + 1) % 3], t), region))
        return out

    def pyramid(self, face, apex, region, inner_test=None):
        a, b, c = self.T[face]
        self.remove_t(face)
        t = self.add_v(apex)
        ids = []
        for x, y in ((a, b), (b, c), (c, a)):
            ids.append(self.add_t((x, y, t), region))
        return t, ids

    def arrays(self):
        L = self.live()
        return np.array(self.V), np.array([self.T[i] for i in L], np.int64)

    def labels(self):
        L = self.live()
        return [self.reg[i] for i in L], [self.mat[i] for i in L]

    def stage(self, path, label):
        V, T = self.arrays()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"label": label, "vertices": V.round(7).tolist(),
                                          "triangles": T.tolist(), "tri_region": self.labels()[0]}))


def tri_ring(c, axis, r, lateral=(0, 1, 0), phase=np.pi / 2):
    u, v, _ = frame_from_axis(axis, lateral)
    return ring(c, u, v, r, r, 3, phase)


def sole_points(paw, fwd_len=0.020, back_len=0.010, half=0.011):
    pl = np.asarray(paw, float).copy()
    pl[2] = 0.0
    fwd = np.array([1.0, 0, 0])
    lat = np.array([0, 1.0, 0])
    pe = pl - fwd * (fwd_len - 2 * back_len) / 3.0       # centroid exactly on the landmark
    return np.array([pe + fwd * fwd_len, pe - fwd * back_len + lat * half, pe - fwd * back_len - lat * half])


def build(stages_dir: Path | None = None) -> dict:
    sk = load_skeleton()
    lm3 = cproxy.landmarks3d()
    T_, N_, H_ = sk["torso"], sk["neck"], sk["head"]
    cfg = CFG
    ph = np.pi / 4
    B = Builder()
    rings = [skel_ring(T_, t, 4, ph, su=s[0], sv=s[1]) for t, s in zip(cfg["torso_ts"], cfg["torso_scale"])]
    rings.append(skel_ring(N_, cfg["neck_t"], 4, ph, su=cfg["neck_scale"][0], sv=cfg["neck_scale"][1]))
    rings.append(skel_ring(H_, cfg["head_t"], 4, ph, su=cfg["head_scale"][0], sv=cfg["head_scale"][1],
                           shift=(0, 0, -cfg["head_drop"])))
    labs = ["torso", "torso", "torso", "head"]
    rump_tip = T_.center(0.0) - T_.tangent(0.0) * cfg["rump_depth"]
    muzzle = lm3["nose"] + H_.tangent(1.0) * cfg["muzzle_push"]

    def mf(kind, k, j, _):
        if kind == "band" and j == 2 and k in (2, 3):
            return "fur_cream"
        if kind == "end" and j == 2:
            return "fur_cream"
        return "fur_orange"
    sp = B.chain(rings, "torso", start="tip", start_tip=rump_tip, end="tip", end_tip=muzzle,
                 regions=labs, mat_fn=mf)
    for tid in sp["end_tris"]:
        B.reg[tid] = "head"
    for tid in sp["bands"][2]:
        B.reg[tid] = "neck"
    M = Mesh(B.V, B.T, B.reg, B.mat)
    if stages_dir:
        M.stage(stages_dir / "01_spine.json", "spine tube (5 box rings, rump + muzzle tips)")
    R = sp["rings"]                    # R[0] rump .. R[2] chest, R[3] neck, R[4] head
    # ring corner index: 0 = top-left (+u+v), 1 = top-right, 2 = bottom-right, 3 = bottom-left
    # ---- legs: bottom quads of band 0 (hips) and band 1 (shoulders) -------------------------
    def bottom_tris(k):
        A, Bn = R[k], R[k + 1]
        # lower flank triangles: each holds both bottom corners of its side quad, so the
        # belly quad stays between the left and right legs (open leg gap, separate legs)
        left = M.find_tri((A[3], Bn[0], Bn[3]))          # cat-left flank (corners 3,0)
        right = M.find_tri((A[1], A[2], Bn[2]))          # cat-right flank (corners 1,2)
        return left, right
    parts = {}
    legs = {}
    hl, hr = bottom_tris(0)
    legs["leg_HL"], legs["leg_HR"] = M.T[hl], M.T[hr]
    # shoulder band: flip the right flank diagonal so FR gets the FRONT-heavy triangle
    # (bottom-rear, bottom-front, top-front) instead of the rear-heavy one
    A, Bn = R[1], R[2]
    i1, i2 = M.find_tri((A[1], A[2], Bn[2])), M.find_tri((A[1], Bn[2], Bn[1]))
    M.T[i1], M.T[i2] = (A[1], A[2], Bn[1]), (A[2], Bn[2], Bn[1])
    hl_, hr_ = legs["leg_HL"], legs["leg_HR"]
    fl = M.find_tri((A[3], Bn[0], Bn[3]))
    fr = M.find_tri((A[2], Bn[2], Bn[1]))
    legs["leg_FL"], legs["leg_FR"] = M.T[fl], M.T[fr]
    order = ["leg_FR", "leg_FL", "leg_HR", "leg_HL"]
    for k, leg in enumerate(order):
        face = M.find_tri(legs[leg])
        part = sk[leg]
        paw = lm3["paw_" + leg.split("_")[1]]
        sole = sole_points(paw)
        rs = []
        if leg == "leg_HR":
            c = part.center(0.55)
            ru, rv, _ = part.radii(0.55)
            rs.append(tri_ring(c, part.tangent(0.55), 0.55 * (ru + rv) * 1.05))
        rs.append(sole)
        parts[leg] = M.extrude(face, rs, leg, end="fan")
        if stages_dir:
            M.stage(stages_dir / f"{2 + k:02d}_{leg}.json", f"extrude {leg}")
    # ---- tail: from the top rump-fan triangle (corners 0, 1 of R[0] + rump tip) --------------
    tail = sk["tail"]
    face = M.find_tri((R[0][0], R[0][1], sp["start_tip"]))
    trs = []
    for t in (0.30, 0.65):
        ru, rv, _ = tail.radii(t)
        trs.append(tri_ring(tail.center(t), tail.tangent(t), 0.5 * (ru + rv) * 1.1))
    tip = tail.center(1.0) + tail.tangent(1.0) * 0.006
    parts["tail"] = M.extrude(face, trs, "tail", end="tip", tip=tip,
                              mats=lambda k, j: "fur_stripe" if k == 1 else "fur_orange")
    if stages_dir:
        M.stage(stages_dir / "06_tail.json", "extrude tail")
    # ---- ears --------------------------------------------------------------------------------
    NBL, NBR = R[3][0], R[3][1]
    TL, TR = R[4][0], R[4][1]
    s = cfg["ear_split"]
    PL = M.edge_split(TL, TR, s)
    PR = M.edge_split(PL, TR, 1 - s / (1 - s))
    QL = M.edge_split(TL, NBL, s)
    QR = M.edge_split(TR, NBL, s)
    for e, (corner, P, Q) in (("ear_L", (TL, PL, QL)), ("ear_R", (TR, PR, QR))):
        face = M.find_tri((corner, P, Q))
        apex = lm3["ear_tip_" + e[-1]]
        t, ids = M.pyramid(face, apex, "ears")
        parts[e] = {"tip": t, "tris": ids}
        # inner ear = the face whose base edge is most forward (towards the face / camera)
        fwd = H_.tangent(0.5)
        best = max(ids, key=lambda i: float(np.dot(np.cross(M.V[M.T[i][1]] - M.V[M.T[i][0]],
                                                            M.V[M.T[i][2]] - M.V[M.T[i][0]]), fwd)
                                            / (np.linalg.norm(np.cross(M.V[M.T[i][1]] - M.V[M.T[i][0]],
                                                                       M.V[M.T[i][2]] - M.V[M.T[i][0]])) + 1e-12)))
        M.mat[best] = "ear_inner"
    if stages_dir:
        M.stage(stages_dir / "07_ears.json", "ears: 2 skull splits + pyramid each")
    # ---- chin / chest breaks -----------------------------------------------------------------
    NFR, NFL = R[3][2], R[3][3]
    chin_pos = 0.5 * (M.V[NFR] + M.V[NFL]) + np.array([0.012, 0, 0.0])
    M.edge_split(NFR, NFL, pos=chin_pos)
    CBR, CBL = R[2][2], R[2][3]
    M.edge_split(CBR, CBL, pos=0.5 * (M.V[CBR] + M.V[CBL]) + np.array([0.008, 0, -0.004]))
    if stages_dir:
        M.stage(stages_dir / "08_chin_chest.json", "chin + chest edge splits")

    V, T = M.arrays()
    reg, mat = M.labels()
    nidx = M.new_index()
    if len(T) != BUDGET:
        raise SystemExit(f"{len(T)} triangles != {BUDGET}: { {r: reg.count(r) for r in sorted(set(reg))} }")
    # closed-manifold check
    from collections import Counter
    ec = Counter()
    for t in T:
        for a_, b_ in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            ec[(a_, b_)] += 1
    assert all(v == 1 for v in ec.values()) and all((b_, a_) in ec for (a_, b_) in ec), "not a closed oriented manifold"
    vol = float((V[T[:, 0]] * np.cross(V[T[:, 1]], V[T[:, 2]])).sum()) / 6
    assert vol > 0, vol

    tris_of = lambda regions: [i for i, r in enumerate(reg) if r in regions]  # noqa: E731
    att = {"ear_tip_R": {"vertex": int(parts["ear_R"]["tip"])},
           "ear_tip_L": {"vertex": int(parts["ear_L"]["tip"])},
           "tail_tip": {"vertex": int(parts["tail"]["tip"])},
           "nose": {"vertex": int(sp["end_tip"])}}
    for nm in ("eye_R", "eye_L", "chin"):
        att[nm] = meshio.attach_point(lm3[nm], V, T, tris_of({"head"}))
    for leg in order:
        att["paw_" + leg.split("_")[1]] = {"tri": nidx[parts[leg]["end_tris"][0]], "bary": [1 / 3, 1 / 3, 1 / 3]}
    tb = [[nidx[i] for i in b if i in nidx] for b in parts["tail"]["bands"]]
    att["tail_base"] = meshio.attach_point(lm3["tail_base"], V, T, tb[0])
    att["tail_mid"] = meshio.attach_point(lm3["tail_mid"], V, T, [i for b in tb for i in b])
    att["shoulder_R"] = meshio.attach_point(lm3["shoulder_R"], V, T, tris_of({"torso"}))
    att["hip_R"] = meshio.attach_point(lm3["hip_R"], V, T, tris_of({"torso"}))
    att["hock_HR"] = meshio.attach_point(lm3["hock_HR"], V, T, tris_of({"leg_HR"}))
    counts = {r: reg.count(r) for r in sorted(set(reg))}
    d = meshio.to_dict(V, T, reg, mat, meshio.landmarks_from_attach(V, T, att), BUDGET, name="cat_100",
                       notes=f"budget-100 welded box-model (one closed manifold); per-region {json.dumps(counts)}",
                       landmark_attach=att)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stages-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    d = build(a.stages_dir)
    assert len(d["triangles"]) == BUDGET
    meshio.save(a.out, d)
    reg = d["tri_region"]
    print(f"wrote {a.out}  triangles={len(d['triangles'])}  vertices={len(d['vertices'])}")
    print({r: reg.count(r) for r in sorted(set(reg))})


if __name__ == "__main__":
    main()
