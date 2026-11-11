#!/usr/bin/env python
"""Budget-1000 fused manifold (route A): designed lofts -> Blender EXACT boolean union ->
sliver collapses / splits to exactly 1000 triangles -> labels, materials and landmarks
carried over from the source lofts -> catmesh-1.

    python3 variants/1000/fuse.py --design f1 --out variants/1000/checkpoints/v3_fused/built.json

Steps
 1. build.build(design, "curv", "v1"): closed lofts per part (eye sockets, nose fan,
    ear_inner faces, flat soles), ABOVE budget.
 2. fuse_blender.py in background Blender (CPU only, no render): sequential EXACT union
    into the torso, merge-by-distance 1.5 mm, dissolve degenerates, triangulate.
 3. Count to exactly 1000: while above, collapse the shortest edge (boolean slivers
    first) subject to the link condition, no normal flip > 60 deg, no new edge < 2 mm,
    feature vertices (ear/tail/nose tips, eye sockets) kept in place; while below, split
    the longest edge of the head/neck/leg regions.  Every edit goes through catlib.topo.
 4. Junction fillets (--fillet): vertices created by the union (not present in any
    source loft) and their 1-ring are moved half-way to the proxy surface
    (catlib.proxy.closest_surface), which rounds the union creases.
 5. Region/material per triangle = those of the nearest source-loft triangle; eye/pupil/
    nose recomputed on the fused surface around the source feature vertices; landmarks
    re-attached to the same anatomical features (CONTRACT §12).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # the case directory (examples/cat)
import os as _os  # noqa: E402
_os.environ["LOWPOLY_CASE"] = str(ROOT)  # catlib.paths reads the case from here
sys.path.insert(0, _os.environ.get("LOWPOLY_SCRIPTS",
                                   str(ROOT.parents[1] / "skills" / "image-to-lowpoly" / "scripts")))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from catlib import meshio, paths, proxy, topo  # noqa: E402
import build as B  # noqa: E402

BUDGET = 1000
REGION_W = {"torso": 1.0, "tail": 1.6, "neck": 2.0, "leg_FR": 2.0, "leg_FL": 2.0, "leg_HR": 2.0,
            "leg_HL": 2.0, "head": 3.0, "eyes": np.inf, "nose": np.inf, "ears": np.inf}
SRC_TREE = None
CAM_LOC = None
HERE = Path(__file__).resolve().parent


def tri_normals(V, T):
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)


def edges_of(T):
    E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    return np.unique(E, axis=0)


def collapse_ok(V, T, a, b, p, min_edge=0.002, max_turn=60.0):
    fan = np.flatnonzero((T == a).any(1) | (T == b).any(1))
    keep = [i for i in fan if not ((T[i] == a).any() and (T[i] == b).any())]
    if not keep:
        return False
    Tk = T[keep].copy()
    n0 = tri_normals(V, Tk)
    V2 = V.copy(); V2[a] = p; V2[b] = p
    Tk2 = np.where(Tk == b, a, Tk)
    n1 = tri_normals(V2, Tk2)
    if (np.einsum("ij,ij->i", n0, n1) < np.cos(np.radians(max_turn))).any():
        return False
    area = 0.5 * np.linalg.norm(np.cross(V2[Tk2[:, 1]] - V2[Tk2[:, 0]], V2[Tk2[:, 2]] - V2[Tk2[:, 0]]), axis=1)
    if (area < 2e-6).any():
        return False
    for t in Tk2:
        for i in range(3):
            if np.linalg.norm(V2[t[i]] - V2[t[(i + 1) % 3]]) < min_edge and a in (t[i], t[(i + 1) % 3]):
                return False
    return True


def to_budget_n(d, protected, target, log):
    """Collapse shortest edges / split longest edges until exactly BUDGET triangles."""
    tries = 0
    while len(d["triangles"]) > target:
        V = np.asarray(d["vertices"]); T = np.asarray(d["triangles"])
        E = edges_of(T)
        L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
        # cost = length x region weight; union slivers first, flanks next, ears never
        reg = np.asarray(d["tri_region"])
        vw = np.ones(len(V))
        for i, t in enumerate(T):
            w = REGION_W.get(reg[i], 1.0)
            vw[t] = np.maximum(vw[t], w)
        newv = SRC_TREE.query(V)[0] > 1e-4
        cost = L * np.maximum(vw[E[:, 0]], vw[E[:, 1]])
        cost = np.where(newv[E[:, 0]] | newv[E[:, 1]], 0.3 * L, cost)
        done = False
        for k in np.argsort(cost):
            if not np.isfinite(cost[k]):
                break
            a, b = int(E[k, 0]), int(E[k, 1])
            if a in protected and b in protected:
                continue
            if b in protected or (V[b, 2] == 0.0 and V[a, 2] != 0.0):
                a, b = b, a
            p = V[a] if (a in protected or V[a, 2] == 0.0) else 0.5 * (V[a] + V[b])
            if not collapse_ok(V, T, a, b, p):
                continue
            try:
                d2 = topo.edge_collapse(d, a, b, pos=p)
            except ValueError:
                continue
            # vertex b removed: indices > b shift down by one
            protected = {x - (x > b) for x in protected}
            d = d2
            log.append(("collapse", float(L[k])))
            done = True
            break
        tries += 1
        if not done:
            raise RuntimeError("no admissible collapse left")
    return d, protected


def ground_point_of_uv(cam, uv):
    """World point where the camera ray through pixel uv meets z = 0."""
    o, dvec = cam.pixel_rays(np.asarray([uv], float))
    o = np.asarray(o, float).reshape(-1, 3)[0]; dvec = np.asarray(dvec, float).reshape(-1, 3)[0]
    return o + dvec * (-o[2] / dvec[2])


def stage(sdir, name, V, T, reg):
    if sdir is None:
        return None
    p = sdir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"vertices": np.round(np.asarray(V, float), 7).tolist(),
                             "triangles": np.asarray(T).tolist(), "tri_region": list(reg)}))
    return str(p.relative_to(sdir))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", default="f1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge", type=float, default=0.0015)
    ap.add_argument("--fillet", type=float, default=0.5, help="blend factor towards proxy surface for union-created vertices (0 = off)")
    ap.add_argument("--paw-align", action="store_true",
                    help="before the union, slide each lower leg (weight 1 at the ground, 0 at 10 cm) so the "
                         "sole centre sits on the ground point under the photo's paw landmark")
    ap.add_argument("--body-only", action="store_true",
                    help="leave head and ears out; the neck loft's top tip fan is removed after the "
                         "union, leaving ONE open neck ring for the shared head kit (writes body.json, "
                         "not a catmesh-1)")
    ap.add_argument("--target", type=int, default=BUDGET,
                    help="triangle count to reduce to (body-only: count of the OPEN body)")
    ap.add_argument("--kit", default=None,
                    help="head-kit json (with --body-only): the neck loft's top ring is placed ON the kit's "
                         "neck_loop before the union and welded 1:1 afterwards (no bridge band); writes a "
                         "full catmesh-1 at --out and <out>.freeze.json (kit vertices except the loop)")
    ap.add_argument("--stages-dir", default=None, help="CONTRACT §13 construction stages (build order)")
    a = ap.parse_args()
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    sdir = Path(a.stages_dir).resolve() if a.stages_dir else None
    stages = []
    from catlib.camera import Camera
    from scipy.spatial import cKDTree
    cam = Camera.from_json()
    ref_lm = json.loads((ROOT / "reference/derived/landmarks.json").read_text())["landmarks"]

    src = B.build(a.design, "curv", "v1")
    SV = np.asarray(src["vertices"]); ST = np.asarray(src["triangles"])
    meta = src["build_meta"]
    vpart = {}
    for k, rings in meta["rings"].items():
        for r in rings:
            for x in r:
                vpart[x] = k
        for x in meta["tips"][k]:
            vpart[x] = k
    vpart_a = np.array([vpart[i] for i in range(len(SV))])
    tpart = np.array([vpart[t[0]] for t in ST])
    sole_target = {}
    if a.paw_align:
        for leg in ("FR", "FL", "HR", "HL"):
            vs = np.flatnonzero(vpart_a == f"leg_{leg}")
            sole = vs[SV[vs, 2] < 0.004]
            cur = SV[sole].mean(0); cur[2] = 0.0
            tgt = ground_point_of_uv(cam, ref_lm[f"paw_{leg}"]["uv"])
            sh = tgt - cur; sh[2] = 0.0
            w = np.clip(1.0 - SV[vs, 2] / 0.10, 0.0, 1.0)
            SV[vs] += w[:, None] * sh
            sole_target[leg] = tgt
            print(f"paw_{leg}: sole shift {np.round(sh * 1000, 1)} mm")
    kit = None
    if a.kit:
        assert a.body_only, "--kit needs --body-only"
        kit = json.loads((ROOT / a.kit).read_text() if not Path(a.kit).is_absolute() else Path(a.kit).read_text())
        KV = np.asarray(kit["vertices"], float); KL = list(kit["neck_loop"])
        top = meta["rings"]["neck"][-1]
        assert len(top) == len(KL), (len(top), len(KL))
        best = None
        for direction in (1, -1):
            for sh in range(len(KL)):
                order_k = [KL[(direction * i + sh) % len(KL)] for i in range(len(KL))]
                cost = np.linalg.norm(SV[top] - KV[order_k], axis=1).sum()
                if best is None or cost < best[0]:
                    best = (cost, order_k)
        kit_order = best[1]                       # kit loop vertex matched to top[i]
        SV[top] = KV[kit_order]
        # the ring below the top becomes a copy of the kit loop's shape (polar interpolation
        # in the loop plane), 1.12x wider and 16 mm further from the head along the loop
        # normal: the neck leaves the jaw/skull base along the head's own surface direction
        # (no collar, no neck band folding into the jaw)
        pen = meta["rings"]["neck"][-2]
        P = KV[KL]; cl = P.mean(0)
        nrm = np.linalg.svd(P - cl)[2][2]
        into = KV.mean(0) - cl
        if nrm @ into > 0:
            nrm = -nrm                                   # nrm points away from the head
        e1 = P[0] - cl; e1 -= (e1 @ nrm) * nrm; e1 /= np.linalg.norm(e1); e2 = np.cross(nrm, e1)
        angL = np.arctan2((P - cl) @ e2, (P - cl) @ e1)
        o = np.argsort(angL); aL = angL[o]; PL = P[o]
        aL = np.concatenate([aL - 2 * np.pi, aL, aL + 2 * np.pi]); PL = np.concatenate([PL, PL, PL])
        for v in pen:
            q = SV[v] - cl
            ang = np.arctan2(q @ e2, q @ e1)
            k = np.searchsorted(aL, ang); w = (ang - aL[k - 1]) / (aL[k] - aL[k - 1])
            lp = PL[k - 1] * (1 - w) + PL[k] * w
            SV[v] = cl + 1.12 * (lp - cl) + 0.016 * nrm
        # head socket: torso/neck loft vertices that would sit inside the kit head (on the
        # head side of the loop plane, within 1.4 loop radii of its axis) are pushed back
        # to 12 mm below that plane, so the chest top does not swallow the kit's throat
        rl = np.linalg.norm(P - cl, axis=1).mean()
        keepset = set(top) | set(pen) | {meta["tips"]["neck"][1]}
        nsock = 0
        for v in np.flatnonzero(np.isin(vpart_a, ["torso", "neck"])):
            if v in keepset:
                continue
            q = SV[v] - cl
            h_ = -(q @ nrm)                              # height on the head side
            rad = np.linalg.norm(q + h_ * nrm)
            if h_ > -0.012 and rad < 1.4 * rl:
                SV[v] = SV[v] + (h_ + 0.012) * nrm
                nsock += 1
        print(f"head socket: {nsock} torso/neck loft vertices pushed below the loop plane")
    src["vertices"] = SV.tolist()
    order = ["torso", "neck", "head", "ear_R", "ear_L", "leg_FR", "leg_FL", "leg_HR", "leg_HL", "tail"]
    if a.body_only:
        order = [k for k in order if k not in ("head", "ear_R", "ear_L")]
    keep_t = np.isin(tpart, order)
    parts = {}
    cumV, cumT, cumR = [], [], []
    for n_, k in enumerate(order):
        ids = np.flatnonzero(tpart == k)
        vs = np.unique(ST[ids])
        remap = {int(v): i for i, v in enumerate(vs)}
        parts[k] = {"V": SV[vs].tolist(), "T": [[remap[int(x)] for x in ST[i]] for i in ids]}
        base = sum(len(x) for x in cumV)
        cumV.append(SV[vs]); cumT.append(np.asarray(parts[k]["T"]) + base)
        cumR += [src["tri_region"][i] for i in ids]
        r_ = stage(sdir, f"construct/{n_ + 1:02d}_{k}.json", np.concatenate(cumV), np.concatenate(cumT), cumR)
        if r_:
            stages.append({"label": f"{k} loft", "kind": "construct", "mesh": r_})
    with tempfile.TemporaryDirectory() as td:
        pin = Path(td) / "parts.json"; pout = Path(td) / "fused.json"
        pin.write_text(json.dumps({"parts": parts, "order": order}))
        r = subprocess.run([str(paths.BLENDER), "-b", "--factory-startup", "--python", str(HERE / "fuse_blender.py"),
                            "--", str(pin), str(pout), str(a.merge)], capture_output=True, text=True)
        (out.parent / "fuse_blender.log").write_text(r.stdout + r.stderr)
        if r.returncode != 0 or not pout.exists():
            raise SystemExit("blender fuse failed; see fuse_blender.log")
        f = json.loads(pout.read_text())
    V = np.asarray(f["vertices"], float); T = np.asarray(f["triangles"], np.int64)
    V[V[:, 2] < 0.001, 2] = 0.0
    n_union = len(T)

    # labels from the nearest source triangle (centroid distance, normal-compatible)
    STk = ST[keep_t]
    SC = SV[STk].mean(1); SN = tri_normals(SV, STk)
    src_reg = [r for r, k in zip(src["tri_region"], keep_t) if k]
    src_mat = [m for m, k in zip(src["tri_material"], keep_t) if k]
    C = V[T].mean(1); N = tri_normals(V, T)
    _, nn = cKDTree(SC).query(C, k=6)
    pick = np.array([nn[i][np.argmax(SN[nn[i]] @ N[i])] for i in range(len(T))])
    reg = [src_reg[j] for j in pick]
    mat = [src_mat[j] for j in pick]
    for i in range(len(T)):
        if mat[i] in ("eye", "pupil", "nose"):
            mat[i] = "fur_orange"
        if reg[i] in ("eyes", "nose"):
            reg[i] = "head"
    r_ = stage(sdir, f"construct/{len(order) + 1:02d}_union.json", V, T, reg)
    if r_:
        stages.append({"label": "boolean union", "kind": "construct", "mesh": r_})
    d = meshio.to_dict(V, T, reg, mat, {}, BUDGET, name="cat_1000")

    att0 = src["landmark_attach"]
    fnames = ["tail_tip"] + ([] if a.body_only else ["ear_tip_R", "ear_tip_L", "nose", "eye_R", "eye_L"])
    def nearest(q):
        return int(np.argmin(np.linalg.norm(np.asarray(d["vertices"]) - q, axis=1)))
    featpos = {k: SV[att0[k]["vertex"]] for k in fnames}
    neck_pos = []
    if a.body_only:
        top = meta["rings"]["neck"][-1]
        neck_pos = [SV[x] for x in top] + [SV[meta["tips"]["neck"][1]]]
    protected = {nearest(q) for q in list(featpos.values()) + neck_pos}
    global SRC_TREE
    SRC_TREE = cKDTree(SV[np.unique(STk)])
    log = []
    target = a.target + (len(neck_pos) - 1 if a.body_only else 0)
    if target < len(d["triangles"]):
        d, protected = to_budget_n(d, protected, target, log)
    V = np.asarray(d["vertices"]); T = np.asarray(d["triangles"])
    feat = {k: nearest(q) for k, q in featpos.items()}
    r_ = stage(sdir, f"construct/{len(order) + 2:02d}_collapsed.json", V, T, d["tri_region"])
    if r_:
        stages.append({"label": f"{sum(1 for x in log if x[0] == 'collapse')} sliver/flank collapses", "kind": "construct", "mesh": r_})

    # junction fillets: union-created vertices (far from every source vertex) + 1-ring
    n_fillet = 0
    if a.fillet > 0:
        dsrc, _ = cKDTree(SV).query(V)
        new = dsrc > 1e-4
        ring = new.copy()
        for t in T:
            if new[t].any():
                ring[t] = True
        ring &= V[:, 2] > 0.0
        ring[list(feat.values())] = False
        if a.body_only:
            ring[[nearest(q) for q in neck_pos]] = False
        idx = np.flatnonzero(ring)
        if len(idx):
            q = proxy.closest_surface(V[idx])[0]
            V[idx] = V[idx] + a.fillet * (np.asarray(q) - V[idx])
        d["vertices"] = V.tolist()
        n_fillet = int(len(idx))

    reg = list(d["tri_region"]); mat = list(d["tri_material"])
    N = tri_normals(V, T)
    lm3 = proxy.landmarks3d()
    global CAM_LOC
    CAM_LOC = np.asarray(cam.location, float)
    att = {}
    if not a.body_only:
        tipn = feat["nose"]
        for j in np.flatnonzero((T == tipn).any(1)):
            if N[j, 2] > -0.35:
                mat[j] = "nose"; reg[j] = "nose"
        att["nose"] = {"vertex": tipn}
        for side in ("R", "L"):
            vi = feat[f"eye_{side}"]
            fan = np.flatnonzero((T == vi).any(1))
            camdir = CAM_LOC - V[vi]; camdir /= np.linalg.norm(camdir)
            facing = fan[np.argsort(-(N[fan] @ camdir))][:3]
            for j in facing:
                mat[j] = "eye"; reg[j] = "eyes"
            mat[facing[0]] = "pupil"
            att[f"eye_{side}"] = {"vertex": vi}
        att["ear_tip_R"] = {"vertex": feat["ear_tip_R"]}
        att["ear_tip_L"] = {"vertex": feat["ear_tip_L"]}
    att["tail_tip"] = {"vertex": feat["tail_tip"]}
    reg_a = np.asarray(reg)
    fz = (("tail_base", "tail"), ("tail_mid", "tail"), ("hock_HR", "leg_HR"), ("shoulder_R", "torso"), ("hip_R", "torso"))
    if not a.body_only:
        fz += (("chin", "head"),)
    for k, rg in fz:
        # proxy feature point (the loft's own tail_base sat on a ring the union buried)
        q = np.asarray(lm3[k], float)
        att[k] = meshio.attach_point(q, V, T, np.flatnonzero(reg_a == rg))
    for leg in ("FR", "FL", "HR", "HL"):
        ids = np.flatnonzero(reg_a == f"leg_{leg}")
        sole = [j for j in ids if N[j, 2] < -0.3]
        sv = np.unique(T[sole]) if sole else np.unique(T[ids])
        sv = sv[V[sv, 2] == 0.0] if (V[sv, 2] == 0.0).any() else sv
        centre = V[sv].mean(0)          # centre of this paw's ground footprint
        att[f"paw_{leg}"] = meshio.attach_point(centre, V, T, sole or ids)
    d["tri_region"] = reg; d["tri_material"] = mat
    d["landmark_attach"] = att
    d["landmarks"] = {}
    meshio.sync_landmarks(d)
    note = (f"budget-1000 fused: lofts design={a.design} paw_align={a.paw_align} -> Blender EXACT union "
            f"({n_union} tris) -> {sum(1 for x in log if x[0]=='collapse')} collapses -> fillet {a.fillet} on {n_fillet} verts")
    d.pop("build_meta", None); d.pop("groups", None)
    if a.body_only:
        # open the neck: drop the neck top tip and its fan -> one open ring
        tip = nearest(neck_pos[-1])
        ringv = [nearest(q) for q in neck_pos[:-1]]
        fan = (np.asarray(d["triangles"]) == tip).any(1)
        assert fan.sum() == len(ringv), (fan.sum(), len(ringv))
        T2 = np.asarray(d["triangles"])[~fan]
        reg2 = [r_ for r_, f_ in zip(reg, fan) if not f_]
        mat2 = [m_ for m_, f_ in zip(mat, fan) if not f_]
        V2 = np.delete(V, tip, 0)
        T2 = T2 - (T2 > tip)
        ringv = [x - (x > tip) for x in ringv]
        # attachments: re-attach on the opened mesh
        att2 = {}
        for k, at in att.items():
            p_ = meshio.landmarks_from_attach(V, np.asarray(d["triangles"]), {k: at})[k]
            att2[k] = {"vertex": int(np.argmin(np.linalg.norm(V2 - p_, axis=1)))} if "vertex" in at \
                else meshio.attach_point(p_, V2, T2, [j for j, r_ in enumerate(reg2) if r_ == reg[at["tri"]]])
        body = {"format": "catbody-1", "budget_open_body": len(T2), "name": "cat_1000_body",
                "vertices": V2.tolist(), "triangles": T2.tolist(), "tri_region": reg2, "tri_material": mat2,
                "landmark_attach": att2,
                "landmarks": {k: v.tolist() for k, v in meshio.landmarks_from_attach(V2, T2, att2).items()},
                "neck_ring": [int(x) for x in ringv],
                "neck_ring_note": "open boundary loop (cyclic order of the neck loft ring, theta = 0 at the "
                                  "ring's u axis); bridge to the head kit's neck_loop",
                "notes": note + f"; neck tip fan removed -> {len(T2)} tris, open neck ring of {len(ringv)}"}
        out.write_text(json.dumps(body))
        if kit is not None:
            # 1:1 weld: kit loop vertex kit_order[i] == body ring vertex ringv[i] (same position)
            nb = len(V2); ntb = len(T2)
            KV = np.asarray(kit["vertices"], float)
            dev = np.abs(V2[ringv] - KV[kit_order]).max()
            print(f"ring deviation from kit loop after fusion: {dev * 1000:.4f} mm")
            assert dev < 1e-4, "ring moved during fusion"
            V2[ringv] = KV[kit_order]          # float32 round-trip through Blender
            kmap = np.full(len(KV), -1)
            for bi, ki in zip(ringv, kit_order):
                kmap[ki] = bi
            rest = [i for i in range(len(KV)) if kmap[i] < 0]
            kmap[rest] = nb + np.arange(len(rest))
            Vf = np.vstack([V2, KV[rest]])
            Tf = np.vstack([T2, kmap[np.asarray(kit["triangles"])]])
            regf = reg2 + list(kit["tri_region"]); matf = mat2 + list(kit["tri_material"])
            attf = {k: v for k, v in att2.items() if k not in kit["landmark_attach"]}
            for k, at in kit["landmark_attach"].items():
                attf[k] = {"vertex": int(kmap[at["vertex"]])} if "vertex" in at else \
                    {"tri": int(at["tri"]) + ntb, "bary": at["bary"]}
            df = meshio.to_dict(Vf, Tf, regf, matf, {}, BUDGET, name="cat_1000",
                                notes=note + f"; open body {ntb} tris welded 1:1 to head kit {kit.get('name', a.kit)} "
                                             f"(lod {kit.get('lod')}, {len(kit['triangles'])} tris) on its neck_loop",
                                landmark_attach=attf)
            assert len(Tf) == BUDGET, len(Tf)
            kitout = out.with_name(out.stem + "_full.json")
            meshio.save(kitout, df)
            freeze = sorted(int(x) for x in kmap if x >= nb)          # kit vertices except the loop
            out.with_name(out.stem + "_full.freeze.json").write_text(json.dumps(freeze))
            r_ = stage(sdir, f"construct/{len(order) + 3:02d}_head_weld.json", Vf, Tf, regf)
            if r_:
                stages.append({"label": "head kit L4 welded 1:1 on the neck ring", "kind": "construct", "mesh": r_})
            print(json.dumps({"full": str(kitout), "triangles": len(Tf), "frozen": len(freeze)}))
        print(json.dumps({"out": str(out), "union": n_union, "open_body_tris": len(T2), "neck_ring": body["neck_ring"],
                          "by_region": {r_: reg2.count(r_) for r_ in sorted(set(reg2))}}))
    else:
        d["notes"] = note
        assert len(d["triangles"]) == a.target
        meshio.save(out, d)
        print(json.dumps({"out": str(out), "union": n_union, "triangles": len(T),
                          "by_region": {r_: reg.count(r_) for r_ in sorted(set(reg))}}))
    if sdir is not None:
        (sdir / "construct_stages.json").write_text(json.dumps(stages, indent=1))


if __name__ == "__main__":
    main()
