#!/usr/bin/env python3
"""Fit the head kit to the reference photo through the locked camera (CPU only).

    python3 scripts/fit_head_kit.py [--evals 2400] [--lod-evals 300] [--seed 0]

One rigid pose + uniform scale + symmetric shape parameters shared by all five LODs
(fitted jointly on L2 and L4), then a small per-LOD symmetric adjustment of the outline
parameters (cheeks/skull/ears) regularised toward the shared values.  Symmetry is exact
by construction (only symmetric parameters exist).

Objective: six head landmarks (reprojection, sigma-weighted), symmetric boundary distance
along the head's free outline (ears, between the ears, both cheeks down to the neck/chest
junction), a penalty for head pixels outside the reference mask inside the `head` window,
and anatomical priors (head depth/width 0.9-1.1, ear base thickness 3-14 mm, eyes in front
of the cheeks, shape near the template).

Writes head/pose.json, head/head_L{0..4}.json (world), head/template/head_L*_local.json
(fitted proportions, local frame), head/renders/L*_overlay.png, head/report.json.
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

from catlib import headkit as hk  # noqa: E402
from catlib.camera import Camera  # noqa: E402
from catlib.raster import silhouette  # noqa: E402
from catlib.metrics import boundary  # noqa: E402

LM_NAMES = ("ear_tip_R", "ear_tip_L", "eye_R", "eye_L", "nose", "chin")
LM_SIGMA = {"ear_tip_R": 4.0, "ear_tip_L": 4.0, "eye_R": 2.0, "eye_L": 2.0, "nose": 3.0, "chin": 10.0}

POSE_KEYS = ["yaw_deg", "pitch_deg", "roll_deg", "tx", "ty", "tz", "scale"]
SHAPE_KEYS = ["skull_w", "skull_h", "skull_d", "cheek_w", "cheek_y", "cheek_z", "nose_x", "nose_z",
              "chin_x", "chin_z", "eye_x", "eye_y", "eye_z", "ear_tip_x", "ear_tip_y", "ear_tip_z",
              "ear_in_y", "ear_out_y", "ear_out_z", "ear_d", "neck_w"]
SHAPE_STEP = {"ear_d": 2.0}
LOD_KEYS = ["skull_w", "skull_h", "cheek_w", "cheek_y", "ear_out_y", "ear_in_y", "ear_tip_y", "ear_tip_z"]
BOUNDS = {"ear_d": (6.0, 16.0), "scale": (0.7, 1.4)}


class Target:
    def __init__(self):
        from scipy.ndimage import distance_transform_edt
        from PIL import Image
        der = ROOT / "reference" / "derived"
        self.cam = Camera.from_json(ROOT / "contract" / "camera.json")
        self.mask = np.asarray(Image.open(der / "mask.png").convert("L")) >= 128
        self.win = [int(x) for x in json.loads((der / "regions.json").read_text())["head"]]
        lm = json.loads((der / "landmarks.json").read_text())["landmarks"]
        self.lm = {k: np.array(lm[k]["uv"], float) for k in LM_NAMES}
        H, W = self.mask.shape
        vv, uu = np.mgrid[0:H, 0:W]
        u0, v0, u1, v1 = self.win
        inwin = (uu >= u0) & (uu < u1) & (vv >= v0) & (vv < v1)
        self.inwin = inwin
        # free outline: head outline above the neck/chest junction (image-left cheek meets
        # the back at v~600, image-right cheek runs into the chest below v~640)
        free = inwin & (((uu < 1100) & (vv <= 600)) | ((uu >= 1100) & (vv <= 640)))
        self.free = free
        rb = boundary(self.mask)
        self.ref_free_b = np.argwhere(rb & free)            # (N,2) row, col
        self.dt_ref_b = distance_transform_edt(~rb)
        self.dist_outside = distance_transform_edt(~self.mask)   # px outside mask -> distance

    def score(self, V, T, lms, detail=False):
        from scipy.ndimage import distance_transform_edt
        sil = silhouette(V, T, self.cam)
        u0, v0, u1, v1 = self.win
        # landmarks
        uv = self.cam.project_uv(np.array([lms[k] for k in LM_NAMES]))
        err = {k: float(np.linalg.norm(uv[i] - self.lm[k])) for i, k in enumerate(LM_NAMES)}
        l_lm = sum((err[k] / LM_SIGMA[k]) ** 2 for k in LM_NAMES)
        # outside mask (anywhere; the head must never leave the cat)
        out_px = sil & ~self.mask
        n_out = int(out_px.sum())
        out_d = self.dist_outside[out_px]
        l_out = float((out_d ** 2).sum()) / 200.0
        # boundary distance along the free outline (pad crop for speed)
        pad = 30
        a0, b0, a1, b1 = v0 - pad, u0 - pad, v1 + pad, u1 + pad
        sb = boundary(sil[a0:a1, b0:b1])
        if sb.any():
            dt = distance_transform_edt(~sb)
            rf = self.ref_free_b - [a0, b0]
            d_ref2ren = dt[rf[:, 0], rf[:, 1]]
            ren_b = np.argwhere(sb) + [a0, b0]
            keep = self.free[ren_b[:, 0], ren_b[:, 1]]
            d_ren2ref = self.dt_ref_b[ren_b[keep, 0], ren_b[keep, 1]]
            bmean = 0.5 * (d_ref2ren.mean() + (d_ren2ref.mean() if len(d_ren2ref) else 0.0))
            bh95 = max(np.percentile(d_ref2ren, 95), np.percentile(d_ren2ref, 95) if len(d_ren2ref) else 0)
        else:
            bmean = bh95 = 999.0
        l_b = bmean ** 2 / 4.0
        total = 1.0 * l_lm + 1.0 * l_out + 1.5 * l_b
        if detail:
            return total, {"landmark_px": err, "outline_boundary_mean_px": float(bmean),
                           "outline_boundary_hd95_px": float(bh95), "pixels_outside_mask": n_out,
                           "max_px_outside_mask": float(out_d.max()) if n_out else 0.0,
                           "head_pixels": int(sil.sum())}, sil
        return total


def priors(Vmm, tpl, S):
    names = tpl["names"]
    skull = np.array([i for i, n in enumerate(names) if not (n.startswith("ear") or n in ("apex", "cup")
                                                             or n.startswith("r1") or n.startswith("in"))])
    nb = len(names)
    skull_full = list(skull) + [tpl["mirror_of"][i] for i in skull if i in tpl["mirror_of"]]
    P = Vmm[skull_full]
    width = P[:, 1].max() - P[:, 1].min()
    depth = P[:, 0].max() - P[:, 0].min()
    r = depth / width
    pen = 30.0 * (max(0, 0.9 - r) + max(0, r - 1.1)) ** 2 * 100
    # eyes in front of the cheeks, on the front face
    pen += 0.5 * max(0, (S["cheek_x"] + 18) - S["eye_x"]) ** 2
    # ear tip above the base, outward
    pen += 0.2 * max(0, S["ear_out_y"] - S["ear_tip_y"] + 2) ** 2
    return pen, {"depth_mm": float(depth), "width_mm": float(width), "depth_over_width": float(r)}


def make_world(tpl, S, pose):
    Vmm = hk.local_vertices(tpl, S)
    V = hk.to_world(Vmm, pose)
    att = hk.landmark_attach(tpl, Vmm, S)
    from catlib.meshio import landmarks_from_attach
    return Vmm, V, landmarks_from_attach(V, tpl["triangles"], att)


def initial_pose(target):
    # proxy skeleton: eye midpoint (proxy/landmarks3d.json) and head axis from the head part
    from catlib.skeleton import load_skeleton
    sk = load_skeleton()
    h = sk["head"]
    c0, c1 = h.center(0.0), h.center(1.0)
    ax = c1 - c0
    yaw = math.degrees(math.atan2(ax[1], ax[0]))
    pitch = math.degrees(math.atan2(-ax[2], math.hypot(ax[0], ax[1])))
    lm3 = json.loads((ROOT / "proxy" / "landmarks3d.json").read_text())["landmarks"]
    em = 0.5 * (np.array(lm3["eye_R"]) + np.array(lm3["eye_L"]))
    R = hk.rot_matrix(yaw, pitch, 0)
    t = em - R @ np.array([0.031, 0, 0])
    return {"yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": 0.0, "tx": t[0], "ty": t[1], "tz": t[2],
            "scale": 1.0}, {"proxy_head_axis_yaw_deg": yaw, "proxy_head_axis_pitch_deg": pitch,
                            "proxy_head_centre": h.center(0.5).tolist(), "proxy_eye_mid": em.tolist()}


def pose_dict(x):
    return {"yaw_deg": x["yaw_deg"], "pitch_deg": x["pitch_deg"], "roll_deg": x["roll_deg"],
            "t": [x["tx"], x["ty"], x["tz"]], "scale": x["scale"]}


def run_cma(f, x0, steps, evals, seed):
    import cma
    es = cma.CMAEvolutionStrategy(np.zeros(len(x0)), 0.5, {"seed": seed + 1, "maxfevals": evals,
                                                            "verbose": -9, "popsize": 14})
    best = (f(np.zeros(len(x0))), np.zeros(len(x0)))
    while not es.stop():
        X = es.ask()
        F = [f(z) for z in X]
        es.tell(X, F)
        i = int(np.argmin(F))
        if F[i] < best[0]:
            best = (F[i], np.array(X[i]))
    return best


def overlay(target, sil, lms_world, lod, ntri, out, V, T):
    from PIL import Image, ImageDraw
    img = Image.open(ROOT / "reference" / "cat.png").convert("RGB")
    u0, v0, u1, v1 = target.win
    pad = 40
    box = (u0 - pad, v0 - pad, u1 + pad, v1 + pad)
    a = np.asarray(img).astype(float)
    b = boundary(sil)
    # wireframe of projected edges
    crop = Image.fromarray(a.astype(np.uint8)).crop(box)
    s = 2
    crop = crop.resize((crop.width * s, crop.height * s), Image.LANCZOS)
    dr = ImageDraw.Draw(crop)
    uv = target.cam.project_uv(V)
    for t in T:
        for k in range(3):
            p, q = uv[t[k]], uv[t[(k + 1) % 3]]
            dr.line([((p[0] - box[0]) * s, (p[1] - box[1]) * s), ((q[0] - box[0]) * s, (q[1] - box[1]) * s)],
                    fill=(255, 255, 255), width=1)
    bb = np.argwhere(b)
    for r, c in bb:
        if box[0] <= c < box[2] and box[1] <= r < box[3]:
            dr.rectangle([(c - box[0]) * s, (r - box[1]) * s, (c - box[0]) * s + 1, (r - box[1]) * s + 1],
                         fill=(0, 220, 255))
    puv = target.cam.project_uv(np.array([lms_world[k] for k in LM_NAMES]))
    for i, k in enumerate(LM_NAMES):
        g = target.lm[k]
        x, y = (g[0] - box[0]) * s, (g[1] - box[1]) * s
        dr.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(0, 255, 0), width=3)
        x, y = (puv[i][0] - box[0]) * s, (puv[i][1] - box[1]) * s
        dr.line([x - 6, y - 6, x + 6, y + 6], fill=(255, 0, 0), width=3)
        dr.line([x - 6, y + 6, x + 6, y - 6], fill=(255, 0, 0), width=3)
    dr.rectangle([0, 0, 470, 24], fill=(0, 0, 0))
    dr.text((6, 6), f"head kit L{lod}: {ntri} tris. cyan=head outline, green=ref lm, red=head lm",
            fill=(255, 255, 255))
    crop.save(out)


def _atomic_write(path: Path, text: str):
    import os
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def depth_align(out: Path, fit_lod: int = 3):
    """Final step (orchestrator correction): resolve the single-view scale<->depth ambiguity
    against the proxy.  ONE scale s about the camera centre C for all LODs,
    X' = C + s (X - C): every reference-view pixel is preserved, the head only slides
    along the viewing rays.  s minimises mean |proxy SDF| over the fit_lod vertices.
    Idempotent: always starts from head/prealign/ (created from the current files on the
    first run)."""
    import shutil
    from scipy.optimize import minimize_scalar
    from catlib.proxy import sdf_lookup
    from catlib.meshio import landmarks_from_attach
    tg = Target()
    C = np.asarray(json.loads((ROOT / "contract" / "camera.json").read_text())["location"], float)
    pre = out / "prealign"
    if not pre.exists():
        pre.mkdir()
        for lod in hk.LODS:
            shutil.copy2(out / f"head_L{lod}.json", pre / f"head_L{lod}.json")
        for n in ("pose.json", "report.json"):
            shutil.copy2(out / n, pre / n)
    heads = {lod: json.loads((pre / f"head_L{lod}.json").read_text()) for lod in hk.LODS}
    pose_j = json.loads((pre / "pose.json").read_text())
    report = json.loads((pre / "report.json").read_text())
    X3 = np.asarray(heads[fit_lod]["vertices"])
    f = lambda s: float(np.abs(sdf_lookup(C + s * (X3 - C))).mean())
    res = minimize_scalar(f, bounds=(0.85, 1.2), method="bounded", options={"xatol": 1e-6})
    s = float(res.x)
    tpls = {lod: hk.build_template(lod) for lod in hk.LODS}
    summary = {"depth_aligned": True, "depth_align_scale": s, "depth_align_fit_lod": f"L{fit_lod}",
               "about": "camera centre (contract/camera.json location)", "per_lod": {}}
    for lod in hk.LODS:
        d = dict(heads[lod])
        X = np.asarray(d["vertices"])
        Xn = C + s * (X - C)
        T = d["triangles"]
        _, det0, _ = tg.score(X, T, d["landmarks"], detail=True)
        lmn = landmarks_from_attach(Xn, T, d["landmark_attach"])
        _, det1, sil = tg.score(Xn, T, lmn, detail=True)
        Vmm = hk.local_vertices(tpls[lod], d["shape"])
        _, pri = priors(Vmm, tpls[lod], d["shape"])
        th = ear_thickness(tpls[lod], Vmm)
        old_scale = d["pose"]["scale"]
        newpose = {**d["pose"], "t": (C + s * (np.asarray(d["pose"]["t"]) - C)).tolist(),
                   "scale": old_scale * s}
        dlm = max(abs(det1["landmark_px"][k] - det0["landmark_px"][k]) for k in LM_NAMES)
        uv0 = tg.cam.project_uv(np.array([d["landmarks"][k] for k in LM_NAMES]))
        uv1 = tg.cam.project_uv(np.array([lmn[k] for k in LM_NAMES]))
        row = {"mean_abs_proxy_sdf_mm": {"before": 1e3 * float(np.abs(sdf_lookup(X)).mean()),
                                         "after": 1e3 * float(np.abs(sdf_lookup(Xn)).mean())},
               "head_width_mm": {"before": pri["width_mm"] * old_scale, "after": pri["width_mm"] * old_scale * s},
               "max_projected_landmark_shift_px": float(np.linalg.norm(uv1 - uv0, axis=1).max()),
               "max_landmark_error_change_px": float(dlm),
               "outline_boundary_change_px": det1["outline_boundary_mean_px"] - det0["outline_boundary_mean_px"],
               "pixels_outside_mask": {"before": det0["pixels_outside_mask"], "after": det1["pixels_outside_mask"]},
               "ear_thickness_world_mm": {"before": {k: v * old_scale for k, v in th.items()},
                                          "after": {k: v * old_scale * s for k, v in th.items()}},
               "bbox_centre_after": (0.5 * (Xn.min(0) + Xn.max(0))).tolist()}
        d["vertices"] = [[float(c) for c in v] for v in Xn]
        d["landmarks"] = {k: [float(c) for c in v] for k, v in lmn.items()}
        d["pose"] = newpose
        d["fit"] = det1
        d["checks"] = hk.check_lod(d)
        d["depth_aligned"] = True
        d["depth_align_scale"] = s
        assert d["checks"]["ok"], (lod, d["checks"])
        _atomic_write(out / f"head_L{lod}.json", json.dumps(d))
        overlay(tg, sil, lmn, lod, d["n_triangles"], out / "renders" / f"L{lod}_overlay.png", Xn, np.asarray(T))
        rl = report["lods"][f"L{lod}"]
        rl.update(det1)
        rl["checks_world"] = d["checks"]
        rl["depth_align"] = row
        summary["per_lod"][f"L{lod}"] = row
        print(f"L{lod}: |sdf| {row['mean_abs_proxy_sdf_mm']['before']:.1f}->{row['mean_abs_proxy_sdf_mm']['after']:.1f} mm "
              f"width {row['head_width_mm']['before']:.1f}->{row['head_width_mm']['after']:.1f} mm "
              f"lm shift {row['max_projected_landmark_shift_px']:.2e}px outline d={row['outline_boundary_change_px']:.3f}px")
    pose_j.update({"depth_aligned": True, "depth_align_scale": s,
                   "depth_align": {k: v for k, v in summary.items() if k != "per_lod"},
                   "depth_align_per_lod": summary["per_lod"],
                   "t": (C + s * (np.asarray(pose_j["t"]) - C)).tolist(), "scale": pose_j["scale"] * s,
                   "prealign_t": pose_j["t"], "prealign_scale": pose_j["scale"]})
    report["pose"] = pose_j
    report["depth_aligned"] = True
    report["depth_align_scale"] = s
    _atomic_write(out / "pose.json", json.dumps(pose_j, indent=1))
    _atomic_write(out / "report.json", json.dumps(report, indent=1))
    print(f"depth_align s={s:.5f}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=2400)
    ap.add_argument("--lod-evals", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "head")
    ap.add_argument("--align-only", action="store_true",
                    help="only run the final depth alignment on the existing head/ files")
    a = ap.parse_args()
    if a.align_only:
        depth_align(a.out)
        return
    t0 = time.time()
    tg = Target()
    tpls = {lod: hk.build_template(lod) for lod in hk.LODS}
    pose0, proxy_info = initial_pose(tg)
    S0 = dict(hk.DEFAULT_SHAPE)
    keys = POSE_KEYS + SHAPE_KEYS
    base = {**pose0, **S0}
    steps = {k: 3.0 for k in SHAPE_KEYS}
    steps.update(SHAPE_STEP)
    steps.update({"yaw_deg": 8.0, "pitch_deg": 8.0, "roll_deg": 5.0, "tx": 0.01, "ty": 0.01, "tz": 0.01,
                  "scale": 0.08})

    def unpack(z, ks, ref):
        x = dict(ref)
        for i, k in enumerate(ks):
            x[k] = ref[k] + steps[k] * z[i]
            if k in BOUNDS:
                x[k] = float(np.clip(x[k], *BOUNDS[k]))
        return x

    def shared_obj(z):
        x = unpack(z, keys, base)
        S = {k: x[k] for k in S0}
        pose = pose_dict(x)
        tot = 0.0
        try:
            for lod, w in ((2, 1.0), (4, 1.0)):
                Vmm, V, lms = make_world(tpls[lod], S, pose)
                tot += w * tg.score(V, tpls[lod]["triangles"], lms)
                pen, _ = priors(Vmm, tpls[lod], S)
                tot += pen
        except Exception:
            return 1e9
        reg = sum(((S[k] - S0[k]) / 8.0) ** 2 for k in SHAPE_KEYS)
        return tot + 0.5 * reg

    f0 = shared_obj(np.zeros(len(keys)))
    fbest, zbest = run_cma(shared_obj, np.zeros(len(keys)), steps, a.evals, a.seed)
    xs = unpack(zbest, keys, base)
    S_shared = {k: float(xs[k]) for k in S0}
    pose = pose_dict(xs)
    print(f"shared fit: {f0:.2f} -> {fbest:.2f}  ({time.time() - t0:.0f}s)")

    # per-LOD small symmetric adjustment
    per_lod = {}
    report = {"lods": {}, "shared_objective": {"initial": f0, "final": fbest}}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "renders").mkdir(exist_ok=True)
    (a.out / "template").mkdir(exist_ok=True)
    for lod in hk.LODS:
        tpl = tpls[lod]

        def lod_obj(z):
            S = dict(S_shared)
            for i, k in enumerate(LOD_KEYS):
                S[k] = S_shared[k] + 2.0 * z[i]
            try:
                Vmm, V, lms = make_world(tpl, S, pose)
                pen, _ = priors(Vmm, tpl, S)
                return tg.score(V, tpl["triangles"], lms) + pen + \
                    sum(((S[k] - S_shared[k]) / 3.0) ** 2 for k in LOD_KEYS)
            except Exception:
                return 1e9
        g0 = lod_obj(np.zeros(len(LOD_KEYS)))
        g1, zb = run_cma(lod_obj, np.zeros(len(LOD_KEYS)), None, a.lod_evals, a.seed + 10 + lod)
        S = dict(S_shared)
        for i, k in enumerate(LOD_KEYS):
            S[k] = S_shared[k] + 2.0 * zb[i]
        per_lod[lod] = {k: S[k] - S_shared[k] for k in LOD_KEYS}
        d = hk.export_lod(tpl, S, pose)
        dl = hk.export_lod(tpl, S)
        chk_l = hk.check_lod(dl)
        chk_w = hk.check_lod(d)
        Vmm = hk.local_vertices(tpl, S)
        _, det, sil = tg.score(np.asarray(d["vertices"]), tpl["triangles"], d["landmarks"], detail=True)
        _, pri = priors(Vmm, tpl, S)
        ear_th = ear_thickness(tpl, Vmm)
        d.update({"name": f"head_kit_L{lod}", "pose": pose, "shape": S, "checks": chk_w,
                  "local_checks": chk_l, "fit": det})
        dl.update({"shape": S, "checks": chk_l})
        (a.out / f"head_L{lod}.json").write_text(json.dumps(d))
        (a.out / "template" / f"head_L{lod}_local.json").write_text(json.dumps(dl))
        overlay(tg, sil, d["landmarks"], lod, d["n_triangles"], a.out / "renders" / f"L{lod}_overlay.png",
                np.asarray(d["vertices"]), np.asarray(d["triangles"]))
        report["lods"][f"L{lod}"] = {"triangles": d["n_triangles"], "neck_loop": len(d["neck_loop"]),
                                     "objective": {"shared": g0, "after_lod_adjust": g1}, **det,
                                     "priors": pri, "ear_min_thickness_mm": ear_th,
                                     "checks_world": chk_w, "mirror_symmetry_local": {
                                         "max_err_m": chk_l["mirror_max_err_m"],
                                         "triangulation_symmetric": chk_l["triangulation_symmetric"]},
                                     "per_lod_shape_delta_mm": per_lod[lod]}
        print(f"L{lod}: {d['n_triangles']} tris  lm " +
              " ".join(f"{k}={v:.1f}" for k, v in det["landmark_px"].items()) +
              f"  outline={det['outline_boundary_mean_px']:.2f}px out={det['pixels_outside_mask']}"
              f" maxout={det['max_px_outside_mask']:.1f} ok={chk_w['ok']}")
    R = hk.rot_matrix(pose["yaw_deg"], pose["pitch_deg"], pose["roll_deg"])
    pj = {"format": "headkit-pose-1",
          "transform": "world = t + scale * 1e-3 * R @ local_mm ;  R = Rz(yaw) Ry(pitch) Rx(roll)",
          "rotation_matrix": R.tolist(), **pose, "shape": S_shared, "shape_units": "mm (head-local)",
          "per_lod_shape_delta_mm": {f"L{k}": v for k, v in per_lod.items()},
          "initial_pose_from_proxy": {**pose_dict(pose0), **proxy_info},
          "seed": a.seed, "evals": a.evals, "lod_evals": a.lod_evals, "elapsed_s": time.time() - t0}
    (a.out / "pose.json").write_text(json.dumps(pj, indent=1))
    report["pose"] = pj
    (a.out / "report.json").write_text(json.dumps(report, indent=1))
    import shutil
    if (a.out / "prealign").exists():
        shutil.rmtree(a.out / "prealign")          # fresh fit -> fresh pre-alignment copy
    depth_align(a.out)
    print(f"done in {time.time() - t0:.0f}s")


def ear_thickness(tpl, Vmm):
    """Ear shell thickness (mm), right ear: distance from the front-surface points (cup
    centre, rim/cup ring, mid-height front corners) to the ear's BACK faces (triangles
    using only back-ring vertices and the apex).  L0-L2 pyramids: base thickness = distance
    from the back corner/edge to the front face's base edge."""
    from catlib.geom import point_mesh_distance
    names = tpl["names"]
    T = np.asarray(tpl["triangles"])
    backset = {names.index(n) for n in ("ear_bo", "ear_bi", "r1ear_bo", "r1ear_bi", "apex", "ear_b")
               if n in names}
    back = [i for i, t in enumerate(T) if set(t.tolist()) <= backset]
    if tpl["lod"] <= 2:
        fi, fo, b = (Vmm[names.index(n)] for n in ("ear_fi", "ear_fo", "ear_b"))
        e = (fo - fi) / np.linalg.norm(fo - fi)
        r = b - fi
        return {"base_thickness": float(np.linalg.norm(r - (r @ e) * e))}
    front = [n for n in ("cup", "in0", "in1", "in4", "r1ear_fi", "r1ear_fo") if n in names]
    d, _ = point_mesh_distance(Vmm[[names.index(n) for n in front]], Vmm, T[back])
    return {n: float(x) for n, x in zip(front, d)}


if __name__ == "__main__":
    main()
