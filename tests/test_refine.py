"""Tests for catlib.topo and the shared vertex refiner (scripts/refine_vertices.py).
CPU only (Blender is used in background mode for assembly / validation, never rendering).

    python3 tests/test_refine.py [--keep DIR] [--iters 6000] [--skip-blender]

The refiner test does not need the real proxy or the locked camera: it builds a smooth
synthetic target (union of ellipsoids and capsules, analytic SDF on a 3 mm grid, dense
marching-cubes surface) behind the CONTRACT §11 file interface in a temp dir, derives a
mask / landmarks through a fixed test camera, perturbs the low-poly tests/fixtures.blob_cat
and checks that refinement raises silhouette IoU and lowers landmark and SDF error without
breaking validity.  Synthetic data only; nothing here is a cat reconstruction.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/
import _setup  # noqa: E402  (skill scripts on sys.path; LOWPOLY_CASE defaults to examples/cat)
REPO = _setup.REPO
ROOT = _setup.CASE

import numpy as np  # noqa: E402

import fixtures  # noqa: E402
from catlib import geom, meshio, paths, topo  # noqa: E402
from catlib.camera import Camera  # noqa: E402

PY = sys.executable
RESULTS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)
    return ok


# =============================================================================== topo tests

def _ellipsoid_dict():
    V, T = fixtures.ellipsoid((0.0, 0.0, 0.2), (0.12, 0.08, 0.07), 8, 5)
    reg = ["torso" if i % 2 else "head" for i in range(len(T))]
    mat = ["fur_orange"] * len(T)
    rng = np.random.default_rng(3)
    pts = V[T[rng.integers(len(T), size=16)]].mean(1) * 1.3 - np.array([0, 0, 0.06])
    _, snapped = geom.point_mesh_distance(pts, V, T)
    lm = dict(zip(meshio.LANDMARKS, snapped))
    d = meshio.to_dict(V, T, reg, mat, lm, 200, "topo_test")
    meshio.attach_landmarks(d)
    d["landmark_attach"]["nose"] = {"vertex": 0}           # one vertex attachment
    meshio.sync_landmarks(d)
    return d


def _rep(d):
    return geom.topology_report(np.asarray(d["vertices"]), np.asarray(d["triangles"]))


def _consistent(d):
    r = _rep(d)
    return r["closed_manifold"] and r["inconsistent_winding_edges"] == 0


def _lm_on_surface(d, tol=2e-3):
    m = meshio.CatMesh(d)
    P = np.stack([m.landmarks[k] for k in meshio.LANDMARKS])
    dist, _ = geom.point_mesh_distance(P, m.vertices, m.triangles)
    return float(dist.max()) <= tol, float(dist.max())


def _labels_ok(d):
    return (len(d["tri_region"]) == len(d["triangles"]) == len(d["tri_material"])
            and not [e for e in meshio.validate_dict(d) if not e.startswith("over_budget")])


def _interior_edge(d):
    T = np.asarray(d["triangles"])
    t = T[5]
    return int(t[0]), int(t[1])


def test_topo():
    d = _ellipsoid_dict()
    n = topo.triangle_count(d)
    check("topo: base ellipsoid closed, consistent, valid", _consistent(d) and _labels_ok(d), f"{n} tris")
    lm0 = meshio.CatMesh(d).landmarks

    a, b = _interior_edge(d)
    f = topo.edge_flip(d, a, b)
    ok_s, md = _lm_on_surface(f)
    check("topo.edge_flip: count unchanged, manifold+consistent, labels+landmarks valid",
          topo.triangle_count(f) == n and _consistent(f) and _labels_ok(f) and ok_s, f"lm max dist {md*1e3:.3f} mm")
    T0, T1 = np.asarray(d["triangles"]), np.asarray(f["triangles"])
    changed = np.flatnonzero((np.sort(T0, 1) != np.sort(T1, 1)).any(1))
    ce = sorted(set(T1[changed[0]]) & set(T1[changed[1]]))
    f2 = topo.edge_flip(f, int(ce[0]), int(ce[1]))
    check("topo.edge_flip back restores the original triangle set",
          _consistent(f2) and sorted(map(sorted, f2["triangles"])) == sorted(map(sorted, d["triangles"])))

    s = topo.edge_split(d, a, b)
    lm_s = meshio.CatMesh(s).landmarks
    moved = max(float(np.linalg.norm(lm_s[k] - lm0[k])) for k in lm0)
    check("topo.edge_split: +2 tris, +1 vertex, manifold, landmarks do not move",
          topo.triangle_count(s) == n + 2 and len(s["vertices"]) == len(d["vertices"]) + 1
          and _consistent(s) and _labels_ok(s) and moved < 1e-9, f"max landmark move {moved:.2e} m")

    c = topo.edge_collapse(d, a, b)
    ok_s, md = _lm_on_surface(c)
    check("topo.edge_collapse: -2 tris, -1 vertex, manifold, landmarks on surface",
          topo.triangle_count(c) == n - 2 and len(c["vertices"]) == len(d["vertices"]) - 1
          and _consistent(c) and _labels_ok(c) and ok_s, f"lm max dist {md*1e3:.3f} mm")
    # link-condition violation: collapsing a tetrahedron edge
    tet = {"vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
           "triangles": [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
           "tri_region": ["torso"] * 4, "tri_material": ["fur_orange"] * 4, "landmarks": {}}
    try:
        topo.edge_collapse(tet, 0, 1)
        raised = False
    except ValueError:
        raised = True
    check("topo.edge_collapse: link-condition violation raises", raised)

    # weld: duplicate one vertex used by some triangles, then weld it back
    u = json.loads(json.dumps(d))
    k = len(u["vertices"])
    u["vertices"].append(list(u["vertices"][3]))
    T = np.asarray(u["triangles"])
    hit = np.flatnonzero((T == 3).any(1))[:2]
    for i in hit:
        u["triangles"][i] = [k if x == 3 else x for x in u["triangles"][i]]
    rep_u = _rep(u)
    w = topo.weld_vertices(u)
    check("topo.weld_vertices: unwelded seam welded, count unchanged, manifold",
          rep_u["unwelded_coincident_vertex_groups"] == 1 and topo.triangle_count(w) == n
          and len(w["vertices"]) == len(d["vertices"]) and _consistent(w) and _labels_ok(w))
    w2 = topo.weld_vertices(d, pairs=[(a, b)])
    check("topo.weld_vertices(pairs): merging an edge removes its 2 triangles", topo.triangle_count(w2) == n - 2)

    # bridge + caps on rings
    for na, nb in ((8, 8), (8, 5)):
        th_a = np.linspace(0, 2 * np.pi, na, endpoint=False)
        th_b = np.linspace(0, 2 * np.pi, nb, endpoint=False) + 0.3
        Va = np.stack([0.05 * np.cos(th_a), 0.05 * np.sin(th_a), np.zeros(na)], 1)
        Vb = np.stack([0.03 * np.cos(th_b), 0.03 * np.sin(th_b), np.full(nb, 0.1)], 1)
        rd = {"vertices": np.vstack([Va, Vb]).tolist(), "triangles": np.zeros((0, 3), int).tolist(),
              "tri_region": [], "tri_material": [], "landmarks": {}}
        A = list(range(na)); B = list(range(na, na + nb))[::-1]          # B deliberately reversed
        br = topo.bridge_rings(rd, A, B, region="tail", material="fur_stripe")
        c1 = topo.cap_tip(br, A, tip=[0, 0, -0.03])
        c2 = topo.cap_fan(c1, B)
        oo = topo.orient_outward(c2)
        r = _rep(oo)
        exp_band = 2 * na if na == nb else na + nb
        check(f"topo.bridge_rings {na}->{nb}: band {exp_band} tris; + cap_tip ({na}) + cap_fan ({nb - 2}) "
              f"-> closed, consistent, positive volume",
              topo.triangle_count(br) == exp_band and topo.triangle_count(c1) == exp_band + na
              and topo.triangle_count(c2) == exp_band + na + nb - 2 and _consistent(c2)
              and r["closed_manifold"] and r["signed_volume_m3"] > 0 and set(br["tri_region"]) == {"tail"},
              f"vol {r['signed_volume_m3']:.2e} m3, inconsistent {r['inconsistent_winding_edges']}")

    # orient_outward: scramble windings
    sc = json.loads(json.dumps(d))
    rng = np.random.default_rng(0)
    for i in rng.choice(n, n // 2, replace=False):
        t = sc["triangles"][i]
        sc["triangles"][i] = [t[0], t[2], t[1]]
    sc_lm = meshio.CatMesh(sc).landmarks
    oo = topo.orient_outward(sc)
    r = _rep(oo)
    lm_o = meshio.CatMesh(oo).landmarks
    moved = max(float(np.linalg.norm(lm_o[k] - sc_lm[k])) for k in sc_lm)
    check("topo.orient_outward: scrambled winding fixed, volume positive, landmarks unchanged",
          r["inconsistent_winding_edges"] == 0 and r["signed_volume_m3"] > 0 and moved < 1e-12,
          f"vol {r['signed_volume_m3']:.2e}, landmark move {moved:.1e}")


# =============================================================================== synthetic target

TEST_CAMERA = {   # frozen copy of the provisional camera so this test never depends on the fit
    "status": "test", "resolution": [1536, 1536], "focal_length_mm": 85.0, "sensor_width_mm": 36.0,
    "sensor_fit": "HORIZONTAL", "location": [1.419986, -2.02795, 0.517933], "look_at": [0.0, 0.0, 0.17],
    "roll_deg": 0.0, "clip_start": 0.05, "clip_end": 50.0}


def _sdf_ellipsoid(P, c, r):
    q = (P - c) / r
    return (np.linalg.norm(q, axis=-1) - 1.0) * min(r)


def _sdf_capsule(P, a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + t[..., None] * ab), axis=-1) - r


def synthetic_sdf(P):
    parts = [
        _sdf_ellipsoid(P, np.array([0.0, 0.0, 0.17]), np.array([0.20, 0.08, 0.075])),
        _sdf_ellipsoid(P, np.array([0.23, -0.02, 0.29]), np.array([0.06, 0.065, 0.06])),
        _sdf_capsule(P, (0.23, -0.055, 0.33), (0.235, -0.07, 0.385), 0.012),
        _sdf_capsule(P, (0.23, 0.015, 0.33), (0.235, 0.03, 0.385), 0.012),
        _sdf_capsule(P, (0.14, -0.045, 0.02), (0.14, -0.045, 0.16), 0.02),
        _sdf_capsule(P, (0.10, 0.045, 0.02), (0.10, 0.045, 0.16), 0.02),
        _sdf_capsule(P, (-0.14, -0.045, 0.02), (-0.14, -0.045, 0.16), 0.02),
        _sdf_capsule(P, (-0.11, 0.045, 0.02), (-0.11, 0.045, 0.16), 0.02),
        _sdf_capsule(P, (-0.19, 0.0, 0.20), (-0.33, 0.0, 0.42), 0.014),
    ]
    d = np.min(np.stack(parts), 0)
    return np.maximum(d, -P[..., 2])          # cut at the ground: soles flat on z = 0


def build_synthetic(work: Path):
    """-> (derived_dir, proxy_dir, camera_json, scene_json)."""
    from skimage.measure import marching_cubes
    h = 0.003
    lo = np.array([-0.40, -0.16, -0.03]); hi = np.array([0.36, 0.16, 0.46])
    n = np.ceil((hi - lo) / h).astype(int) + 1
    gx, gy, gz = (lo[k] + h * np.arange(n[k]) for k in range(3))
    G = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), -1)
    sdf = synthetic_sdf(G).astype(np.float32)
    verts, faces, _, _ = marching_cubes(sdf, 0.0, spacing=(h, h, h))
    verts = verts + lo
    verts[:, 2] = np.maximum(verts[:, 2], 0.0)
    pdir = work / "proxy"; pdir.mkdir(parents=True, exist_ok=True)
    np.savez(pdir / "proxy_mesh.npz", vertices=verts.astype(np.float64), triangles=faces.astype(np.int32))
    np.savez(pdir / "sdf.npz", origin=lo, spacing=np.float64(h), sdf=sdf)
    lm_raw = {"ear_tip_R": (0.236, -0.072, 0.40), "ear_tip_L": (0.236, 0.032, 0.40),
              "eye_R": (0.28, -0.05, 0.31), "eye_L": (0.285, 0.0, 0.31), "nose": (0.30, -0.02, 0.28),
              "paw_FR": (0.14, -0.045, 0.0), "paw_FL": (0.10, 0.045, 0.0),
              "paw_HR": (-0.14, -0.045, 0.0), "paw_HL": (-0.11, 0.045, 0.0),
              "tail_base": (-0.20, 0.0, 0.215), "tail_mid": (-0.26, -0.015, 0.31),
              "tail_tip": (-0.335, 0.0, 0.435), "shoulder_R": (0.13, -0.08, 0.2),
              "hip_R": (-0.13, -0.08, 0.18), "hock_HR": (-0.15, -0.065, 0.06), "chin": (0.25, -0.02, 0.23)}
    from catlib import proxy as P
    proj, _ = P.closest_surface(np.array(list(lm_raw.values())), path=pdir / "sdf.npz")
    proj[:, 2] = np.maximum(proj[:, 2], 0.0)
    lm3 = dict(zip(lm_raw, proj))
    (pdir / "landmarks3d.json").write_text(json.dumps({"landmarks": {k: v.tolist() for k, v in lm3.items()},
                                                       "note": "synthetic test proxy"}))
    camj = work / "camera.json"
    camj.write_text(json.dumps(TEST_CAMERA))
    cam = Camera.from_json(camj)
    target = {"vertices": verts.tolist(), "triangles": faces.tolist(),
              "landmarks": {k: v.tolist() for k, v in lm3.items()}}
    fixtures.synthetic_derived(target, work / "derived", cam, jitter_px=0.0, dilate_px=0)
    lj = json.loads((work / "derived" / "landmarks.json").read_text())
    for v in lj["landmarks"].values():
        v["sigma_px"] = 3.0
    (work / "derived" / "landmarks.json").write_text(json.dumps(lj))
    sc = json.loads(paths.SCENE_JSON.read_text())
    sc["orbit_target"] = [0.0, 0.0, 0.17]
    (work / "scene.json").write_text(json.dumps(sc))
    return work / "derived", pdir, camj, work / "scene.json", len(faces)


def perturbed_start(budget=None):
    d = fixtures.blob_cat(name="cat_refine_test")
    meshio.attach_landmarks(d)
    V = np.asarray(d["vertices"], float)
    T = np.asarray(d["triangles"])
    reg = np.array(d["tri_region"])
    rng = np.random.default_rng(7)
    vreg = {}
    for i, t in enumerate(T):
        for v in t:
            vreg[int(v)] = reg[i]
    for v, r in vreg.items():
        if r == "torso":
            V[v] = np.array([0.0, 0.0, 0.17]) + (V[v] - np.array([0.0, 0.0, 0.17])) * np.array([0.88, 0.85, 0.85])
        elif r == "head":
            V[v] += np.array([-0.025, 0.0, -0.02])
        elif r == "ears":
            V[v] += np.array([-0.02, 0.0, -0.03])
        elif r == "tail":
            V[v] += np.array([0.03, 0.0, -0.03]) * (V[v, 2] - 0.2) / 0.2
    sole = V[:, 2] < 1e-3
    noise = rng.normal(0, 0.004, V.shape)
    noise[sole, 2] = 0
    V += noise
    V[:, 2] = np.maximum(V[:, 2], 0.0)
    d["vertices"] = V.tolist()
    meshio.sync_landmarks(d)
    if budget:
        d["budget"] = budget
    return d


def test_refiner(work: Path, iters: int, skip_blender: bool):
    from catlib.refine import Refiner, Settings, Sources
    t = time.time()
    derived, pdir, camj, scenej, nproxy = build_synthetic(work)
    check("synthetic target built (analytic SDF 3 mm grid + marching cubes)", nproxy > 10000,
          f"{nproxy} proxy triangles, {time.time() - t:.1f}s")
    start = perturbed_start()
    sj = work / "start" / "mesh.json"
    meshio.save(sj, start)
    out = work / "refined" / "mesh.json"
    t = time.time()
    p = subprocess.run([PY, str(paths.SCRIPTS / "refine_vertices.py"), "--mesh", str(sj), "--out", str(out),
                        "--iters", str(iters), "--seed", "0", "--derived-dir", str(derived), "--camera", str(camj),
                        "--scene", str(scenej), "--proxy-dir", str(pdir), "--quiet"],
                       capture_output=True, text=True)
    dt = time.time() - t
    if not check("refine_vertices CLI exit 0", p.returncode == 0, p.stdout[-600:] + p.stderr[-1500:]):
        return
    print(p.stdout.strip())
    log = json.loads(out.with_suffix(".trace.json").read_text())
    b, a = log["unofficial_before"], log["unofficial_after"]
    src = Sources(derived, camj, scenej, pdir)
    from catlib import proxy as P

    def sdf_err(dd):
        V = np.asarray(dd["vertices"]); T = np.asarray(dd["triangles"])
        pts = np.vstack([V, V[T].mean(1)])
        return float(np.abs(P.sdf_lookup(pts, src.proxy_sdf)).mean() * 1000)

    rd = json.loads(out.read_text())
    s0, s1 = sdf_err(start), sdf_err(rd)
    print(f"  before: IoU {b['iou']:.4f} bnd {b['boundary_mean_px']:.2f}px lm {b['landmark_sharp_mean_px']:.2f}px "
          f"|sdf| {s0:.2f}mm nv-IoU {b['novel_view_iou_vs_proxy']:.4f} chamfer {b['chamfer_mm_vs_proxy']:.2f}mm")
    print(f"  after : IoU {a['iou']:.4f} bnd {a['boundary_mean_px']:.2f}px lm {a['landmark_sharp_mean_px']:.2f}px "
          f"|sdf| {s1:.2f}mm nv-IoU {a['novel_view_iou_vs_proxy']:.4f} chamfer {a['chamfer_mm_vs_proxy']:.2f}mm")
    print(f"  {log['proposals']} proposals in {log['optimise_s']}s ({log['proposals_per_s']}/s); counters "
          + json.dumps({k: (v['accepted'], v['proposed']) for k, v in log['counters'].items()}))
    check("refiner: triangle count unchanged and output schema-valid",
          len(rd["triangles"]) == len(start["triangles"]) and not meshio.validate_dict(rd))
    check("refiner: silhouette IoU raised substantially (>= +0.08 abs, and >= 0.85)",
          a["iou"] - b["iou"] >= 0.08 and a["iou"] >= 0.85, f"{b['iou']:.4f} -> {a['iou']:.4f}")
    check("refiner: boundary distance lowered by >= 30%", a["boundary_mean_px"] <= 0.7 * b["boundary_mean_px"],
          f"{b['boundary_mean_px']:.2f} -> {a['boundary_mean_px']:.2f} px")
    check("refiner: sharp landmark error lowered by >= 30%",
          a["landmark_sharp_mean_px"] <= 0.7 * b["landmark_sharp_mean_px"],
          f"{b['landmark_sharp_mean_px']:.2f} -> {a['landmark_sharp_mean_px']:.2f} px")
    check("refiner: mean |SDF| lowered by >= 30%", s1 <= 0.7 * s0, f"{s0:.2f} -> {s1:.2f} mm")
    check("refiner: novel-view IoU vs proxy not degraded",
          a["novel_view_iou_vs_proxy"] >= b["novel_view_iou_vs_proxy"] - 0.005,
          f"{b['novel_view_iou_vs_proxy']:.4f} -> {a['novel_view_iou_vs_proxy']:.4f}")
    ac = log["after_constraints"]
    check("refiner: constraints hold (no degenerate, min z >= 0, edges >= 1.5 mm)",
          ac["degenerate_triangles"] == 0 and ac["min_z"] >= 0 and ac["min_edge_m"] >= 1.5e-3 - 1e-9,
          json.dumps(ac))
    lm_att = meshio.CatMesh(rd)
    check("refiner: landmarks recomputed from landmark_attach",
          rd.get("landmark_attach") == start["landmark_attach"]
          and all(np.allclose(lm_att.landmarks[k], rd["landmarks"][k], atol=1e-6) for k in meshio.LANDMARKS))
    # determinism
    s = Settings(iters=400, res=(384,), seed=5, verbose=False)
    r1 = Refiner(start, src, s).run(); r2 = Refiner(start, src, s).run()
    check("refiner: deterministic for a fixed seed and --iters",
          np.array_equal(np.asarray(r1["vertices"]), np.asarray(r2["vertices"])))
    # flips keep the count and validity
    s = Settings(iters=2500, res=(384,), seed=1, flips=True, verbose=False)
    rf = Refiner(start, src, s)
    df = rf.run()
    rep = geom.topology_report(np.asarray(df["vertices"]), np.asarray(df["triangles"]))
    fc = rf.counters.get("flip", {})
    check("refiner --flips: count unchanged, still closed manifold + consistent",
          len(df["triangles"]) == len(start["triangles"]) and rep["closed_manifold"]
          and rep["inconsistent_winding_edges"] == 0, f"flips accepted {fc.get('accepted')} of {fc.get('proposed')}")
    if skip_blender:
        print("[SKIP] Blender validation")
        return
    pa = subprocess.run([str(paths.BLENDER), "-b", "--factory-startup", "--python",
                         str(paths.SCRIPTS / "assemble_variant.py"), "--", "--mesh", str(out),
                         "--out-dir", str(out.parent)], capture_output=True, text=True)
    if not check("assemble_variant on refined mesh", pa.returncode == 0, pa.stdout[-500:] + pa.stderr[-500:]):
        return
    import validate_variant
    rep = validate_variant.validate(out)
    c = rep["counts"]
    check("validate_variant on refined mesh: no hard failures, counts agree",
          rep["valid"] and c["all_agree"], f"A={c['A_mesh_json']} B={c['B_blend_evaluated']} "
          f"C={c['C_glb_reimported']} D={c['D_glb_parsed']}; hard {rep['hard_failures']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=Path, default=None)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--skip-blender", action="store_true")
    a = ap.parse_args()
    work = a.keep or Path(tempfile.mkdtemp(prefix="cat_refine_test_"))
    if a.keep and work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    print("work dir:", work)
    t0 = time.time()
    test_topo()
    test_refiner(work, a.iters, a.skip_blender)
    nf = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - nf}/{len(RESULTS)} checks passed in {time.time() - t0:.1f}s; work dir {work}")
    (work / "test_results.json").write_text(json.dumps(
        [{"check": n, "ok": o, "detail": d} for n, o, d in RESULTS], indent=1))
    return 0 if nf == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
