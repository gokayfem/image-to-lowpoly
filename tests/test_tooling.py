"""Self-test of the shared tooling.  CPU only (Blender is used in background mode for
assembly / counting / ray casts, never for rendering).

    python3 tests/test_tooling.py [--keep DIR] [--skip-real]

Proves: catlib.camera vs Blender projection (< 0.5 px) for the reference and every novel
view; stored matrix_world consistency; catlib.raster vs a Blender BVH pixel-centre ray
cast at 1536 px; four-way count agreement on a good fixture; that every hard-failure check
fires on a deliberately bad fixture; GLB non-TRIANGLES detection; metric sanity; and a
--no-render evaluation on synthetic reference data (plus on the real reference/derived
files when they exist).  Prints rasteriser timings.
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
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
import validate_variant  # noqa: E402
from catlib import meshio, metrics, paths, raster  # noqa: E402
from catlib.camera import Camera  # noqa: E402

PY = sys.executable
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)
    return ok


def blender(script: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run([str(paths.BLENDER), "-b", "--factory-startup", "--python", str(script),
                           "--", *map(str, args)], capture_output=True, text=True)


def assemble(mesh_json: Path) -> None:
    p = blender(paths.SCRIPTS / "assemble_variant.py", "--mesh", mesh_json, "--out-dir", mesh_json.parent)
    if p.returncode != 0:
        raise RuntimeError(p.stdout[-2000:] + p.stderr[-2000:])


def make_case(base: Path, name: str, d: dict) -> Path:
    ck = base / name
    ck.mkdir(parents=True, exist_ok=True)
    mj = ck / "mesh.json"
    # write raw json (meshio.save refuses schema-invalid dicts; some cases are invalid on purpose)
    mj.write_text(json.dumps(d))
    return mj


def fires(rep: dict, needle: str) -> bool:
    return any(needle.lower() in h.lower() for h in rep["hard_failures"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=Path, default=None, help="work dir to keep (default: temp)")
    ap.add_argument("--skip-real", action="store_true")
    a = ap.parse_args()
    work = a.keep or Path(tempfile.mkdtemp(prefix="cat_tooling_test_"))
    if work.exists() and a.keep:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    print("work dir:", work)
    t_start = time.time()
    cam = Camera.from_json()

    # ------------------------------------------------------------------ camera
    st = cam.check_stored_matrix()
    check("camera: stored matrix_world == derived from location/look_at/roll", st["ok"],
          f"max |diff| {st['max_abs_diff']:.2e}")
    good = fixtures.blob_cat(name="cat_fixture")
    n = len(good["triangles"])
    good_mj = make_case(work, "good", good)
    p = blender(_setup.TESTS / "blender_checks.py", "--mesh", good_mj, "--out", work / "good")
    if not check("blender_checks ran", p.returncode == 0 and "BLENDER_CHECKS_OK" in p.stdout,
                 p.stderr[-500:] if p.returncode else ""):
        return 1
    bc = json.loads((work / "good" / "blender_checks.json").read_text())
    for vname, r in bc["projection"].items():
        check(f"projection vs world_to_camera_view [{vname} {r['size'][0]}px]", r["max_err_px"] < 0.5,
              f"max {r['max_err_px']:.5f} px, mean {r['mean_err_px']:.5f} px, depth {r['max_depth_err_m']:.1e} m")
    rc_mask = np.load(work / "good" / "raycast_mask.npy")
    rs = raster.silhouette(np.array(good["vertices"]), np.array(good["triangles"]), cam)
    x = metrics.iou(rs, rc_mask)
    check("raster vs Blender BVH pixel-centre ray cast @1536 IoU >= 0.995", x >= 0.995,
          f"IoU {x:.6f}, xor {int((rs ^ rc_mask).sum())} px of {int(rc_mask.sum())}")
    # resolution scaling consistency: 768 raster vs 2x2-downsampled 1536 raster
    r768 = raster.silhouette(np.array(good["vertices"]), np.array(good["triangles"]), cam, 768)
    down = rs.reshape(768, 2, 768, 2).mean((1, 3)) >= 0.5
    x2 = metrics.iou(r768, down)
    check("raster @768 consistent with downsampled @1536 (IoU >= 0.98)", x2 >= 0.98, f"IoU {x2:.4f}")

    # ------------------------------------------------------------------ rasteriser timings
    timings = {}
    for seg, rings in ((10, 6), (26, 20)):
        V, T = fixtures.ellipsoid((0, 0, 0.17), (0.25, 0.09, 0.12), seg, rings)
        for size in (384, 768, 1536):
            raster.silhouette(V, T, cam, size)
            k = 100
            t = time.perf_counter()
            for _ in range(k):
                raster.silhouette(V, T, cam, size)
            timings[f"{len(T)}tris@{size}"] = round((time.perf_counter() - t) / k * 1000, 3)
    print("raster ms/call:", timings)

    # ------------------------------------------------------------------ good fixture
    assemble(good_mj)
    rep = validate_variant.validate(good_mj)
    c = rep["counts"]
    check("good fixture: counts A=B=C=D", c["all_agree"] and c["A_mesh_json"] == n,
          f"A={c['A_mesh_json']} B={c['B_blend_evaluated']} C={c['C_glb_reimported']} D={c['D_glb_parsed']}")
    check("good fixture: no hard failures", rep["valid"], "; ".join(rep["hard_failures"]))
    dsc = rep["disclosed"]
    check("good fixture: disclosed topology present", dsc["closed_manifold"] and dsc["components"] == 9,
          f"components {dsc['components']} {dsc['component_triangles']}, self-intersecting pairs "
          f"{dsc['self_intersecting_pairs']}, hidden {dsc['hidden_triangles']}, flat {dsc['flat_redundant_vertices']}")
    p = subprocess.run([PY, str(paths.SCRIPTS / "validate_variant.py"), "--mesh", str(good_mj)],
                       capture_output=True, text=True)
    check("validate_variant CLI exit 0 on good fixture", p.returncode == 0, p.stdout.strip().splitlines()[0])

    # ------------------------------------------------------------------ bad fixtures (mesh.json)
    def bad(name, d, needle, inject=None, expect_assemble=True):
        mj = make_case(work, name, d)
        try:
            assemble(mj)
        except RuntimeError as e:
            if expect_assemble:
                return check(f"bad[{name}] assembles", False, str(e)[-300:])
        if inject:
            p = blender(_setup.TESTS / "inject_blend.py", "--blend", mj.parent / f"{d['name']}.blend",
                        "--mode", inject)
            if p.returncode != 0:
                return check(f"bad[{name}] inject", False, p.stdout[-500:] + p.stderr[-500:])
        r = validate_variant.validate(mj)
        cc = r["counts"]
        ok = (not r["valid"]) and all(fires(r, nd) for nd in ([needle] if isinstance(needle, str) else needle))
        check(f"bad[{name}] fires {needle!r}", ok,
              f"A={cc.get('A_mesh_json')} B={cc.get('B_blend_evaluated')} C={cc.get('C_glb_reimported')} "
              f"D={cc.get('D_glb_parsed')}; failures: {[h[:90] for h in r['hard_failures']]}")
        return r

    d = fixtures.blob_cat(name="cat_fixture"); d["budget"] = n - 1
    bad("over_budget", d, "over budget")

    d = fixtures.blob_cat(name="cat_fixture")
    d["triangles"].append(list(d["triangles"][0])); d["tri_region"].append("torso")
    d["tri_material"].append("fur_orange"); d["budget"] = n + 1
    bad("duplicate", d, "duplicate triangles")

    d = fixtures.blob_cat(name="cat_fixture")
    d2 = json.loads(json.dumps(d))
    d2["vertices"] = list(d["vertices"]) + [[0.0, 0.0, 0.25], [0.05, 0.0, 0.25], [0.1, 0.0, 0.25]]
    k = len(d["vertices"])
    d2["triangles"].append([k, k + 1, k + 2]); d2["tri_region"].append("torso")
    d2["tri_material"].append("fur_orange"); d2["budget"] = n + 1
    bad("degenerate", d2, "degenerate")

    d = fixtures.blob_cat(name="cat_fixture"); d["budget"] = n + 20
    r = bad("hidden_object", d, ["expected exactly one MESH object", "counts disagree", "hidden/unlinked"], inject="hidden_object")
    if isinstance(r, dict):
        check("hidden object counted in B (B = A + 12)",
              r["counts"]["B_blend_evaluated"] == n + 12, f"B={r['counts']['B_blend_evaluated']}")

    d = fixtures.blob_cat(name="cat_fixture")
    r = bad("quad_trap", d, "non-triangle polygons", inject="quad_trap")
    if isinstance(r, dict):
        co = r["blend"]["cat_object"]
        check("quad trap: B counts loop triangles, not polygons",
              r["counts"]["B_blend_evaluated"] == n and co["polygons"] < n,
              f"polygons {co['polygons']} vs B {r['counts']['B_blend_evaluated']} (A={n})")

    d = fixtures.blob_cat(name="cat_fixture")
    r = bad("modifier", d, ["counts disagree", "modifiers"], inject="modifier")

    d = fixtures.blob_cat(name="cat_fixture")
    bad("orphan_mesh", d, ["orphan mesh"], inject="orphan_mesh")

    d = fixtures.blob_cat(name="cat_fixture")
    d["vertices"] = [[x, 0.08 * y, z] for x, y, z in d["vertices"]]
    d["landmarks"] = {kk: [v[0], 0.08 * v[1], v[2]] for kk, v in d["landmarks"].items()}
    bad("billboard", d, "billboard")

    d = fixtures.blob_cat(name="cat_fixture")
    d["vertices"] = [[x, y, z - 0.01] for x, y, z in d["vertices"]]
    bad("below_ground", d, "min z")

    d = fixtures.blob_cat(name="cat_fixture"); del d["landmarks"]["tail_mid"]
    mj = make_case(work, "missing_landmark", d)
    errs = meshio.validate_dict(d)
    check("schema: missing landmark detected", any("tail_mid" in e for e in errs), str(errs))
    p = subprocess.run([PY, str(paths.SCRIPTS / "evaluate.py"), "--mesh", str(mj), "--no-render"],
                       capture_output=True, text=True)
    check("evaluate exits non-zero on schema failure", p.returncode == 2, f"rc {p.returncode}")

    d = fixtures.blob_cat(name="cat_fixture")
    d["triangles"][0] = [0, 1, 2, 3]
    check("schema: quad face in mesh.json rejected",
          any("not triangles" in e for e in meshio.validate_dict(d)))

    # ------------------------------------------------------------------ GLB non-TRIANGLES mode
    glb = (work / "good" / "cat_fixture.glb").read_bytes()
    clen, _ = struct.unpack_from("<I4s", glb, 12)
    js = json.loads(glb[20:20 + clen])
    js["meshes"][0]["primitives"][0]["mode"] = 1
    nj = json.dumps(js).encode()
    nj += b" " * ((4 - len(nj) % 4) % 4)
    rest = glb[20 + clen:]
    out = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(nj) + len(rest)) + struct.pack("<I4s", len(nj), b"JSON") + nj + rest
    (work / "lines.glb").write_bytes(out)
    gp = validate_variant.parse_glb(work / "lines.glb")
    check("GLB parser flags non-TRIANGLES primitive", any("not TRIANGLES" in e for e in gp["errors"]), str(gp["errors"]))

    # ------------------------------------------------------------------ metric sanity
    A_ = np.zeros((100, 100), bool); A_[20:60, 20:60] = True
    B_ = np.zeros((100, 100), bool); B_[20:60, 23:63] = True
    mm = metrics.mask_metrics(A_, A_, {"legs": [0, 50, 100, 100]}, 50.0)
    check("metrics: identical masks -> IoU 1, boundary 0", mm["iou"] == 1 and mm["boundary_max_px"] == 0)
    mm = metrics.mask_metrics(A_, B_, {"legs": [0, 50, 100, 100]}, 50.0)
    check("metrics: 3px shift -> IoU 37/43, boundary HD95 3px",
          abs(mm["iou"] - 37 / 43) < 1e-9 and abs(mm["boundary_hd95_px"] - 3) < 1e-9,
          f"IoU {mm['iou']:.4f} mean {mm['boundary_mean_px']:.3f} hd95 {mm['boundary_hd95_px']} "
          f"legs_gap {mm['legs_gap_iou']:.4f}")

    # ------------------------------------------------------------------ tooling hash scope (§7)
    from catlib import provenance
    th0, files0 = provenance.tooling_hash()
    exp = [f"catlib/{n}.py" for n in ("__init__", "paths", "camera", "raster", "geom", "meshio", "metrics",
                                      "overlay", "provenance", "gpulock")]
    exp += [f"scripts/{n}" for n in paths.SHARED_SCRIPTS if (paths.SCRIPTS / n).is_file()]

    def touched(path):
        """hash as if `path` had one extra byte (in memory only: never writes shared files)."""
        target = (paths.ROOT / path).resolve()
        real = Path.read_bytes

        def fake(self):
            data = real(self)
            return data + b"\n" if Path(self).resolve() == target else data
        Path.read_bytes = fake
        try:
            return provenance.tooling_hash()[0]
        finally:
            Path.read_bytes = real
    h_ref = touched("catlib/refine.py") if (paths.CATLIB / "refine.py").is_file() else th0
    h_met = touched("catlib/metrics.py")
    check("tooling_hash: explicit evaluation file list; refine.py excluded, metrics.py included",
          files0 == exp and h_ref == th0 and h_met != th0 and provenance.tooling_hash()[0] == th0,
          f"{len(files0)} files")
    pv = provenance.provenance(paths.CAMERA_JSON)
    check("provenance records proxy_sha256", "proxy_sha256" in pv and pv["tooling_files"] == exp,
          str(pv["proxy_sha256"])[:16])

    # ------------------------------------------------------------------ occlusion tolerance (§7)
    check("metrics: occluded tolerance 60 mm for paw_*, 25 mm otherwise",
          metrics.occluded_tolerance("paw_HL") == 0.060 and metrics.occluded_tolerance("nose") == 0.025
          and metrics.occluded_tolerance("tail_tip") == 0.025)
    # a landmark 40 mm behind the first hit: occluded for 'nose', not for 'paw_FR'
    Vq = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0], [-1, -1, -.2], [1, -1, -.2], [0, 1, -.2]], float)
    Tq = np.array([[0, 1, 2], [3, 5, 4]])
    class _M:  # minimal CatMesh stand-in
        vertices, triangles = Vq, Tq
        landmarks = {"nose": np.array([0, 0, -0.04]), "paw_FR": np.array([0, 0, -0.04])}
    ocam = Camera(64, 64, 64.0, np.eye(4), name="occ")       # at z=1 looking down -Z
    ocam.matrix_world[:3, 3] = [0, 0, 1.0]
    lmq = metrics.landmark_metrics(_M(), ocam, {}, 50.0)
    check("metrics: 40 mm behind surface -> nose occluded, paw_FR not",
          lmq["landmarks"]["nose"]["occluded"] and not lmq["landmarks"]["paw_FR"]["occluded"],
          str(lmq["landmark_flags"]["occluded"]))

    # ------------------------------------------------------------------ landmark_attach (§12)
    da = fixtures.blob_cat(name="cat_fixture")
    att = meshio.attach_landmarks(da)
    ok_att = (not meshio.validate_dict(da)) and set(att) == set(meshio.LANDMARKS)
    db = json.loads(json.dumps(da))
    db["vertices"] = [[x + 0.01, y, z + 0.02] for x, y, z in db["vertices"]]
    stale = {k: list(v) for k, v in db["landmarks"].items()}       # stale on purpose
    mb = meshio.CatMesh(db)
    moved = all(np.allclose(mb.landmarks[k], np.array(stale[k]) + [0.01, 0, 0.02], atol=1e-9)
                for k in meshio.LANDMARKS)
    meshio.save(work / "attach" / "mesh.json", db)
    rb = json.loads((work / "attach" / "mesh.json").read_text())
    saved = (rb.get("landmark_attach") == db["landmark_attach"]
             and np.allclose(rb["landmarks"]["nose"], mb.landmarks["nose"], atol=1e-6))
    dbad = json.loads(json.dumps(da)); dbad["landmark_attach"]["nose"] = {"tri": 10 ** 6, "bary": [1, 0, 0]}
    check("landmark_attach: derive, follow geometry on load, round-trip save, bad index rejected",
          ok_att and moved and saved and any("landmark_attach.nose" in e for e in meshio.validate_dict(dbad)),
          f"valid {ok_att}, moved {moved}, saved {saved}")

    # ------------------------------------------------------------------ evaluate --no-render (synthetic)
    syn = work / "synthetic_derived"
    fixtures.synthetic_derived(good, syn, cam)
    pv = np.array(good["vertices"]) * 1.03
    np.savez(work / "proxy.npz", vertices=pv, triangles=np.array(good["triangles"]))
    vck = work / "variants" / "fx" / "checkpoints" / "v0"
    vck.mkdir(parents=True)
    shutil.copy(good_mj, vck / "mesh.json")
    p = subprocess.run([PY, str(paths.SCRIPTS / "evaluate.py"), "--mesh", str(vck / "mesh.json"),
                        "--no-render", "--derived-dir", str(syn), "--proxy", str(work / "proxy.npz")],
                       capture_output=True, text=True)
    ok = p.returncode == 0
    if ok:
        mt = json.loads((vck / "metrics.json").read_text())
        hist = json.loads((vck.parent.parent / "history.json").read_text())
        ok = (mt["official"] is False and mt["metrics"]["iou"] > 0.9 and len(hist) == 1
              and mt["metrics"]["secondary"]["chamfer_mm_vs_proxy"] is not None
              and (vck / "renders" / "ref_overlay_unofficial.png").is_file())
        detail = (f"IoU {mt['metrics']['iou']:.4f}, boundary {mt['metrics']['boundary_mean_px']:.2f}px, "
                  f"lm {mt['metrics']['landmark_sharp_mean_px']:.2f}px, novel-view IoU vs proxy "
                  f"{mt['metrics']['secondary']['novel_view_iou_vs_proxy']:.3f}, chamfer "
                  f"{mt['metrics']['secondary']['chamfer_mm_vs_proxy']:.2f} mm, {mt['elapsed_s']}s")
    else:
        detail = p.stdout[-800:] + p.stderr[-800:]
    check("evaluate --no-render on synthetic derived data (official=false, history row)", ok, detail)
    p = subprocess.run([PY, str(paths.SCRIPTS / "evaluate.py"), "--mesh", str(vck / "mesh.json"),
                        "--no-render", "--derived-dir", str(syn), "--proxy", str(work / "no_proxy.npz")],
                       capture_output=True, text=True)
    hist = json.loads((vck.parent.parent / "history.json").read_text())
    check("evaluate appends to history.json and nulls secondary without proxy",
          p.returncode == 0 and len(hist) == 2 and json.loads((vck / "metrics.json").read_text())
          ["metrics"]["secondary"]["chamfer_mm_vs_proxy"] is None)
    if not (paths.SCRIPTS / "render_variant.py").is_file():
        p = subprocess.run([PY, str(paths.SCRIPTS / "evaluate.py"), "--mesh", str(vck / "mesh.json"),
                            "--derived-dir", str(syn)], capture_output=True, text=True)
        check("evaluate without --no-render and no render script -> exit 3", p.returncode == 3, f"rc {p.returncode}")

    # ------------------------------------------------------------------ real derived data
    real = paths.derived_paths()
    if not a.skip_real and real["mask"].is_file() and real["landmarks"].is_file() and real["regions"].is_file():
        rck = work / "variants" / "real" / "checkpoints" / "fixture_vs_real"
        rck.mkdir(parents=True)
        shutil.copy(good_mj, rck / "mesh.json")
        p = subprocess.run([PY, str(paths.SCRIPTS / "evaluate.py"), "--mesh", str(rck / "mesh.json"),
                            "--no-render"], capture_output=True, text=True)
        ok = p.returncode == 0
        detail = p.stdout.strip().splitlines()[-2] if ok else p.stdout[-800:] + p.stderr[-800:]
        check("evaluate --no-render against real reference/derived (fixture mesh; numbers meaningless)", ok, detail)
    else:
        print("[SKIP] real reference/derived files not (all) present yet")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed in {time.time() - t_start:.1f}s; work dir {work}")
    (work / "test_results.json").write_text(json.dumps(
        {"results": [{"check": n_, "ok": o, "detail": d_} for n_, o, d_ in RESULTS], "raster_ms": timings},
        indent=1))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
