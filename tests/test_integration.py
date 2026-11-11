"""Integration-tooling test on STAND-IN data (synthetic blob cats; NOT reconstructions).

    python3 tests/test_integration.py [--phase standins|finalize|outputs|negative|all]
                                               [--work .cache/integ_test] [--rebuild]

Phases (each reuses what earlier phases left in --work, so a failing late phase can be
re-run without re-rendering):

  standins  build four stand-in catmesh-1 meshes (budgets 100/200/500/1000, different
            ring/segment counts, all <= budget, deliberately NOT all == budget) with two
            checkpoints each (v0_initial, v1) under <work>/variants/<N>/checkpoints/ and
            evaluate every one with `scripts/evaluate.py --views` (official Blender
            renders; evaluate.py takes the GPU lock itself).  Also one stand-in decimation
            baseline at <work>/baselines/decimate/100 and a stand-in proxy fit report.
  finalize  scripts/finalize_variant.py for each budget (--allow-provisional).  Only the
            100 stand-in renders a turntable (~3 min); the others use --no-turntable.
  outputs   make_comparison (finals and --checkpoints mode), make_critic_packet,
            aggregate_metrics, make_comparison_blend, make_turntable_mosaic into
            <work>/outputs and checks every artefact.
  negative  aggregate_metrics must fail loudly on: a mismatched tooling hash, an
            over-budget count, disagreeing counts, a non-official final, a provisional
            camera without --allow-provisional.

The stand-in meshes are made of separate closed primitives (ellipsoids, prisms,
tetrahedra) purely to exercise the tooling.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/
import _setup  # noqa: E402  (skill scripts on sys.path; LOWPOLY_CASE defaults to examples/cat)
REPO = _setup.REPO
ROOT = _setup.CASE

import numpy as np  # noqa: E402

import fixtures  # noqa: E402
from catlib import geom, meshio, paths  # noqa: E402

PY = str(paths.VENV_PYTHON)
SCRIPTS = paths.SCRIPTS
BUDGETS = (100, 200, 500, 1000)
RESULTS: list[tuple[str, bool, str]] = []

# torso (seg, rings), head (seg, rings), leg prism sides, tail prism sides
STANDIN_CFG = {
    100: dict(torso=(6, 3), head=(5, 3), leg_n=3, tail_n=3),     # 24+20+8+32+8   = 92
    200: dict(torso=(8, 5), head=(8, 4), leg_n=4, tail_n=4),     # 64+48+8+48+12  = 180
    500: dict(torso=(14, 9), head=(12, 7), leg_n=6, tail_n=6),   # 224+144+8+80+20 = 476
    1000: dict(torso=(22, 12), head=(16, 10), leg_n=8, tail_n=8),  # 484+288+8+112+28 = 920
}


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)
    return ok


def run(cmd: list, log: Path | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    print("$ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("$ " + " ".join(cmd) + "\n--- stdout\n" + p.stdout + "\n--- stderr\n" + p.stderr)
    return p


# --------------------------------------------------------------------------- stand-in meshes
def prism(p0, p1, r, n):
    """Closed n-sided prism from p0 to p1: 2n side + 2(n-2) cap triangles."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = p1 - p0
    ax /= np.linalg.norm(ax)
    h = np.array([1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1.0, 0])
    a = np.cross(ax, h); a /= np.linalg.norm(a)
    b = np.cross(ax, a)
    ring = [r * (np.cos(2 * np.pi * k / n + np.pi / n) * a + np.sin(2 * np.pi * k / n + np.pi / n) * b)
            for k in range(n)]
    V = [p0 + q for q in ring] + [p1 + q for q in ring]
    T = []
    for k in range(1, n - 1):
        T.append((0, k + 1, k))
        T.append((n, n + k, n + k + 1))
    for k in range(n):
        k1 = (k + 1) % n
        T += [(k, k1, n + k1), (k, n + k1, n + k)]
    return fixtures.orient_outward(np.array(V), np.array(T))


def standin_cat(budget: int, grow: float = 1.0) -> dict:
    c = STANDIN_CFG[budget]
    tr = (0.20 * grow, 0.08 * grow, 0.075 * grow)
    parts = [
        ("torso", "fur_orange", fixtures.ellipsoid((0.0, 0.0, 0.17), tr, *c["torso"])),
        ("head", "fur_orange", fixtures.ellipsoid((0.23, -0.02, 0.29), (0.06, 0.065, 0.06), *c["head"])),
        ("ears", "ear_inner", fixtures.tetra((0.23, -0.055, 0.335), (0.235, -0.07, 0.40), 0.022,
                                              np.array([1.0, 0, 0]))),
        ("ears", "ear_inner", fixtures.tetra((0.23, 0.015, 0.335), (0.235, 0.03, 0.40), 0.022,
                                              np.array([1.0, 0, 0]))),
        ("leg_FR", "fur_cream", prism((0.14, -0.045, 0.0), (0.14, -0.045, 0.16), 0.02, c["leg_n"])),
        ("leg_FL", "fur_cream", prism((0.10, 0.045, 0.0), (0.10, 0.045, 0.16), 0.02, c["leg_n"])),
        ("leg_HR", "fur_cream", prism((-0.14, -0.045, 0.0), (-0.14, -0.045, 0.16), 0.022, c["leg_n"])),
        ("leg_HL", "fur_cream", prism((-0.11, 0.045, 0.0), (-0.11, 0.045, 0.16), 0.022, c["leg_n"])),
        ("tail", "fur_stripe", prism((-0.19, 0.0, 0.20), (-0.33, 0.0, 0.42), 0.016, c["tail_n"])),
    ]
    Vs, Ts, reg, mat = [], [], [], []
    off = 0
    for r, mt, (V, T) in parts:
        Vs.append(V); Ts.append(T + off); off += len(V)
        reg += [r] * len(T); mat += [mt] * len(T)
    V = np.concatenate(Vs); T = np.concatenate(Ts)
    rough = {
        "ear_tip_R": (0.235, -0.07, 0.40), "ear_tip_L": (0.235, 0.03, 0.40),
        "eye_R": (0.28, -0.045, 0.30), "eye_L": (0.28, 0.005, 0.30), "nose": (0.29, -0.02, 0.28),
        "paw_FR": (0.14, -0.045, 0.0), "paw_FL": (0.10, 0.045, 0.0),
        "paw_HR": (-0.14, -0.045, 0.0), "paw_HL": (-0.11, 0.045, 0.0),
        "tail_base": (-0.19, 0.0, 0.20), "tail_mid": (-0.26, -0.01, 0.31), "tail_tip": (-0.33, 0.0, 0.43),
        "shoulder_R": (0.13, -0.08, 0.2), "hip_R": (-0.13, -0.08, 0.18), "hock_HR": (-0.15, -0.07, 0.06),
        "chin": (0.25, -0.02, 0.23),
    }
    _, snapped = geom.point_mesh_distance(np.array(list(rough.values())), V, T)
    d = meshio.to_dict(V, T, reg, mat, dict(zip(rough, snapped)), budget, f"cat_{budget}",
                       notes="STAND-IN for integration-tooling tests (blob primitives), not a reconstruction")
    assert len(T) <= budget, (budget, len(T))
    return d


def evaluate(mesh_json: Path, *extra) -> subprocess.CompletedProcess:
    p = run([PY, SCRIPTS / "evaluate.py", "--mesh", mesh_json, *extra],
            log=mesh_json.parent / "logs" / "evaluate_cli.log")
    return p


def phase_standins(work: Path, rebuild: bool) -> None:
    for N in BUDGETS:
        vdir = work / "variants" / str(N)
        for tag, grow in (("v0_initial", 1.0), ("v1", 1.15)):
            ck = vdir / "checkpoints" / tag
            mj = ck / "mesh.json"
            if (ck / "metrics.json").is_file() and not rebuild:
                print(f"standin {N}/{tag}: already evaluated, reusing")
                continue
            d = standin_cat(N, grow)
            meshio.save(mj, d)
            p = evaluate(mj, "--views")
            check(f"standin {N}/{tag}: evaluate.py --views exit 0 ({len(d['triangles'])} tris)",
                  p.returncode == 0, (p.stdout + p.stderr).strip().splitlines()[-1] if p.returncode else "")
    # one stand-in decimation baseline (only budget 100) evaluated the official way
    bdir = work / "baselines" / "decimate" / "100"
    if not (bdir / "metrics.json").is_file() or rebuild:
        d = standin_cat(100, 0.95)
        d["notes"] = "STAND-IN baseline for integration tests"
        meshio.save(bdir / "mesh.json", d)
        p = evaluate(bdir / "mesh.json")
        check("standin baseline decimate/100: evaluate exit 0", p.returncode == 0)
    # stand-in proxy fit report
    (work / "proxy").mkdir(parents=True, exist_ok=True)
    (work / "proxy" / "fit_report.json").write_text(json.dumps({
        "note": "STAND-IN proxy report for integration tests",
        "metrics": {"iou": 0.91, "boundary_mean_px": 6.5, "boundary_hd95_px": 18.0,
                    "landmark_sharp_mean_px": 14.0, "legs_gap_iou": 0.8}}, indent=1))


# --------------------------------------------------------------------------- phases using my scripts
def common(work: Path) -> list:
    return ["--variants-root", work / "variants", "--out", work / "outputs"]


def phase_finalize(work: Path) -> None:
    for N in BUDGETS:
        args = [PY, SCRIPTS / "finalize_variant.py", "--variants-root", work / "variants", "--budget", N,
                "--checkpoint", "v1", "--reason", "stand-in test: v1 chosen to exercise finalize",
                "--allow-provisional", "--force"]
        if N != 100:
            args.append("--no-turntable")
        p = run(args, log=work / "logs" / f"finalize_{N}.log", timeout=1800)
        fin = work / "variants" / str(N) / "final"
        ok = p.returncode == 0
        check(f"finalize {N}: exit 0", ok, (p.stdout + p.stderr).strip()[-600:] if not ok else "")
        need = [f"cat_{N}.blend", f"cat_{N}.glb", "mesh.json", "metrics.json", "validation.json", "CHOSEN.json",
                "renders/ref_beauty.png", "renders/ref_clay.png", "renders/ref_wire.png",
                "renders/ref_silhouette.png", "renders/view_side_L_wire.png"]
        if N == 100:
            need += ["renders/turntable_clay_wire.mp4", "renders/turntable_beauty.mp4"]
        missing = [x for x in need if not (fin / x).is_file()]
        check(f"finalize {N}: final/ complete", not missing, f"missing {missing}" if missing else "")
        if ok:
            ch = json.loads((fin / "CHOSEN.json").read_text())
            check(f"finalize {N}: CHOSEN.json records checkpoint + reason",
                  ch.get("checkpoint") == "v1" and "stand-in" in ch.get("reason", ""))
    # refusal: unlocked camera without --allow-provisional must be refused (camera is provisional now)
    cam = json.loads(paths.CAMERA_JSON.read_text())
    if cam.get("status") != "locked":
        p = run([PY, SCRIPTS / "finalize_variant.py", "--variants-root", work / "variants", "--budget", 200,
                 "--checkpoint", "v0_initial", "--reason", "should be refused", "--no-turntable",
                 "--dry-run"])
        check("finalize refuses a provisional-camera checkpoint without --allow-provisional",
              p.returncode != 0 and "camera" in (p.stdout + p.stderr).lower(), (p.stdout + p.stderr).strip()[-300:])


def png_size(p: Path):
    from PIL import Image
    with Image.open(p) as im:
        return im.size, im.info


def phase_outputs(work: Path) -> None:
    out = work / "outputs"
    if out.exists():
        shutil.rmtree(out)
    base = common(work)
    extra = ["--baselines-root", work / "baselines", "--proxy-report", work / "proxy" / "fit_report.json"]

    p = run([PY, SCRIPTS / "make_comparison.py", *base, *extra], log=work / "logs" / "make_comparison.log")
    check("make_comparison (finals): exit 0", p.returncode == 0, p.stderr[-800:])
    for f in ("comparison_reference_plus_four.png", "comparison_clay_wire.png", "comparison_multiview_clay.png",
              "comparison_multiview_wire.png", "comparison_metrics_table.png"):
        check(f"make_comparison wrote {f}", (out / f).is_file(),
              str(png_size(out / f)[0]) if (out / f).is_file() else "")

    p = run([PY, SCRIPTS / "make_comparison.py", *common(work)[:2], "--out", work / "outputs" / "midrun",
             "--checkpoints", "100:v0_initial,200:v1,500:v0_initial,1000:v1", *extra],
            log=work / "logs" / "make_comparison_midrun.log")
    check("make_comparison --checkpoints: exit 0", p.returncode == 0, p.stderr[-800:])

    p = run([PY, SCRIPTS / "make_critic_packet.py", *base, "--round", 1], log=work / "logs" / "critic.log")
    check("make_critic_packet: exit 0", p.returncode == 0, p.stderr[-800:])
    pk = out / "critic" / "round_1"
    expect = {"reference.png", "sheet_clay_wire.png", "PACKET.md"}
    for N in BUDGETS:
        expect |= {f"budget_{N}_{k}.png" for k in ("clay", "wire", "beauty", "overlay_outline", "views_clay")}
    got = {x.name for x in pk.iterdir()} if pk.is_dir() else set()
    check("critic packet: exactly the allowed files", got == expect,
          f"extra {sorted(got - expect)} missing {sorted(expect - got)}")
    banned = ("iou", "score", "metric", "initial", "final", "v0", "v1", "checkpoint", "hd95", "boundary",
              "landmark", "px")
    bad_names = [x for x in got if any(b in x.lower() for b in banned)]
    check("critic packet: no score/tag words in file names", not bad_names, str(bad_names))
    meta_bad = []
    for f in sorted(pk.glob("*.png")):
        _, info = png_size(f)
        if any(k.lower() not in ("dpi", "srgb", "gamma", "chromaticity") for k in info):
            meta_bad.append((f.name, list(info)))
    check("critic packet: PNGs carry no text metadata", not meta_bad, str(meta_bad))
    md = (pk / "PACKET.md").read_text().lower() if (pk / "PACKET.md").is_file() else ""
    check("critic packet: PACKET.md free of score words",
          md and not any(w in md for w in ("iou", "score", "metric", "checkpoint", "initial", "final", "hd95")))
    p = run([PY, SCRIPTS / "make_critic_packet.py", *common(work)[:2], "--out", work / "outputs",
             "--round", 2, "--checkpoints", "100:v0_initial,200:v0_initial,500:v0_initial,1000:v0_initial"])
    check("make_critic_packet --checkpoints: exit 0", p.returncode == 0, p.stderr[-500:])

    p = run([PY, SCRIPTS / "aggregate_metrics.py", *base, *extra, "--allow-provisional"],
            log=work / "logs" / "aggregate.log")
    check("aggregate_metrics (provisional allowed): exit 0", p.returncode == 0, (p.stdout + p.stderr)[-800:])
    if (out / "metrics.json").is_file():
        m = json.loads((out / "metrics.json").read_text())
        rows = {r["budget"]: r for r in m["variants"]}
        ok = all(rows[N]["triangles"]["all_agree"] and rows[N]["triangles"]["final"] <= N for N in BUDGETS)
        check("aggregate: four variants, counts agree and within budget", set(rows) == set(BUDGETS) and ok)
        check("aggregate: initial and final metrics present",
              all(rows[N]["initial"]["metrics"]["iou"] is not None and rows[N]["final"]["metrics"]["iou"]
                  is not None for N in BUDGETS))
        check("aggregate: unused budget reported for stand-in 100 (92 tris)",
              rows[100]["triangles"]["unused_budget"] == 8, str(rows[100]["triangles"]))
        check("aggregate: baseline row picked up", bool(rows[100]["baselines"].get("decimate")))
        check("aggregate: proxy report included", m.get("proxy_fit_report") is not None)
        check("aggregate: CSV written", (out / "metrics.csv").is_file())

    p = run([PY, SCRIPTS / "make_comparison_blend.py", *base], log=work / "logs" / "comparison_blend.log",
            timeout=900)
    check("make_comparison_blend: exit 0 (reopen + evaluated loop-triangle counts match)", p.returncode == 0,
          (p.stdout + p.stderr)[-1200:])
    check("comparison.blend exists", (out / "comparison.blend").is_file())
    print("\n".join(l for l in p.stdout.splitlines() if "CAT_" in l or "PRESENTATION" in l))

    p = run([PY, SCRIPTS / "make_turntable_mosaic.py", *base, "--allow-missing"],
            log=work / "logs" / "mosaic.log", timeout=900)
    check("make_turntable_mosaic --allow-missing: exit 0", p.returncode == 0, (p.stdout + p.stderr)[-800:])
    for f in ("turntable_all_clay_wire.mp4", "turntable_all_beauty.mp4"):
        ok = (out / f).is_file()
        detail = ""
        if ok:
            q = subprocess.run(["/opt/homebrew/bin/ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                                "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_read_frames",
                                "-of", "json", str(out / f)], capture_output=True, text=True)
            s = json.loads(q.stdout)["streams"][0]
            ok = s["codec_name"] == "h264" and s["pix_fmt"] == "yuv420p" and int(s["nb_read_frames"]) == 72
            detail = str(s)
        check(f"mosaic {f}: h264 yuv420p 72 frames", ok, detail)
    check("per-variant turntable copied", (out / "turntables" / "turntable_100_clay_wire.mp4").is_file())
    p = run([PY, SCRIPTS / "make_turntable_mosaic.py", *common(work)[:2], "--out", work / "outputs" / "strict"])
    check("make_turntable_mosaic without --allow-missing fails when turntables are missing", p.returncode != 0,
          (p.stdout + p.stderr).strip()[-200:])


def phase_negative(work: Path) -> None:
    """Copy the four finals into scratch variant roots, corrupt one thing, expect a loud failure."""
    src = work / "variants"

    def scratch(name: str) -> Path:
        dst = work / "neg" / name / "variants"
        if dst.exists():
            shutil.rmtree(dst)
        for N in BUDGETS:
            s = src / str(N)
            d = dst / str(N)
            d.mkdir(parents=True)
            shutil.copytree(s / "final", d / "final", ignore=shutil.ignore_patterns("*.mp4", "*.blend", "*.glb"))
            for f in ("history.json",):
                if (s / f).is_file():
                    shutil.copy2(s / f, d / f)
            shutil.copytree(s / "checkpoints", d / "checkpoints",
                            ignore=shutil.ignore_patterns("*.mp4", "*.blend", "*.glb", "*.png"))
        return dst

    def edit(path: Path, fn) -> None:
        d = json.loads(path.read_text())
        fn(d)
        path.write_text(json.dumps(d, indent=1))

    def expect_fail(name: str, needle: str, extra=(), mutate=None):
        root = scratch(name)
        if mutate:
            mutate(root)
        p = run([PY, SCRIPTS / "aggregate_metrics.py", "--variants-root", root, "--out", root.parent / "out",
                 "--baselines-root", root.parent / "nobaselines", *extra])
        msg = (p.stdout + p.stderr)
        check(f"aggregate negative [{name}]: non-zero exit and message mentions '{needle}'",
              p.returncode != 0 and needle.lower() in msg.lower(), msg.strip().splitlines()[-1] if msg.strip() else "")

    def tooling(root):
        edit(root / "500" / "final" / "metrics.json",
             lambda d: d["provenance"].__setitem__("tooling_hash", "0" * 64))
    expect_fail("tooling_hash", "tooling_hash", ["--allow-provisional"], tooling)

    def over(root):
        def f(d):
            d["counts"]["A_mesh_json"] = d["counts"]["B_blend_evaluated"] = d["counts"]["C_glb_reimported"] = \
                d["counts"]["D_glb_parsed"] = d["counts"]["final_triangles"] = 1001
        edit(root / "1000" / "final" / "validation.json", f)
    expect_fail("over_budget", "exceeds budget", ["--allow-provisional"], over)

    def disagree(root):
        edit(root / "200" / "final" / "validation.json",
             lambda d: (d["counts"].__setitem__("C_glb_reimported", 181), d["counts"].__setitem__("all_agree", False)))
    expect_fail("count_disagree", "disagree", ["--allow-provisional"], disagree)

    def unofficial(root):
        edit(root / "100" / "final" / "metrics.json", lambda d: d.__setitem__("official", False))
    expect_fail("unofficial", "official", ["--allow-provisional"], unofficial)

    cam = json.loads(paths.CAMERA_JSON.read_text())
    if cam.get("status") != "locked":
        expect_fail("camera_provisional", "camera", [])
    # sanity: an untouched scratch copy passes
    root = scratch("control")
    p = run([PY, SCRIPTS / "aggregate_metrics.py", "--variants-root", root, "--out", root.parent / "out",
             "--baselines-root", root.parent / "nobaselines", "--allow-provisional"])
    check("aggregate control copy (untouched) passes", p.returncode == 0, (p.stdout + p.stderr)[-400:])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase", default="all", choices=["standins", "finalize", "outputs", "negative", "all"])
    ap.add_argument("--work", type=Path, default=paths.CACHE / "integ_test")
    ap.add_argument("--rebuild", action="store_true", help="re-evaluate stand-in checkpoints even if present")
    a = ap.parse_args()
    work = a.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    phases = ["standins", "finalize", "outputs", "negative"] if a.phase == "all" else [a.phase]
    for ph in phases:
        print(f"\n===== phase {ph} =====", flush=True)
        {"standins": lambda: phase_standins(work, a.rebuild), "finalize": lambda: phase_finalize(work),
         "outputs": lambda: phase_outputs(work), "negative": lambda: phase_negative(work)}[ph]()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed in {time.time() - t0:.0f}s")
    for name, ok, d in RESULTS:
        if not ok:
            print("FAILED:", name, d)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
