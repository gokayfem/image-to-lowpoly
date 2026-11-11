"""End-to-end test of the render tooling on a throw-away test cat.

    python3 tests/test_render.py [--quick] [--no-lock-test]

Steps (all output under .cache/render_test/):
  1. tests/make_test_blend.py -> cat_test.blend (CONTRACT section 6 layout, CPU only)
  2. scripts/render_variant.py: silhouette, clay, wire, beauty, --views, --turntable (unless --quick)
     -> checks every CONTRACT section 8 file exists with the right size/format, the silhouette is
        strictly binary, render_meta.json is complete, and the .blend was not modified.
  3. tests/blender_projection_check.py: formula vs world_to_camera_view (< 0.5 px), rendered
     marker centroids vs formula (< 0.5 px), numpy rasteriser vs Blender silhouette (IoU).
  4. GPU-lock serialisation: two render_variant runs launched at the same moment; from the
     timestamps in their logs, the second acquires the lock only after the first released it.
Exit code 0 only if everything passes. Functions are also pytest-collectable (test_*).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/
import _setup  # noqa: E402  (skill scripts on sys.path; LOWPOLY_CASE defaults to examples/cat)
REPO = _setup.REPO
ROOT = _setup.CASE
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
FFPROBE = "/opt/homebrew/bin/ffprobe"
WORK = ROOT / ".cache" / "render_test"
BLEND = WORK / "cat_test.blend"
VIEWS = ["front", "side_R", "back", "side_L", "threeq_front_L", "threeq_back_L", "threeq_back_R", "top"]


def blender(script: str, *args: str, log: Path | None = None) -> subprocess.CompletedProcess:
    cmd = [BLENDER, "-b", "--factory-startup", "--python", str(script if Path(script).is_absolute() else REPO / script), "--", *args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if log:
        log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        raise AssertionError(f"{script} failed ({r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_make_blend():
    WORK.mkdir(parents=True, exist_ok=True)
    r = blender("tests/make_test_blend.py", "--out", str(BLEND))
    assert BLEND.exists(), r.stdout
    print(" ", [l for l in r.stdout.splitlines() if "make_test_blend" in l][0])


def test_render_all(turntable: bool = True):
    out = WORK / "full"
    before = sha(BLEND)
    args = ["--blend", str(BLEND), "--out", str(out), "--passes", "silhouette,clay,wire,beauty", "--views"]
    if turntable:
        args.append("--turntable")
    t0 = time.time()
    blender(str(_setup.SCRIPTS / "render_variant.py"), *args, log=WORK / "full_log.txt")
    dt = time.time() - t0
    assert sha(BLEND) == before, "render_variant modified the variant .blend"
    expect = {"ref_silhouette.png": 1536, "ref_clay.png": 1536, "ref_wire.png": 1536, "ref_beauty.png": 1536}
    for v in VIEWS:
        expect[f"view_{v}_clay.png"] = 1024
        expect[f"view_{v}_wire.png"] = 1024
    for name, size in expect.items():
        im = Image.open(out / name)
        assert im.size == (size, size), (name, im.size)
    sil = np.asarray(Image.open(out / "ref_silhouette.png").convert("L"))
    vals = set(np.unique(sil).tolist())
    assert vals <= {0, 255}, f"silhouette not binary: {sorted(vals)[:10]}"
    meta = json.loads((out / "render_meta.json").read_text())
    for k in ("engine", "device", "samples_shaded", "passes_seconds", "blender_version", "camera_sha256",
              "scene_sha256", "lights_created", "silhouette", "triangles_rendered"):
        assert k in meta, k
    assert meta["silhouette"]["agreement_with_id_pass"] == 1.0
    # wire must actually differ from clay (lines drawn) and be darker on average
    clay = np.asarray(Image.open(out / "ref_clay.png")).astype(float)
    wire = np.asarray(Image.open(out / "ref_wire.png")).astype(float)
    line_px = int(((clay - wire).max(-1) > 40).sum())
    assert line_px > 1000, f"too few wire pixels: {line_px}"
    if turntable:
        for kind in ("clay_wire", "beauty"):
            p = out / f"turntable_{kind}.mp4"
            r = subprocess.run([FFPROBE, "-v", "error", "-show_entries",
                                "stream=codec_name,pix_fmt,width,height,nb_frames,r_frame_rate", "-of", "json", str(p)],
                               capture_output=True, text=True, check=True)
            s = json.loads(r.stdout)["streams"][0]
            assert (s["codec_name"], s["pix_fmt"], s["width"], s["height"], int(s["nb_frames"]), s["r_frame_rate"]) == \
                ("h264", "yuv420p", 768, 768, 72, "24/1"), s
        assert not (out / "turntable_frames").exists(), "frames should be deleted by default"
    print(f"  full render ok in {dt:.1f}s wall; per pass {meta['passes_seconds']}; device {meta['device']}; "
          f"silhouette {meta['silhouette']}; wire line pixels {line_px}")
    return meta


def test_projection():
    out = WORK / "projection.json"
    blender("tests/blender_projection_check.py", "--blend", str(BLEND), "--out", str(out))
    r = json.loads(out.read_text())
    print(f"  formula vs world_to_camera_view max {r['formula_vs_w2cv_px']['max']:.5f}px; "
          f"marker centroid vs formula max {r['formula_vs_marker_px']['max']:.3f}px; "
          f"raster vs render IoU {r['raster_vs_render_iou']:.6f} (xor {r['xor_px']} px); "
          f"8 novel views renderer vs catlib.camera max {r['novel_views_max_px']:.5f}px "
          f"(radius {r['orbit_radius_m']} m)")
    assert r["pass"], r
    return r


LOCK_RE = re.compile(r"\[gpulock \S+ t=([\d.]+) pid=(\d+)\] (acquired|released|waiting)")


def test_lock_serialises():
    procs = []
    for tag in ("A", "B"):
        cmd = [BLENDER, "-b", "--factory-startup", "--python", str(_setup.SCRIPTS / "render_variant.py"), "--",
               "--blend", str(BLEND), "--out", str(WORK / f"lock_{tag}"), "--passes", "silhouette,clay"]
        f = open(WORK / f"lock_{tag}_log.txt", "w")
        procs.append((tag, subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True), f, time.time()))
    for tag, p, f, _ in procs:
        assert p.wait() == 0, f"lock run {tag} failed"
        f.close()
    ev = {}
    for tag, _p, _f, t_start in procs:
        lines = (WORK / f"lock_{tag}_log.txt").read_text().splitlines()
        d = {"started": t_start, "waiting": []}
        for l in lines:
            m = LOCK_RE.search(l)
            if m:
                if m.group(3) == "waiting":
                    d["waiting"].append(float(m.group(1)))
                else:
                    d[m.group(3)] = float(m.group(1))
        ev[tag] = d
    first, second = sorted(ev, key=lambda k: ev[k]["acquired"])
    a, b = ev[first], ev[second]
    print(f"  launched {abs(ev['A']['started'] - ev['B']['started']):.3f}s apart; "
          f"{first}: acquired {a['acquired']:.3f} released {a['released']:.3f}; "
          f"{second}: acquired {b['acquired']:.3f} (waited {b['acquired'] - b['started']:.2f}s, "
          f"'waiting' logged at {[round(x, 3) for x in b['waiting']]}) released {b['released']:.3f}")
    assert b["acquired"] >= a["released"] - 0.01, "second process acquired the lock before the first released it"
    assert b["waiting"], "second process never reported waiting"
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the turntable")
    ap.add_argument("--no-lock-test", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    steps = [("make test blend", test_make_blend),
             ("render all passes" + ("" if a.quick else " + views + turntable"), lambda: test_render_all(not a.quick)),
             ("projection self-test", test_projection)]
    if not a.no_lock_test:
        steps.append(("gpu lock serialisation", test_lock_serialises))
    failed = 0
    for name, fn in steps:
        print(f"[test_render] {name} ...", flush=True)
        try:
            fn()
            print(f"[test_render] PASS {name}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[test_render] FAIL {name}: {e}", flush=True)
    print(f"[test_render] {'ALL PASSED' if not failed else f'{failed} FAILED'} in {time.time() - t0:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
