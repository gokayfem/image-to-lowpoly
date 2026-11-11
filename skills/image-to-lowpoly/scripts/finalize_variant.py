"""Promote one evaluated checkpoint of one budget to variants/<N>/final/ and re-evaluate it there.

    python3 scripts/finalize_variant.py --budget 500 --checkpoint v3 \
        --reason "best IoU/boundary/landmarks without leg-gap regression; critic round 2 clean" \
        [--variants-root variants] [--allow-provisional] [--force] [--no-turntable] [--dry-run]

Steps
  1. Refuse unless checkpoints/<TAG>/ holds official metrics.json + validation.json with no
     hard failures, all four triangle counts agreeing and <= N, render ok, no evaluation
     errors, and a LOCKED camera (--allow-provisional overrides only the camera check; it is
     for tooling tests while contract/camera.json is still provisional).
  2. Copy checkpoints/<TAG>/mesh.json -> final/mesh.json (byte-identical; sha checked).
  3. Run the official `scripts/evaluate.py --mesh final/mesh.json --views --turntable
     --tag final` there (it assembles cat_<N>.blend / .glb, validates, renders under the GPU
     lock it takes itself, writes metrics.json / validation.json, appends a history row
     tagged "final").  --no-turntable is for tooling tests only and is recorded.
  4. Check the final evaluation reproduces the checkpoint (same mesh hash, same counts,
     IoU within 1e-6) and write final/CHOSEN.json (checkpoint, reason, hashes, both metric
     sets, command lines, time).

Exit 0 on success; 2 refused; 3 final evaluation failed or did not reproduce.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the folder that holds catlib/
from catlib import paths as _paths  # noqa: E402  (resolves the case: --case / LOWPOLY_CASE / cwd)
ROOT = _paths.ROOT
sys.path.insert(0, str(_paths.SCRIPTS))

from catlib import paths, provenance  # noqa: E402
from integ_common import headline, read_json  # noqa: E402


def refusal_reasons(ck: Path, N: int, allow_provisional: bool) -> list[str]:
    r = []
    mp, vp, mj = ck / "metrics.json", ck / "validation.json", ck / "mesh.json"
    for p in (mj, mp, vp):
        if not p.is_file():
            r.append(f"missing {p}")
    if r:
        return r
    m, v = read_json(mp), read_json(vp)
    if not m.get("official"):
        r.append("metrics.json is not official (raster-only / --no-render evaluation)")
    if m.get("provenance", {}).get("mesh_sha256") != provenance.sha256_file(mj):
        r.append("metrics.json was computed for a different mesh.json (stale): re-evaluate the checkpoint")
    hard = v.get("hard_failures") or m.get("validation", {}).get("hard_failures") or []
    if hard:
        r.append(f"hard failures: {hard}")
    c = v.get("counts", {})
    if not c.get("all_agree"):
        r.append(f"triangle counts disagree: {c}")
    if c.get("final_triangles") is None or c["final_triangles"] > N:
        r.append(f"triangle count {c.get('final_triangles')} exceeds budget {N}")
    if m.get("budget") != N:
        r.append(f"mesh budget {m.get('budget')} != --budget {N}")
    if not (m.get("render") or {}).get("ok"):
        r.append(f"render not ok: {(m.get('render') or {}).get('error')}")
    if m.get("evaluation_errors"):
        r.append(f"evaluation errors: {m['evaluation_errors']}")
    cam_now = read_json(paths.CAMERA_JSON).get("status")
    if m.get("camera_status") != "locked" or cam_now != "locked":
        msg = (f"camera is not locked (checkpoint scored with {m.get('camera_status')!r}, "
               f"contract/camera.json is {cam_now!r})")
        if allow_provisional:
            print("WARNING (allowed by --allow-provisional):", msg)
        else:
            r.append(msg + "; pass --allow-provisional only for tooling tests")
    if m.get("provenance", {}).get("camera_sha256") != provenance.sha256_file(paths.CAMERA_JSON):
        msg = "checkpoint was scored with a different camera.json than the current one"
        if allow_provisional:
            print("WARNING (allowed by --allow-provisional):", msg)
        else:
            r.append(msg)
    th, _ = provenance.tooling_hash()
    if m.get("provenance", {}).get("tooling_hash") != th:
        print("NOTE: checkpoint was scored with older tooling; the final re-evaluation uses the current "
              "tooling and its numbers are the ones reported.")
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Copy a validated checkpoint to final/ and re-run the official evaluation.")
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--checkpoint", required=True, help="checkpoint tag under variants/<N>/checkpoints/")
    ap.add_argument("--reason", required=True, help="why this checkpoint was chosen (recorded in CHOSEN.json)")
    ap.add_argument("--variants-root", type=Path, default=paths.VARIANTS)
    ap.add_argument("--out", type=Path, default=paths.OUTPUTS, help="unused; accepted for CLI uniformity")
    ap.add_argument("--allow-provisional", action="store_true", help="tooling tests only: accept an unlocked camera")
    ap.add_argument("--force", action="store_true", help="replace an existing final/")
    ap.add_argument("--no-turntable", action="store_true", help="tooling tests only: skip the ~3 min turntable")
    ap.add_argument("--dry-run", action="store_true", help="only run the refusal checks")
    a = ap.parse_args(argv)

    N = a.budget
    vdir = (a.variants_root / str(N)).resolve()
    ck = vdir / "checkpoints" / a.checkpoint
    if not ck.is_dir():
        print(f"finalize: {ck} does not exist", file=sys.stderr)
        return 2
    reasons = refusal_reasons(ck, N, a.allow_provisional)
    if reasons:
        print(f"finalize REFUSED for budget {N} checkpoint {a.checkpoint}:", file=sys.stderr)
        for x in reasons:
            print("  -", x, file=sys.stderr)
        return 2
    if a.dry_run:
        print(f"finalize: checkpoint {a.checkpoint} of budget {N} is acceptable (dry run)")
        return 0

    fin = vdir / "final"
    if fin.exists():
        if not a.force:
            print(f"finalize: {fin} exists; pass --force to replace it", file=sys.stderr)
            return 2
        shutil.rmtree(fin)
    fin.mkdir(parents=True)
    src_mesh = ck / "mesh.json"
    shutil.copy2(src_mesh, fin / "mesh.json")
    if provenance.sha256_file(src_mesh) != provenance.sha256_file(fin / "mesh.json"):
        print("finalize: copy of mesh.json is not byte-identical", file=sys.stderr)
        return 3

    cmd = [str(paths.VENV_PYTHON), str(paths.SCRIPTS / "evaluate.py"), "--mesh", str(fin / "mesh.json"),
           "--views", "--tag", "final"]
    if not a.no_turntable:
        cmd.append("--turntable")
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = round(time.time() - t0, 1)
    if p.returncode != 0:
        print(f"finalize: final evaluation failed (exit {p.returncode}); see {fin / 'logs'}", file=sys.stderr)
        return 3

    m_ck, m_fin = read_json(ck / "metrics.json"), read_json(fin / "metrics.json")
    v_fin = read_json(fin / "validation.json")
    problems = []
    if m_fin["provenance"]["mesh_sha256"] != m_ck["provenance"]["mesh_sha256"]:
        problems.append("final mesh hash differs from checkpoint")
    if v_fin["counts"].get("final_triangles") != read_json(ck / "validation.json")["counts"].get("final_triangles"):
        problems.append("final triangle count differs from checkpoint")
    i_ck, i_fin = m_ck["metrics"].get("iou"), m_fin["metrics"].get("iou")
    same_tooling = m_ck["provenance"]["tooling_hash"] == m_fin["provenance"]["tooling_hash"]
    if same_tooling and (i_ck is None or i_fin is None or abs(i_ck - i_fin) > 1e-6):
        problems.append(f"final IoU {i_fin} does not reproduce checkpoint IoU {i_ck} under identical tooling")
    if not m_fin.get("official"):
        problems.append("final metrics not official")
    rd = fin / "renders"
    want = ["ref_silhouette.png", "ref_clay.png", "ref_wire.png", "ref_beauty.png", "ref_overlay.png"]
    if not a.no_turntable:
        want += ["turntable_clay_wire.mp4", "turntable_beauty.mp4"]
    missing = [w for w in want if not (rd / w).is_file()] + \
              ([] if len(list(rd.glob("view_*_clay.png"))) == 8 else ["8 view_*_clay.png"])
    if missing:
        problems.append(f"missing renders {missing}")

    chosen = {
        "budget": N,
        "checkpoint": a.checkpoint,
        "reason": a.reason,
        "source_dir": str(ck.relative_to(ROOT) if ck.is_relative_to(ROOT) else ck),
        "mesh_sha256": m_fin["provenance"]["mesh_sha256"],
        "final_triangles": v_fin["counts"].get("final_triangles"),
        "counts": v_fin["counts"],
        "camera_status": m_fin.get("camera_status"),
        "allow_provisional": a.allow_provisional,
        "turntable_rendered": not a.no_turntable,
        "checkpoint_headline": headline(m_ck),
        "final_headline": headline(m_fin),
        "same_tooling_as_checkpoint": same_tooling,
        "reproduction_problems": problems,
        "command": " ".join(cmd),
        "evaluate_elapsed_s": elapsed,
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (fin / "CHOSEN.json").write_text(json.dumps(chosen, indent=1, default=float))
    if problems:
        for x in problems:
            print("finalize PROBLEM:", x, file=sys.stderr)
        return 3
    print(f"finalized budget {N} from {a.checkpoint}: {chosen['final_triangles']} triangles, "
          f"IoU {i_fin:.4f}; wrote {fin / 'CHOSEN.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
