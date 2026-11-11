"""Shared fixed-connectivity vertex refinement CLI (CONTRACT §12).  CPU only.

    python3 scripts/refine_vertices.py --mesh in.json --out out.json \
        [--iters N | --minutes M] [--res 384,768] [--nv-res 160] \
        [--w-iou .. --w-bnd .. --w-lm .. --w-sdf .. --w-nv .. --w-reg .. --w-sym ..] \
        [--flips] [--freeze-verts FILE|i,j,k] [--symmetry-pairs FILE] [--seed 0] \
        [--log trace.json] [--derived-dir DIR] [--camera JSON] [--scene JSON] [--proxy-dir DIR]
        [--snapshot-dir DIR --snapshot-every K]   (CONTRACT §13 film snapshots; off by default)

The internal score is a SEARCH HEURISTIC ONLY.  The before/after numbers printed and
logged are UNOFFICIAL (catlib.metrics through catlib.raster at 1536 px).  Only
scripts/evaluate.py (Blender render, locked camera) produces official numbers, and those
decide which checkpoint is kept.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the folder that holds catlib/
from catlib import paths as _paths  # noqa: E402  (resolves the case: --case / LOWPOLY_CASE / cwd)
ROOT = _paths.ROOT

import numpy as np  # noqa: E402

from catlib import geom, meshio, paths  # noqa: E402
from catlib.refine import DEFAULT_WEIGHTS, Refiner, Settings, Sources, unofficial_metrics  # noqa: E402


def _indices(arg):
    if not arg:
        return ()
    p = Path(arg)
    if p.is_file():
        txt = p.read_text().strip()
        try:
            v = json.loads(txt)
            if isinstance(v, dict):
                v = v.get("vertices", v.get("freeze", []))
            return tuple(int(x) for x in v)
        except json.JSONDecodeError:
            return tuple(int(x) for x in txt.replace(",", " ").split())
    return tuple(int(x) for x in arg.split(",") if x.strip())


def _pairs(arg):
    if not arg:
        return ()
    v = json.loads(Path(arg).read_text())
    if isinstance(v, dict):
        v = v.get("pairs", [])
    return tuple((int(a), int(b)) for a, b in v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Fixed-connectivity vertex refinement. Its objective is a search heuristic "
                    "only; official scripts/evaluate.py numbers decide what is kept. Never changes "
                    "the triangle count.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mesh", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--iters", type=int, default=None, help="total proposals (deterministic); default 20000")
    g.add_argument("--minutes", type=float, default=None, help="wall-clock budget (not deterministic)")
    ap.add_argument("--res", default="384,768", help="coarse-to-fine working resolutions (divisors of 1536 preferred)")
    ap.add_argument("--nv-res", type=int, default=160, help="novel-view coverage resolution")
    for k, v in DEFAULT_WEIGHTS.items():
        ap.add_argument(f"--w-{k}", type=float, default=v)
    ap.add_argument("--flips", action="store_true", help="also try same-region edge flips")
    ap.add_argument("--freeze-verts", default=None, help="file (JSON list or whitespace ints) or i,j,k")
    ap.add_argument("--symmetry-pairs", default=None, help="JSON [[i,j],...]: v_i ~ mirror_y(v_j)")
    ap.add_argument("--group-frac", type=float, default=0.1, help="share of stochastic proposals that are rigid group moves")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", type=Path, default=None, help="trace JSON (default: <out>.trace.json)")
    ap.add_argument("--derived-dir", type=Path, default=paths.DERIVED)
    ap.add_argument("--camera", type=Path, default=paths.CAMERA_JSON)
    ap.add_argument("--scene", type=Path, default=paths.SCENE_JSON)
    ap.add_argument("--proxy-dir", type=Path, default=paths.ROOT / "proxy")
    ap.add_argument("--no-metrics", action="store_true", help="skip the unofficial before/after metrics")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--snapshot-dir", type=Path, default=None,
                    help="CONTRACT §13: write snap_<proposal>.json here (vertices + objective); off by default")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="snapshot every K proposals (accepted or rejected); needs --snapshot-dir")
    a = ap.parse_args(argv)
    if bool(a.snapshot_dir) != (a.snapshot_every > 0):
        ap.error("--snapshot-dir and --snapshot-every K (K > 0) go together")

    d = json.loads(a.mesh.read_text())
    errs = [e for e in meshio.validate_dict(d) if not e.startswith("over_budget")]
    if errs:
        print("refine_vertices: invalid input mesh:\n  " + "\n  ".join(errs), file=sys.stderr)
        return 2
    src = Sources(a.derived_dir, a.camera, a.scene, a.proxy_dir)
    weights = {k: getattr(a, f"w_{k}") for k in DEFAULT_WEIGHTS}
    iters = a.iters if a.iters is not None else (None if a.minutes else 20000)
    st = Settings(weights=weights, res=tuple(int(x) for x in a.res.split(",")), nv_res=a.nv_res,
                  iters=iters, minutes=a.minutes, flips=a.flips, seed=a.seed,
                  freeze=_indices(a.freeze_verts), sym_pairs=_pairs(a.symmetry_pairs),
                  group_frac=a.group_frac, verbose=not a.quiet)
    if a.snapshot_every:
        st.snapshot_dir, st.snapshot_every = str(a.snapshot_dir), int(a.snapshot_every)
    t0 = time.time()
    before = None if a.no_metrics else unofficial_metrics(d, src)
    t_m = time.time() - t0
    ref = Refiner(d, src, st)
    for n in ref.notes:
        print("note:", n)
    t1 = time.time()
    out = ref.run()
    t_opt = time.time() - t1
    if len(out["triangles"]) != len(d["triangles"]):
        print("BUG: triangle count changed", file=sys.stderr)
        return 1
    topo = geom.topology_report(np.asarray(out["vertices"]), np.asarray(out["triangles"]))
    stamp = (f"[refine_vertices seed={a.seed} proposals={ref.proposals} res={a.res} "
             f"weights={json.dumps(weights, sort_keys=True)} flips={a.flips} {t_opt:.1f}s]")
    out["notes"] = (out.get("notes", "") + " " + stamp).strip()
    meshio.save(a.out, out)
    after = None if a.no_metrics else unofficial_metrics(out, src)
    log = {
        "tool": "scripts/refine_vertices.py", "disclaimer": "objective is a search heuristic; metrics "
        "here are UNOFFICIAL (catlib.raster); official numbers come from scripts/evaluate.py",
        "input": str(a.mesh), "output": str(a.out), "triangles": len(out["triangles"]),
        "settings": {"weights": weights, "res": st.res, "nv_res": st.nv_res, "iters": st.iters,
                     "minutes": st.minutes, "flips": st.flips, "seed": st.seed,
                     "frozen": len(st.freeze), "sym_pairs": len(st.sym_pairs), "group_frac": st.group_frac},
        "sources": {k: str(getattr(src, k)) for k in ("mask", "landmarks", "bbox", "camera", "scene",
                                                     "proxy_mesh", "proxy_sdf")},
        "notes": ref.notes, "proposals": ref.proposals, "optimise_s": round(t_opt, 2),
        "proposals_per_s": round(ref.proposals / max(t_opt, 1e-9), 1),
        "counters": ref.counters, "pinned_vertices": int(ref.pinned.sum()),
        "groups": {k: len(v) for k, v in ref.groups.items()},
        "after_constraints": {"min_triangle_area_m2": topo["min_triangle_area_m2"],
                              "min_edge_m": topo["min_edge_m"], "min_z": topo["min_z"],
                              "degenerate_triangles": len(topo["degenerate_triangles"]),
                              "closed_manifold": topo["closed_manifold"]},
        "unofficial_before": before, "unofficial_after": after, "metrics_s": round(t_m, 2),
        "trace": ref.trace,
    }
    if a.snapshot_every:
        log["snapshots"] = {"dir": str(a.snapshot_dir), "every": a.snapshot_every}
    lp = a.log or a.out.with_suffix(".trace.json")
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(json.dumps(log, indent=1, default=float))
    print(f"wrote {a.out} ({len(out['triangles'])} triangles) and {lp}; {ref.proposals} proposals "
          f"in {t_opt:.1f}s ({log['proposals_per_s']}/s)")
    if before and after:
        for k in ("iou", "boundary_mean_px", "landmark_sharp_mean_px", "novel_view_iou_vs_proxy",
                  "chamfer_mm_vs_proxy"):
            b, c = before.get(k), after.get(k)
            if b is not None and c is not None:
                print(f"  UNOFFICIAL {k:26s} {b:9.4f} -> {c:9.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
