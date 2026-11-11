"""Single evaluation entry point (CONTRACT §9).

    python3 scripts/evaluate.py --mesh variants/100/checkpoints/v0_initial/mesh.json \
        [--views] [--turntable] [--no-render] [--tag NAME] [--derived-dir DIR] [--proxy NPZ]

Steps: schema check -> assemble (.blend/.glb, Blender CPU) -> validate (4-way count +
topology, validation.json) -> render (scripts/render_variant.py; it takes the GPU lock
itself, so this script never does) -> metrics from the Blender silhouette -> overlay ->
metrics.json -> one row appended to <variant dir>/history.json.

--no-render: no Blender render; metrics come from catlib.raster and are marked
"official": false.  They go to metrics.json only if that file does not already hold
official metrics for the same mesh hash (otherwise metrics_unofficial.json); the overlay is
renders/ref_overlay_unofficial.png so official render files are never overwritten.

Exit codes: 0 ok, 2 validation hard failure, 3 render failed, 4 evaluation failed
(cross-check < 0.99, missing reference data), 1 tool error.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the folder that holds catlib/
from catlib import paths as _paths  # noqa: E402  (resolves the case: --case / LOWPOLY_CASE / cwd)
ROOT = _paths.ROOT
sys.path.insert(0, str(_paths.SCRIPTS))

import numpy as np  # noqa: E402

from catlib import meshio, metrics, overlay, paths, provenance, raster  # noqa: E402
from catlib.camera import Camera, load_scene  # noqa: E402
import validate_variant  # noqa: E402

CROSSCHECK_MIN = 0.99
HEADLINE = ("iou", "ref_uncovered_frac", "render_outside_frac", "boundary_mean_px",
            "boundary_hd95_px", "boundary_max_px", "boundary_mean_norm", "legs_gap_iou",
            "region_iou", "landmark_sharp_mean_px", "landmark_sharp_max_px",
            "landmark_sharp_mean_norm", "landmark_fuzzy_mean_px", "raster_vs_blender_iou")


def variant_dir_of(mesh_path: Path) -> Path:
    p = mesh_path.parent
    if p.parent.name == "checkpoints":
        return p.parent.parent
    if p.name == "final":
        return p.parent
    return p


def run_blender(script: Path, args: list[str], log: Path) -> int:
    cmd = [str(paths.BLENDER), "-b", "--factory-startup", "--python", str(script), "--", *args]
    with log.open("w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Assemble, validate, render, measure one mesh.json.")
    ap.add_argument("--mesh", required=True, type=Path)
    ap.add_argument("--views", action="store_true", help="also render the fixed novel views")
    ap.add_argument("--turntable", action="store_true", help="also render turntables")
    ap.add_argument("--no-render", action="store_true", help="unofficial raster-only metrics")
    ap.add_argument("--passes", default="silhouette,clay,wire,beauty")
    ap.add_argument("--tag", default=None, help="history tag (default: checkpoint folder name)")
    ap.add_argument("--derived-dir", type=Path, default=None, help="default reference/derived")
    ap.add_argument("--proxy", type=Path, default=paths.PROXY_NPZ)
    ap.add_argument("--camera", type=Path, default=paths.CAMERA_JSON)
    a = ap.parse_args(argv)

    t0 = time.time()
    timings = {}
    mesh_path = a.mesh.resolve()
    ck = mesh_path.parent
    tag = a.tag or ck.name
    vdir = variant_dir_of(mesh_path)
    renders = ck / "renders"
    logs = ck / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    derived = paths.derived_paths(a.derived_dir)

    # ---------------------------------------------------------------- 1. schema
    raw = json.loads(mesh_path.read_text())
    schema = [e for e in meshio.validate_dict(raw) if not e.startswith("over_budget")]
    if schema:
        print("evaluate: mesh.json schema errors:\n  " + "\n  ".join(schema), file=sys.stderr)
        return 2
    mesh = meshio.CatMesh(raw, mesh_path)
    name = mesh.name

    # ---------------------------------------------------------------- 2. assemble
    t = time.time()
    rc = run_blender(paths.SCRIPTS / "assemble_variant.py",
                     ["--mesh", str(mesh_path), "--out-dir", str(ck)], logs / "assemble.log")
    timings["assemble_s"] = round(time.time() - t, 2)
    if rc != 0:
        print(f"evaluate: assemble failed (rc {rc}), see {logs / 'assemble.log'}", file=sys.stderr)
        return 1
    blend, glb = ck / f"{name}.blend", ck / f"{name}.glb"

    # ---------------------------------------------------------------- 3. validate
    t = time.time()
    val = validate_variant.validate(mesh_path, blend, glb, ck / "validation.json")
    timings["validate_s"] = round(time.time() - t, 2)
    hard = val["hard_failures"]
    c = val["counts"]
    print(f"{name} [{tag}]: A={c.get('A_mesh_json')} B={c.get('B_blend_evaluated')} "
          f"C={c.get('C_glb_reimported')} D={c.get('D_glb_parsed')} budget={mesh.budget}")
    for h in hard:
        print("HARD FAILURE:", h)

    # ---------------------------------------------------------------- 4. render
    render_ok = None
    rendered_mask = None
    render_error = None
    do_render = not a.no_render and not hard
    if not a.no_render and hard:
        render_error = "render skipped: validation hard failures"
    if do_render:
        script = paths.SCRIPTS / "render_variant.py"
        if not script.is_file():
            render_error = f"{script} does not exist"
        else:
            renders.mkdir(parents=True, exist_ok=True)
            sil = renders / "ref_silhouette.png"
            if sil.exists():
                sil.unlink()                      # never score a stale render
            args = ["--blend", str(blend), "--out", str(renders), "--passes", a.passes]
            if a.views:
                args.append("--views")
            if a.turntable:
                args.append("--turntable")
            t = time.time()
            rc = run_blender(script, args, logs / "render.log")
            timings["render_s"] = round(time.time() - t, 2)
            if rc != 0 or not sil.is_file():
                render_error = f"render_variant failed (rc {rc}), see {logs / 'render.log'}"
            else:
                rendered_mask = metrics.load_mask(sil)
        render_ok = render_error is None
        if render_error:
            print("RENDER FAILURE:", render_error, file=sys.stderr)

    # ---------------------------------------------------------------- 5. metrics
    t = time.time()
    cam = Camera.from_json(a.camera)
    met: dict = {}
    eval_errors = []
    if not derived["mask"].is_file():
        eval_errors.append(f"reference mask missing: {derived['mask']}")
        M = raster.silhouette(mesh.vertices, mesh.triangles, cam)
    else:
        met, M = metrics.compute_metrics(mesh, cam, derived, rendered_mask, a.proxy, load_scene())
        if met["official"]:
            x = met["raster_vs_blender_iou"]
            if x is None or x < CROSSCHECK_MIN:
                eval_errors.append(f"raster-vs-Blender silhouette IoU {x} < {CROSSCHECK_MIN}")
    timings["metrics_s"] = round(time.time() - t, 2)
    official = bool(met.get("official"))

    # ---------------------------------------------------------------- 6. overlay
    renders.mkdir(parents=True, exist_ok=True)
    if met:
        R = metrics.load_mask(derived["mask"])
        regions = metrics.load_regions(derived["regions"]) if derived["regions"].is_file() else {}
        ov = renders / ("ref_overlay.png" if official else "ref_overlay_unofficial.png")
        overlay.make_overlay(ov, M, met, regions, R, title=f"{name}  [{tag}]  "
                             f"{c.get('final_triangles')} tris / budget {mesh.budget}")
        met["overlay"] = str(ov)
    if not official:
        metrics.save_mask(renders / "raster_silhouette_unofficial.png", M)

    # ---------------------------------------------------------------- 7. metrics.json
    prov = provenance.provenance(mesh_path, a.derived_dir, a.camera, proxy_path=a.proxy)
    elapsed = round(time.time() - t0, 2)
    doc = {
        "name": name, "tag": tag, "budget": mesh.budget,
        "official": official,
        "camera_status": cam.status,
        "warnings": ([] if cam.status == "locked" else
                     [f"camera status is {cam.status!r}: numbers are not comparable to final tables"])
                    + ([] if official else ["UNOFFICIAL: silhouette from catlib.raster, not a Blender render"]),
        "triangles": {"target_budget": mesh.budget, "mesh_json": c.get("A_mesh_json"),
                      "blend_evaluated": c.get("B_blend_evaluated"),
                      "glb_reimported": c.get("C_glb_reimported"), "glb_parsed": c.get("D_glb_parsed"),
                      "all_agree": c.get("all_agree")},
        "validation": {"valid": val["valid"], "hard_failures": hard,
                       "disclosed": {k: v for k, v in val.get("disclosed", {}).items()
                                     if k not in ("hidden_triangle_ids", "self_intersecting_pair_list")},
                       "region_allocation": val.get("region_allocation")},
        "render": {"requested": not a.no_render, "ok": render_ok, "error": render_error,
                   "passes": a.passes, "views": a.views, "turntable": a.turntable,
                   "dir": str(renders)},
        "evaluation_errors": eval_errors,
        "metrics": met,
        "provenance": prov,
        "timings": timings,
        "elapsed_s": elapsed,
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    mpath = ck / "metrics.json"
    if not official and mpath.is_file():
        try:
            old = json.loads(mpath.read_text())
            if old.get("official") and old.get("provenance", {}).get("mesh_sha256") == prov["mesh_sha256"]:
                mpath = ck / "metrics_unofficial.json"
        except json.JSONDecodeError:
            pass
    mpath.write_text(json.dumps(doc, indent=1, default=float))

    # ---------------------------------------------------------------- 8. history
    hist_path = vdir / "history.json"
    hist = json.loads(hist_path.read_text()) if hist_path.is_file() else []
    row = {"tag": tag, "timestamp": doc["timestamp"], "mesh": str(mesh_path.relative_to(paths.ROOT))
           if mesh_path.is_relative_to(paths.ROOT) else str(mesh_path),
           "official": official, "valid": val["valid"], "triangles": doc["triangles"],
           "headline": {k: met.get(k) for k in HEADLINE} if met else None,
           "hard_failures": hard, "render_error": render_error, "evaluation_errors": eval_errors,
           "elapsed_s": elapsed, "camera_status": cam.status,
           "provenance": {k: v for k, v in prov.items() if k != "tooling_files"}}
    hist.append(row)
    hist_path.write_text(json.dumps(hist, indent=1, default=float))

    # ---------------------------------------------------------------- summary
    if met:
        print(f"{'OFFICIAL' if official else 'UNOFFICIAL'} metrics: IoU {met['iou']:.4f}  "
              f"boundary mean {met['boundary_mean_px']:.2f}px HD95 {met['boundary_hd95_px']:.2f}px  "
              f"landmarks sharp {met['landmark_sharp_mean_px']}px  legs_gap_iou {met['legs_gap_iou']}"
              + (f"  raster_vs_blender {met['raster_vs_blender_iou']:.4f}" if official else ""))
    for e in eval_errors:
        print("EVALUATION FAILURE:", e)
    print(f"wrote {mpath} and {ck / 'validation.json'}; history row -> {hist_path} ({elapsed}s)")
    if hard:
        return 2
    if render_error and not a.no_render:
        return 3
    if eval_errors:
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
