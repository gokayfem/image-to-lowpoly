"""Aggregate the four finals into outputs/metrics.json and outputs/metrics.csv.

    python3 scripts/aggregate_metrics.py [--variants-root variants] [--out outputs] \
        [--baselines-root baselines] [--proxy-report proxy/fit_report.json] [--allow-provisional]

Per budget: target, actual triangles four ways (mesh.json / evaluated .blend / reimported GLB /
parsed GLB), unused budget, initial (first official evaluation = checkpoints/v0_initial) and
final official metrics, deltas (final - initial), every disclosed topology finding, triangle
allocation by region, evaluated checkpoints, summed evaluation seconds, optimiser
iterations/seconds when a checkpoint left a refine report, the chosen checkpoint and reason
(final/CHOSEN.json), baseline rows (baselines/decimate*/<N>) and the proxy's own scores.

FAILS (exit 2, every problem listed) when: a final is missing; a final is not official; the
camera is not locked (unless --allow-provisional); the finals do not share identical
reference / mask / landmarks / regions / camera / scene / tooling hashes; a final's
metrics.json is stale for its mesh.json; any of the four counts disagree, is missing or
exceeds the budget; validation has hard failures.  Nothing is written on failure.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the folder that holds catlib/
from catlib import paths as _paths  # noqa: E402  (resolves the case: --case / LOWPOLY_CASE / cwd)
ROOT = _paths.ROOT
sys.path.insert(0, str(_paths.SCRIPTS))

from catlib import paths, provenance  # noqa: E402
import integ_common as ic  # noqa: E402
from integ_common import BUDGETS, headline, read_json  # noqa: E402

HASH_KEYS = ("reference_sha256", "mask_sha256", "landmarks_sha256", "regions_sha256", "camera_sha256",
             "scene_sha256", "tooling_hash")
COUNT_KEYS = ("A_mesh_json", "B_blend_evaluated", "C_glb_reimported", "D_glb_parsed")
DROP_DISCLOSED = {"hidden_triangle_ids", "self_intersecting_pair_list", "landmark_surface_dist_mm",
                  "component_signed_volume_m3", "bbox_min", "bbox_max"}


def optimiser_info(vdir: Path) -> dict:
    """Optimiser effort from scripts/refine_vertices.py trace logs found anywhere under the
    variant dir (default name <out>.trace.json, i.e. checkpoints/<tag>/mesh.trace.json):
    every JSON whose "tool" is scripts/refine_vertices.py contributes its "proposals"
    (vertex-move proposals = optimiser iterations) and "optimise_s"."""
    props, secs, files = 0, 0.0, []
    for p in sorted(vdir.rglob("*.json")):
        if "final" in p.relative_to(vdir).parts or p.stat().st_size > 200_000_000:
            continue
        if p.name in ("mesh.json", "metrics.json", "validation.json", "history.json", "CHOSEN.json"):
            continue
        try:
            d = read_json(p)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        if isinstance(d, dict) and str(d.get("tool", "")).endswith("refine_vertices.py"):
            props += int(d.get("proposals") or 0)
            secs += float(d.get("optimise_s") or 0.0)
            files.append(str(p.relative_to(vdir)))
    return {"refine_logs": files, "refine_runs": len(files),
            "optimizer_iterations": props if files else None,
            "optimizer_elapsed_s": round(secs, 2) if files else None,
            "note": "sum of 'proposals' and 'optimise_s' over refine_vertices.py trace logs under the variant dir; "
                    "build-script time is not included"}


def history_info(vdir: Path) -> dict:
    hp = vdir / "history.json"
    rows = read_json(hp) if hp.is_file() else []
    ev = [r for r in rows if r.get("tag") != "final"]
    return {"evaluations": len(ev), "official_evaluations": sum(1 for r in ev if r.get("official")),
            "distinct_checkpoints": len({r.get("tag") for r in ev}),
            "checkpoint_tags": sorted({r.get("tag") for r in ev}),
            "evaluation_elapsed_s_sum": round(sum(float(r.get("elapsed_s") or 0) for r in ev), 2),
            "final_evaluations": len(rows) - len(ev)}


def deltas(h1: dict, h0: dict) -> dict:
    out = {}
    for k, v in h1.items():
        b = h0.get(k)
        out[k] = (v - b) if isinstance(v, (int, float)) and isinstance(b, (int, float)) else None
    return out


def check_final(N: int, fin: Path, allow_provisional: bool, errors: list[str]) -> tuple[dict, dict] | None:
    mp, vp = fin / "metrics.json", fin / "validation.json"
    if not mp.is_file() or not vp.is_file():
        errors.append(f"budget {N}: {fin} lacks metrics.json / validation.json (run finalize_variant.py)")
        return None
    m, v = read_json(mp), read_json(vp)
    if not m.get("official"):
        errors.append(f"budget {N}: final metrics are not official (silhouette not from a Blender render)")
    if m.get("evaluation_errors"):
        errors.append(f"budget {N}: evaluation errors {m['evaluation_errors']}")
    if m.get("camera_status") != "locked" and not allow_provisional:
        errors.append(f"budget {N}: camera status is {m.get('camera_status')!r}, not locked "
                      "(pass --allow-provisional only for tooling tests)")
    mj = fin / "mesh.json"
    if mj.is_file() and provenance.sha256_file(mj) != m.get("provenance", {}).get("mesh_sha256"):
        errors.append(f"budget {N}: final metrics.json is stale (mesh hash differs from final/mesh.json)")
    c = v.get("counts", {})
    vals = [c.get(k) for k in COUNT_KEYS]
    if any(x is None for x in vals):
        errors.append(f"budget {N}: missing triangle count(s) {dict(zip(COUNT_KEYS, vals))}")
    elif len(set(vals)) != 1 or not c.get("all_agree"):
        errors.append(f"budget {N}: triangle counts disagree {dict(zip(COUNT_KEYS, vals))}")
    for k, x in zip(COUNT_KEYS, vals):
        if x is not None and x > N:
            errors.append(f"budget {N}: {k} = {x} exceeds budget {N}")
    tm = m.get("triangles", {})
    if tm.get("mesh_json") is not None and tm.get("mesh_json") != c.get("A_mesh_json"):
        errors.append(f"budget {N}: metrics.json triangles {tm} disagree with validation.json counts")
    if v.get("hard_failures"):
        errors.append(f"budget {N}: hard failures {v['hard_failures']}")
    if m.get("budget") != N:
        errors.append(f"budget {N}: mesh budget field is {m.get('budget')}")
    return m, v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate final metrics (fails loudly on any inconsistency).")
    ap.add_argument("--variants-root", type=Path, default=paths.VARIANTS)
    ap.add_argument("--out", type=Path, default=paths.OUTPUTS)
    ap.add_argument("--baselines-root", type=Path, default=paths.BASELINES)
    ap.add_argument("--proxy-report", type=Path, default=ROOT / "proxy" / "fit_report.json")
    ap.add_argument("--allow-provisional", action="store_true", help="tooling tests only: accept an unlocked camera")
    a = ap.parse_args(argv)

    errors: list[str] = []
    warnings: list[str] = []
    finals = {}
    for N in BUDGETS:
        r = check_final(N, a.variants_root / str(N) / "final", a.allow_provisional, errors)
        if r:
            finals[N] = r
    # identical provenance across the four finals
    for k in HASH_KEYS:
        vals = {N: (m.get("provenance") or {}).get(k) for N, (m, _) in finals.items()}
        if len(set(vals.values())) > 1 or None in vals.values():
            errors.append(f"provenance {k} differs between finals (or is missing): "
                          + ", ".join(f"{N}={str(x)[:12]}" for N, x in vals.items()))
    if errors:
        print("aggregate_metrics FAILED; nothing written:", file=sys.stderr)
        for e in errors:
            print("  -", e, file=sys.stderr)
        return 2

    prov0 = finals[BUDGETS[0]][0]["provenance"]
    variants = []
    for N in BUDGETS:
        m, v = finals[N]
        vdir = a.variants_root / str(N)
        init, init_tag = ic.initial_entry(vdir)
        if init is None:
            warnings.append(f"budget {N}: no official initial evaluation found")
        elif any((init.get("provenance") or {}).get(k) != prov0.get(k) for k in HASH_KEYS):
            warnings.append(f"budget {N}: initial checkpoint {init_tag} was scored with different hashes "
                            "(camera/tooling/annotations changed since); its numbers are not comparable — "
                            "re-evaluate it")
        h0, h1 = headline(init), headline(m)
        chosen = read_json(vdir / "final" / "CHOSEN.json") if (vdir / "final" / "CHOSEN.json").is_file() else {}
        c = v["counts"]
        bl = {}
        for kind, bm in ic.baselines_for(a.baselines_root, N).items():
            comparable = all((bm.get("provenance") or {}).get(k) == prov0.get(k) for k in HASH_KEYS)
            if not comparable:
                warnings.append(f"budget {N}: baseline {kind} scored with different hashes; marked not comparable")
            bl[kind] = {"path": bm.get("path"), "official": bm.get("official"), "comparable": comparable,
                        "triangles": bm.get("triangles"), "metrics": headline(bm),
                        "camera_status": bm.get("camera_status")}
        variants.append({
            "budget": N,
            "triangles": {"target": N, "mesh_json": c["A_mesh_json"], "blend_evaluated": c["B_blend_evaluated"],
                          "glb_reimported": c["C_glb_reimported"], "glb_parsed": c["D_glb_parsed"],
                          "all_agree": bool(c["all_agree"]), "final": c["final_triangles"],
                          "unused_budget": N - c["final_triangles"], "exact": c["final_triangles"] == N},
            "chosen_checkpoint": chosen.get("checkpoint"), "chosen_reason": chosen.get("reason"),
            "initial": {"tag": init_tag, "metrics": h0,
                        "triangles": (init or {}).get("triangles", {}).get("mesh_json"),
                        "camera_status": (init or {}).get("camera_status")},
            "final": {"metrics": h1, "mesh_sha256": m["provenance"]["mesh_sha256"],
                      "camera_status": m.get("camera_status"),
                      "landmark_flags": (m.get("metrics") or {}).get("landmark_flags")},
            "delta_final_minus_initial": deltas(h1, h0),
            "topology": {"hard_failures": v.get("hard_failures", []),
                         "disclosed": {k: x for k, x in (v.get("disclosed") or {}).items() if k not in DROP_DISCLOSED}},
            "region_allocation": v.get("region_allocation") or (m.get("validation") or {}).get("region_allocation"),
            "history": history_info(vdir),
            "optimizer": optimiser_info(vdir),
            "baselines": bl,
            "files": {"blend": str(vdir / "final" / f"cat_{N}.blend"), "glb": str(vdir / "final" / f"cat_{N}.glb"),
                      "renders": str(vdir / "final" / "renders")},
        })
    proxy = None
    if a.proxy_report and a.proxy_report.is_file():
        proxy = read_json(a.proxy_report)
    else:
        warnings.append(f"proxy fit report {a.proxy_report} not found")
    doc = {
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "camera_status": finals[BUDGETS[0]][0].get("camera_status"),
        "provenance": {k: prov0.get(k) for k in HASH_KEYS},
        "provenance_identical_across_finals": True,
        "allow_provisional": a.allow_provisional,
        "warnings": warnings,
        "variants": variants,
        "proxy_fit_report": proxy,
        "notes": ["Official metrics: Blender silhouette render through the shared camera (CONTRACT §7).",
                  "novel_view_iou_vs_proxy and chamfer_mm_vs_proxy compare against inferred geometry, not ground truth.",
                  "delta = final - initial; for IoU higher is better, for *_px / chamfer lower is better."],
    }
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(doc, indent=1, default=float))

    # CSV: one row per budget x stage
    mkeys = list(headline(None).keys())
    regions = ("head", "ears", "eyes", "nose", "neck", "torso", "leg_FR", "leg_FL", "leg_HR", "leg_HL", "tail")
    topo_keys = ("components", "boundary_edges", "non_manifold_edges", "inconsistent_winding_edges",
                 "self_intersecting_pairs", "hidden_triangles", "flat_redundant_vertices", "closed_manifold")
    header = (["budget", "stage", "tag", "camera_status", "tris_final", "tris_mesh_json", "tris_blend_evaluated",
               "tris_glb_reimported", "tris_glb_parsed", "counts_agree", "unused_budget"] + mkeys
              + [f"topo_{k}" for k in topo_keys] + [f"alloc_{r}" for r in regions]
              + ["evaluations", "evaluation_elapsed_s_sum", "optimizer_iterations", "optimizer_elapsed_s",
                 "chosen_reason"])
    rows = []
    for vr in variants:
        t = vr["triangles"]
        alloc = vr["region_allocation"] or {}
        base = [vr["budget"]]
        common_tail = ([vr["topology"]["disclosed"].get(k) for k in topo_keys]
                       + [(alloc.get(r) or {}).get("triangles") for r in regions]
                       + [vr["history"]["evaluations"], vr["history"]["evaluation_elapsed_s_sum"],
                          vr["optimizer"]["optimizer_iterations"], vr["optimizer"]["optimizer_elapsed_s"]])
        rows.append(base + ["initial", vr["initial"]["tag"], vr["initial"]["camera_status"], vr["initial"]["triangles"],
                            "", "", "", "", "", ""] + [vr["initial"]["metrics"][k] for k in mkeys]
                    + [""] * (len(topo_keys) + len(regions) + 4) + [""])
        rows.append(base + ["final", vr["chosen_checkpoint"], vr["final"]["camera_status"], t["final"], t["mesh_json"],
                            t["blend_evaluated"], t["glb_reimported"], t["glb_parsed"], t["all_agree"],
                            t["unused_budget"]] + [vr["final"]["metrics"][k] for k in mkeys] + common_tail
                    + [vr["chosen_reason"]])
        rows.append(base + ["delta_final_minus_initial", "", "", "", "", "", "", "", "", ""]
                    + [vr["delta_final_minus_initial"][k] for k in mkeys] + [""] * (len(topo_keys) + len(regions) + 5))
        for kind, b in vr["baselines"].items():
            bt = b.get("triangles") or {}
            rows.append(base + [f"baseline:{kind}" + ("" if b["comparable"] else " (NOT comparable)"), "",
                                b.get("camera_status"), bt.get("mesh_json"), bt.get("mesh_json"),
                                bt.get("blend_evaluated"), bt.get("glb_reimported"), bt.get("glb_parsed"),
                                bt.get("all_agree"), ""] + [b["metrics"][k] for k in mkeys]
                        + [""] * (len(topo_keys) + len(regions) + 5))
    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if x is None else (f"{x:.6g}" if isinstance(x, float) else x) for x in r])
    for wmsg in warnings:
        print("WARNING:", wmsg)
    for vr in variants:
        t, f, i = vr["triangles"], vr["final"]["metrics"], vr["initial"]["metrics"]
        print(f"budget {vr['budget']:>4}: {t['final']} tris (4-way agree, unused {t['unused_budget']}); "
              f"IoU {ic.fmt(i['iou'], 4)} -> {ic.fmt(f['iou'], 4)}; boundary mean "
              f"{ic.fmt(i['boundary_mean_px'], 1)} -> {ic.fmt(f['boundary_mean_px'], 1)} px; sharp LM "
              f"{ic.fmt(i['landmark_sharp_mean_px'], 1)} -> {ic.fmt(f['landmark_sharp_mean_px'], 1)} px")
    print(f"wrote {out / 'metrics.json'} and {out / 'metrics.csv'} (camera {doc['camera_status']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
