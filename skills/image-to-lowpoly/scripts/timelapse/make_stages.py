#!/usr/bin/env python
"""Fallback builder of a build-timelapse stages.json (CONTRACT §13) for a budget whose worker has not written
variants/<N>/timelapse/stages.json. A worker-written stages.json always takes precedence (timelapse.py checks it
first); this tool never writes under variants/.

    python3 skills/image-to-lowpoly/scripts/timelapse/make_stages.py --budget 200 [--checkpoint v3_tailbase] [--every K] [--force]

Output: release_video/work/<N>/stages.json (+ construct/, refine_<k>/snaps/, edit_<k>/, reproduce.json).

How the history of the chosen checkpoint is recovered (all real files, nothing invented):
  1. refine runs: every scripts/refine_vertices.py trace JSON under variants/<N>/ (by stated output path and by
     the mesh.json next to it) and every documented `refine_vertices.py` command in variants/<N>/NOTES.md and
     *.sh (by --out). The command gives the exact --freeze-verts / --symmetry-pairs files; otherwise a
     freeze file with the trace's recorded count is searched next to the input / checkpoint.
  2. walk back from the checkpoint: mesh <- refine(input) <- ... ; an input that is not a refine output but has
     exactly the vertex positions of another mesh under variants/<N>/ with different triangles is an `edit`
     (e.g. hand edge flips); the first mesh with neither is the build output.
  3. construct stages reveal the build output's own triangles part by part (tri_region groups, build order).
  4. every refine run is RE-RUN with the recorded seed/iters/weights/flags plus --snapshot-dir/--snapshot-every
     into release_video/work/<N>/refine_<k>/; the last snapshot is compared with the recorded output and the
     max vertex deviation is written to reproduce.json. A run that does not reproduce (or cannot be re-run, e.g.
     a --minutes run) keeps its real endpoints only, and the sequence ends on the true checkpoint mesh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
ROOT = C.ROOT
_paths = C._paths

PY = str(_paths.VENV_PYTHON)
REFINE = _paths.SCRIPTS / "refine_vertices.py"
REPRO_TOL = 1e-6      # metres; meshio.save rounds coordinates, so exact equality is not expected

# construction groups in build order: (label, regions)
GROUPS = [("torso", ["torso"]), ("neck + head", ["neck", "head"]), ("ears, eyes, nose", ["ears", "eyes", "nose"]),
          ("front legs", ["leg_FR", "leg_FL"]), ("hind legs", ["leg_HR", "leg_HL"]), ("tail", ["tail"])]


def R(p) -> Path:
    p = Path(p)
    return (p if p.is_absolute() else ROOT / p).resolve()


def vhash(d) -> str:
    return hashlib.sha256(np.round(np.asarray(d["vertices"], float), 9).tobytes()).hexdigest()


def thash(d) -> str:
    return hashlib.sha256(np.asarray(d["triangles"], np.int64).tobytes()).hexdigest()


# ------------------------------------------------------------------ discover refine runs
def documented_commands(vdir: Path) -> dict:
    """--out -> argv list, from NOTES.md / *.sh / *.md in the variant dir."""
    out = {}
    for f in sorted(list(vdir.glob("*.md")) + list(vdir.glob("*.sh")) + list(vdir.glob("**/*.sh"))):
        txt = f.read_text(errors="replace")
        env = {"V": str(vdir.relative_to(ROOT)), "PY": "python3"}
        for m in re.finditer(r"^\s*([A-Z_][A-Z0-9_]*)=([^\s;#]+)", txt, re.M):
            env[m.group(1)] = m.group(2).strip("'\"")
        txt = re.sub(r"\\\n", " ", txt)
        for line in txt.splitlines():
            if "refine_vertices.py" not in line:
                continue
            line = line.strip().strip("`")
            line = re.sub(r"\$\{?([A-Z_][A-Z0-9_]*)\}?", lambda m: env.get(m.group(1), m.group(0)), line)
            try:
                toks = shlex.split(line, comments=True)
            except ValueError:
                continue
            k = next((i for i, t in enumerate(toks) if t.endswith("refine_vertices.py")), None)
            if k is None:
                continue
            argv = toks[k + 1:]
            if "--out" in argv and "--mesh" in argv:
                o = R(argv[argv.index("--out") + 1])
                out.setdefault(o, {"argv": argv, "source": C.rel(f)})
    return out


def traces(vdir: Path) -> dict:
    out = {}
    for p in sorted(vdir.rglob("*.json")):
        if p.name in ("mesh.json", "metrics.json", "validation.json", "history.json") or p.stat().st_size > 3e8:
            continue
        if "timelapse" in p.parts:
            continue
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        if not (isinstance(d, dict) and str(d.get("tool", "")).endswith("refine_vertices.py")):
            continue
        rec = {"trace": p, "d": d}
        out.setdefault(R(d["output"]), rec)
        sib = p.parent / "mesh.json"
        if p.name == "mesh.trace.json" and sib.is_file():
            out.setdefault(sib.resolve(), rec)
    return out


def freeze_candidates(*dirs):
    seen = []
    for d in dirs:
        for pat in ("*freeze*.json", "rims.json", "freeze_verts.json"):
            for p in sorted(Path(d).glob(pat)):
                if p not in seen:
                    seen.append(p)
    return seen


def count_indices(p: Path) -> int:
    try:
        v = json.loads(p.read_text())
        if isinstance(v, dict):
            v = v.get("vertices", v.get("freeze", []))
        return len(v)
    except json.JSONDecodeError:
        return len(p.read_text().replace(",", " ").split())


def build_argv(run: dict, mesh_in: Path, mesh_out_recorded: Path) -> tuple[list | None, str]:
    """Exact argv for a re-run (without --out/--log/snapshot flags) or (None, why)."""
    doc, tr = run.get("doc"), run.get("tr")
    s = (tr or {}).get("d", {}).get("settings", {})
    if doc:
        a = list(doc["argv"])
        drop = {"--out", "--log", "--snapshot-dir", "--snapshot-every"}
        argv, i = [], 0
        while i < len(a):
            if a[i] in drop:
                i += 2
                continue
            if a[i] in ("--quiet", "--no-metrics"):
                i += 1
                continue
            argv.append(a[i])
            i += 1
        if "--minutes" in argv:
            return None, "--minutes run (wall-clock stop) is not deterministic"
        if "--iters" not in argv:
            argv += ["--iters", "20000"]
        if tr:     # cross-check against the trace
            it = int(argv[argv.index("--iters") + 1])
            if s.get("iters") not in (None, it):
                return None, f"documented --iters {it} disagrees with trace {s.get('iters')}"
        return argv, f"documented command ({doc['source']})"
    if not tr:
        return None, "no trace and no documented command"
    if s.get("minutes") or not s.get("iters"):
        return None, "--minutes run (wall-clock stop) is not deterministic"
    argv = ["--mesh", str(mesh_in), "--iters", str(s["iters"]), "--seed", str(s.get("seed", 0)),
            "--res", ",".join(str(x) for x in s.get("res", [384, 768])), "--nv-res", str(s.get("nv_res", 160)),
            "--group-frac", str(s.get("group_frac", 0.1))]
    for k, v in (s.get("weights") or {}).items():
        argv += [f"--w-{k}", repr(float(v))]
    if s.get("flips"):
        argv.append("--flips")
    nf = int(s.get("frozen") or 0)
    if nf:
        cands = [p for p in freeze_candidates(mesh_in.parent, mesh_out_recorded.parent,
                                              tr["trace"].parent) if count_indices(p) == nf]
        if not cands:
            return None, f"trace froze {nf} vertices but no freeze file with that count was found"
        argv += ["--freeze-verts", str(cands[0])]
    if int(s.get("sym_pairs") or 0):
        return None, "trace used symmetry pairs; the pairs file is not recorded"
    return argv, f"trace {C.rel(tr['trace'])} (freeze file located by count)" if nf else f"trace {C.rel(tr['trace'])}"


def mesh_index(vdir: Path) -> dict:
    """vertex-hash -> [(path, tri-hash)] of every mesh-like JSON under the variant dir."""
    idx = {}
    for p in sorted(vdir.rglob("*.json")):
        if p.stat().st_size > 5e7 or "timelapse" in p.parts:
            continue
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        if isinstance(d, dict) and "vertices" in d and "triangles" in d and "tool" not in d:
            idx.setdefault(vhash(d), []).append((p.resolve(), thash(d)))
    return idx


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--checkpoint", default=None, help="default: final, else BEST.json, else highest official IoU")
    ap.add_argument("--every", type=int, default=0, help="snapshot every K proposals (default iters/120)")
    ap.add_argument("--force", action="store_true", help="re-run refines even if cached")
    a = ap.parse_args(argv)
    N = a.budget
    t0 = time.time()
    vdir = ROOT / "variants" / str(N)
    sel = C.select_checkpoint(N, a.checkpoint)
    final_path = Path(sel["mesh"]).resolve()
    out = C.WORK / str(N)
    out.mkdir(parents=True, exist_ok=True)

    docs, trs = documented_commands(vdir), traces(vdir)
    midx = mesh_index(vdir)
    chain = []          # walking back: dicts {"kind": refine|edit, "in": path, "out": path, ...}
    cur = final_path
    if sel["status"] == "final":      # final/mesh.json is a copy: find the checkpoint it came from
        fd = json.loads(final_path.read_text())
        for p, th in midx.get(vhash(fd), []):
            if th == thash(fd) and "checkpoints" in p.parts:
                cur = p
                break
    seen = {cur}
    start = cur
    for _ in range(10):
        doc, tr = docs.get(cur), trs.get(cur)
        if doc or tr:
            src = R(doc["argv"][doc["argv"].index("--mesh") + 1]) if doc else R(tr["d"]["input"])
            if src in seen:
                break
            seen.add(src)
            chain.append({"kind": "refine", "in": src, "out": cur, "doc": doc, "tr": tr})
            cur = src
            continue
        d = json.loads(cur.read_text())
        same = [q for q, th in midx.get(vhash(d), []) if th == thash(d) and q not in seen and (q in docs or q in trs)]
        if same:        # an identical copy of a refine output (e.g. refined2.json copied to mesh.json)
            cur = same[0]
            seen.add(cur)
            continue
        par = [(p, th) for p, th in midx.get(vhash(d), []) if th != thash(d) and p not in seen
               and (p in docs or p in trs)]
        if not par:     # same triangles, a few vertices moved by hand (e.g. an untangle) after a refine output
            V0 = np.asarray(d["vertices"], float)
            best = None
            for q in set(docs) | set(trs):
                if q in seen or not q.is_file() or not (cur == start or q.parent == cur.parent):
                    continue
                e = json.loads(q.read_text())
                if e.get("triangles") != d["triangles"] or len(e["vertices"]) != len(V0):
                    continue
                dev = float(np.abs(np.asarray(e["vertices"], float) - V0).max())
                if dev > 0 and (best is None or dev < best[1]):
                    best = (q, dev)
            if best:
                par = [(best[0], None)]
        if par:
            chain.append({"kind": "edit", "in": par[0][0], "out": cur})
            cur = par[0][0]
            seen.add(cur)
            continue
        break
    chain.reverse()
    build_mesh = cur
    print(f"budget {N}: checkpoint {sel['tag']} ({sel['status']}); build output {C.rel(build_mesh)}; "
          f"{sum(c['kind'] == 'refine' for c in chain)} refine run(s), {sum(c['kind'] == 'edit' for c in chain)} edit(s)")

    stages, repro = [], {"budget": N, "checkpoint": sel["tag"], "checkpoint_status": sel["status"],
                         "build_output": C.rel(build_mesh), "runs": []}
    # construct: reveal the build output's own triangles, cumulative, by region group
    bd = json.loads(build_mesh.read_text())
    regs = bd.get("tri_region") or ["torso"] * len(bd["triangles"])
    present = set(regs)
    order = [g for g in GROUPS if present & set(g[1])]
    leftovers = sorted(present - {r for _, rr in GROUPS for r in rr})
    if leftovers:
        order.append(("other", leftovers))
    (out / "construct").mkdir(exist_ok=True)
    shown = []
    for i, (label, rr) in enumerate(order, 1):
        if label not in ("front legs", "hind legs"):
            label = ", ".join(r.replace("_", " ") for r in rr if r in present)
            label = label.replace("neck, head", "neck + head")
        shown += [k for k, r in enumerate(regs) if r in rr]
        sub = sorted(shown)
        p = out / "construct" / f"{i:02d}_{label.replace(' ', '_').replace(',', '')}.json"
        p.write_text(json.dumps({"source": C.rel(build_mesh), "vertices": bd["vertices"],
                                 "triangles": [bd["triangles"][k] for k in sub],
                                 "tri_region": [regs[k] for k in sub], "parent_triangle_ids": sub}))
        stages.append({"label": label, "kind": "construct", "mesh": str(p.relative_to(out))})

    ok_all = True
    for k, step in enumerate(chain, 1):
        if step["kind"] == "edit":
            stages.append({"label": "edit: topology change", "kind": "edit", "before": str(step["in"]),
                           "mesh": str(step["out"])})
            repro["runs"].append({"kind": "edit", "before": C.rel(step["in"]), "after": C.rel(step["out"])})
            continue
        argv, how = build_argv(step, step["in"], step["out"])
        rd = out / f"refine_{k}"
        rec = {"kind": "refine", "input": C.rel(step["in"]), "recorded_output": C.rel(step["out"]), "how": how}
        if argv is None:
            ok_all = False
            rec.update({"reproduced": False, "why": how})
            stages.append({"label": "refine (not re-runnable: endpoints only)", "kind": "edit",
                           "before": str(step["in"]), "mesh": str(step["out"])})
            repro["runs"].append(rec)
            print(f"  run {k}: NOT re-runnable: {how}")
            continue
        iters = int(argv[argv.index("--iters") + 1])
        every = a.every or max(25, iters // 120)
        snaps = rd / "snaps"
        cmd = [PY, str(REFINE)] + argv + ["--out", str(rd / "out.json"), "--log", str(rd / "trace.json"),
                                          "--snapshot-dir", str(snaps), "--snapshot-every", str(every),
                                          "--quiet", "--no-metrics"]
        key = hashlib.sha256(json.dumps([cmd[2:], C.sha256(step["in"])]).encode()).hexdigest()
        stamp = rd / "cmd.json"
        cached = stamp.is_file() and json.loads(stamp.read_text()).get("key") == key and (rd / "out.json").is_file()
        if not cached or a.force:
            if snaps.exists():
                for f in snaps.glob("snap_*.json"):
                    f.unlink()
            rd.mkdir(parents=True, exist_ok=True)
            print(f"  run {k}: re-running ({how}): {' '.join(shlex.quote(x) for x in cmd[1:])}", flush=True)
            t1 = time.time()
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            (rd / "refine.log").write_text(r.stdout + r.stderr)
            if r.returncode:
                ok_all = False
                rec.update({"reproduced": False, "why": f"refine exited {r.returncode}, see {C.rel(rd / 'refine.log')}"})
                stages.append({"label": "refine (re-run failed: endpoints only)", "kind": "edit",
                               "before": str(step["in"]), "mesh": str(step["out"])})
                repro["runs"].append(rec)
                continue
            stamp.write_text(json.dumps({"key": key, "cmd": cmd, "seconds": round(time.time() - t1, 1)}, indent=1))
        files = sorted(snaps.glob("snap_*.json"))
        last = json.loads(files[-1].read_text())
        recd = json.loads(step["out"].read_text())
        dev = float(np.abs(np.asarray(last["vertices"]) - np.asarray(recd["vertices"])).max())
        tri_ok = last.get("triangles", recd["triangles"]) == recd["triangles"]
        good = dev <= REPRO_TOL and tri_ok
        rec.update({"command": cmd[1:], "snapshots": len(files), "every": every, "iters": iters,
                    "max_vertex_deviation_m": dev, "triangles_identical": tri_ok, "reproduced": good,
                    "rerun_seconds": json.loads(stamp.read_text()).get("seconds")})
        repro["runs"].append(rec)
        print(f"  run {k}: {len(files)} snapshots every {every}; last snapshot vs recorded output: "
              f"max |dv| = {dev:.3g} m, triangles {'identical' if tri_ok else 'DIFFER'} -> "
              f"{'reproduced' if good else 'NOT reproduced'}")
        ok_all &= good
        stages.append({"label": "refine", "kind": "refine", "base_mesh": str(step["in"]),
                       "snapshots": str(snaps.relative_to(out)), "reproduced": good,
                       "max_vertex_deviation_m": dev})
        if not good:      # end on the true recorded mesh, never on a non-reproduced one
            stages.append({"label": "recorded result (re-run differs)", "kind": "edit",
                           "before": str(snaps / files[-1].name), "mesh": str(step["out"])})
    stages.append({"label": "final", "kind": "final", "mesh": str(final_path)})
    doc = {"budget": N, "final_checkpoint": sel["tag"], "checkpoint_status": sel["status"],
           "generated_by": "release_video/make_stages.py (fallback; a worker-written "
                           "variants/<N>/timelapse/stages.json takes precedence)",
           "all_refines_reproduced": ok_all, "stages": stages}
    (out / "stages.json").write_text(json.dumps(doc, indent=1))
    repro["all_reproduced"] = ok_all
    repro["seconds"] = round(time.time() - t0, 1)
    (out / "reproduce.json").write_text(json.dumps(repro, indent=1))
    print(f"wrote {C.rel(out / 'stages.json')} ({len(stages)} stages) in {repro['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
