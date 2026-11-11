"""Naive decimation control every budget must beat or explain (EXPERIMENT_BRIEF step 3).

    python3 scripts/make_decimation_baseline.py --budget 100 [--budget 200 ...]
        [--eval official|no-render|none] [--refine-iters N | --refine-minutes M]
        [--skip-refine] [--proxy-dir proxy] [--seed 0]

1. Blender (background, CPU only, no rendering; this same file re-run inside Blender)
   applies Decimate COLLAPSE (triangulate on) to proxy/proxy_mesh.npz, bisecting the ratio
   to land on exactly N triangles or the closest count below N.
2. tri_region from catlib.proxy.region_of(face centroids); every face fur_orange;
   landmarks = closest point on the decimated surface to proxy/landmarks3d.json, then
   meshio.attach_landmarks -> baselines/decimate/<N>/checkpoints/v0/mesh.json ("decimate_<N>").
3. The same mesh after scripts/refine_vertices.py with default weights ->
   baselines/decimate_refined/<N>/checkpoints/v0/mesh.json ("decimate_refined_<N>").
4. Both evaluated with scripts/evaluate.py --mesh ... --views (official: it renders
   through the shared GPU lock by itself; --eval no-render gives unofficial raster numbers).

Vertices below z = 0 after decimation (if any) are clamped to z = 0 and counted in the notes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the folder that holds catlib/
from catlib import paths as _paths  # noqa: E402  (resolves the case: --case / LOWPOLY_CASE / cwd)
ROOT = _paths.ROOT

try:
    import bpy  # noqa: F401
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


# =============================================================================== Blender side

def blender_main(argv):
    import bpy
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, required=True)
    a = ap.parse_args(argv)
    z = np.load(a.inp)
    V = z["vertices"].astype(float)
    T = z["triangles"].astype(int)
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    me = bpy.data.meshes.new("proxy")
    me.from_pydata(V.tolist(), [], T.tolist())
    me.update()
    ob = bpy.data.objects.new("proxy", me)
    bpy.context.scene.collection.objects.link(ob)
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.use_collapse_triangulate = True
    mod.use_symmetry = False
    N = a.budget
    tried = {}

    def count(r):
        r = float(min(max(r, 1e-6), 1.0))
        if r in tried:
            return tried[r][0]
        mod.ratio = r
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        ev = ob.evaluated_get(dg)
        m = ev.to_mesh()
        m.calc_loop_triangles()
        n = len(m.loop_triangles)
        tried[r] = (n, None)
        ev.to_mesh_clear()
        return n

    lo, hi = 0.0, min(1.0, 4.0 * N / len(T))
    best = None
    r = N / len(T)
    for _ in range(48):
        n = count(r)
        if n <= N and (best is None or n > best[1] or (n == best[1] and r < best[0])):
            best = (r, n)
        if n == N:
            break
        if n < N:
            lo = r
        else:
            hi = r
        r = 0.5 * (lo + hi)
    if best is None or best[1] < N:                    # small local scan around the best ratio
        r0 = best[0] if best else N / len(T)
        for k in range(1, 200):
            for s in (1, -1):
                rr = r0 * (1 + s * k * 2e-4)
                n = count(rr)
                if n <= N and (best is None or n > best[1]):
                    best = (rr, n)
            if best and best[1] == N:
                break
    mod.ratio = best[0]
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = ob.evaluated_get(dg)
    m = ev.to_mesh()
    m.calc_loop_triangles()
    EV = np.array([v.co[:] for v in m.vertices], float)
    ET = np.array([lt.vertices[:] for lt in m.loop_triangles], int)
    ev.to_mesh_clear()
    np.savez(a.out, vertices=EV, triangles=ET, ratio=best[0], count=best[1],
             tried=np.array([[k, v[0]] for k, v in sorted(tried.items())], float))
    print(f"DECIMATE_OK budget {N} ratio {best[0]:.6f} -> {best[1]} triangles ({len(tried)} ratios tried)")


# =============================================================================== venv side

def run(cmd, log=None):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text("$ " + " ".join(str(c) for c in cmd) + "\n" + p.stdout + p.stderr)
    return p


def make_one(N: int, a) -> dict:
    import numpy as np
    from catlib import geom, meshio, paths
    from catlib import proxy as P
    pdir = Path(a.proxy_dir)
    work = paths.CACHE / "decimate"
    work.mkdir(parents=True, exist_ok=True)
    dec_npz = work / f"decimate_{N}.npz"
    ck = paths.BASELINES / "decimate" / str(N) / "checkpoints" / "v0"
    ck.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = run([paths.BLENDER, "-b", "--factory-startup", "--python", Path(__file__).resolve(), "--",
             "--in", pdir / "proxy_mesh.npz", "--out", dec_npz, "--budget", N], ck / "logs" / "decimate.log")
    if p.returncode != 0 or "DECIMATE_OK" not in p.stdout:
        raise RuntimeError(f"Blender decimation failed for {N}: {p.stdout[-1500:]} {p.stderr[-1500:]}")
    print([ln for ln in p.stdout.splitlines() if "DECIMATE_OK" in ln][0])
    z = np.load(dec_npz)
    V = z["vertices"].astype(float)
    T = z["triangles"].astype(np.int64)
    used = np.unique(T)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    V, T = V[used], remap[T]
    nclamp = int((V[:, 2] < 0).sum())
    V[:, 2] = np.maximum(V[:, 2], 0.0)
    reg = P.region_of(V[T].mean(1), pdir / "part_labels.npz")
    lm3 = P.landmarks3d(pdir / "landmarks3d.json")
    names = [n for n in meshio.LANDMARKS if n in lm3]
    _, snapped = geom.point_mesh_distance(np.stack([lm3[n] for n in names]), V, T)
    d = meshio.to_dict(V, T, reg, ["fur_orange"] * len(T), dict(zip(names, snapped)), N, f"decimate_{N}",
                       notes=(f"NAIVE CONTROL: Blender Decimate COLLAPSE (triangulate) of proxy/proxy_mesh.npz, "
                              f"ratio {float(z['ratio']):.6f} -> {len(T)} triangles; tri_region from "
                              f"catlib.proxy.region_of; all fur_orange; landmarks = closest surface point to "
                              f"proxy/landmarks3d.json, attached; {nclamp} vertices clamped to z>=0."))
    meshio.attach_landmarks(d)
    mj = meshio.save(ck / "mesh.json", d)
    rep = geom.topology_report(V, T)
    info = {"budget": N, "triangles": len(T), "ratio": float(z["ratio"]), "clamped_z": nclamp,
            "degenerate": len(rep["degenerate_triangles"]), "components": rep["components"],
            "closed_manifold": rep["closed_manifold"], "decimate_s": round(time.time() - t0, 1),
            "mesh": str(mj)}
    print(json.dumps(info))
    out = {"decimate": info}
    if a.eval != "none":
        out["decimate_eval_rc"] = evaluate(mj, a)
    if a.skip_refine:
        return out
    rk = paths.BASELINES / "decimate_refined" / str(N) / "checkpoints" / "v0"
    rk.mkdir(parents=True, exist_ok=True)
    rj = rk / "mesh.json"
    cmd = [paths.VENV_PYTHON, paths.SCRIPTS / "refine_vertices.py", "--mesh", mj, "--out", rj,
           "--seed", a.seed, "--log", rk / "refine_trace.json", "--proxy-dir", pdir]
    cmd += ["--minutes", a.refine_minutes] if a.refine_minutes else ["--iters", a.refine_iters]
    t1 = time.time()
    p = run(cmd, rk / "logs" / "refine.log")
    if p.returncode != 0:
        raise RuntimeError(f"refine failed for {N}: {p.stdout[-1500:]} {p.stderr[-1500:]}")
    print("\n".join(p.stdout.strip().splitlines()[-7:]))
    rd = json.loads(rj.read_text())
    rd["name"] = f"decimate_refined_{N}"
    rd["notes"] = ("NAIVE CONTROL + shared refiner (default weights): " + rd.get("notes", ""))
    meshio.save(rj, rd)
    out["refined"] = {"mesh": str(rj), "triangles": len(rd["triangles"]), "refine_s": round(time.time() - t1, 1)}
    if a.eval != "none":
        out["refined_eval_rc"] = evaluate(rj, a)
    return out


def evaluate(mesh_json, a) -> int:
    from catlib import paths
    cmd = [paths.VENV_PYTHON, paths.SCRIPTS / "evaluate.py", "--mesh", mesh_json, "--proxy",
           Path(a.proxy_dir) / "proxy_mesh.npz"]
    cmd += ["--views"] if a.eval == "official" else ["--no-render"]
    p = run(cmd)
    print(p.stdout.strip()[-1200:])
    if p.returncode != 0:
        print(f"evaluate rc {p.returncode}: {p.stderr[-800:]}", file=sys.stderr)
    return p.returncode


def main(argv=None) -> int:
    from catlib import paths
    ap = argparse.ArgumentParser(description="Decimation baseline (naive control) for given budgets.")
    ap.add_argument("--budget", type=int, action="append", required=True)
    ap.add_argument("--eval", choices=("official", "no-render", "none"), default="official")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--refine-iters", type=int, default=20000)
    g.add_argument("--refine-minutes", type=float, default=None)
    ap.add_argument("--skip-refine", action="store_true")
    ap.add_argument("--proxy-dir", type=Path, default=paths.ROOT / "proxy")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    summary = {}
    for N in a.budget:
        summary[N] = make_one(N, a)
    print(json.dumps(summary, indent=1))
    bad = [N for N, s in summary.items() if any(v not in (0, None) for k, v in s.items() if k.endswith("_rc"))]
    return 1 if bad else 0


if __name__ == "__main__":
    if IN_BLENDER:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
        blender_main(argv)
    else:
        sys.exit(main())
