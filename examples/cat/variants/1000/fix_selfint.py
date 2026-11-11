#!/usr/bin/env python
"""Targeted edit: remove self-intersections the refiner created (it has no intersection
constraint).  Repeatedly: find intersecting pairs (Blender BVH, same test as the
validator), Laplacian-relax the vertices of the offending triangles half-way to their
1-ring centroid (ground vertices keep z = 0, landmark-attached vertices are not moved).
Connectivity and count unchanged.
    python3 variants/1000/fix_selfint.py --mesh in.json --out out.json [--max-iter 12]"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]  # the case directory (examples/cat)
import os as _os  # noqa: E402
_os.environ["LOWPOLY_CASE"] = str(ROOT)  # catlib.paths reads the case from here
sys.path.insert(0, _os.environ.get("LOWPOLY_SCRIPTS",
                                   str(ROOT.parents[1] / "skills" / "image-to-lowpoly" / "scripts")))
from catlib import meshio, paths

def pairs_of(d, td):
    p = Path(td) / "m.json"; q = Path(td) / "p.json"
    p.write_text(json.dumps({"vertices": d["vertices"], "triangles": d["triangles"]}))
    subprocess.run([str(paths.BLENDER), "-b", "--factory-startup", "--python",
                    str(Path(__file__).with_name("selfint_blender.py")), "--", str(p), str(q)],
                   capture_output=True, check=True)
    return json.loads(q.read_text())

ap = argparse.ArgumentParser(); ap.add_argument("--mesh", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--max-iter", type=int, default=12); ap.add_argument("--alpha", type=float, default=0.2); ap.add_argument("--freeze-verts", default=None); a = ap.parse_args()
d = json.loads(Path(a.mesh).read_text())
T = np.asarray(d["triangles"])
nb = [set() for _ in range(len(d["vertices"]))]
for t in T:
    for i in range(3):
        nb[t[i]].update((t[(i + 1) % 3], t[(i + 2) % 3]))
fixed = {int(x["vertex"]) for x in d.get("landmark_attach", {}).values() if "vertex" in x}
if a.freeze_verts:
    fixed |= set(json.loads(Path(a.freeze_verts).read_text()))
with tempfile.TemporaryDirectory() as td:
    P = pairs_of(d, td); hist = [len(P)]
    for it in range(a.max_iter):
        if not P:
            break
        V = np.asarray(d["vertices"], float)
        vs = sorted({int(v) for i, j in P for v in list(T[i]) + list(T[j])} - fixed)
        for v in vs:
            c = V[list(nb[v])].mean(0)
            z0 = V[v, 2]
            V[v] = (1 - a.alpha) * V[v] + a.alpha * c
            if z0 == 0.0:
                V[v, 2] = 0.0
        d["vertices"] = V.tolist()
        P = pairs_of(d, td); hist.append(len(P))
meshio.save(a.out, d)
print(json.dumps({"out": a.out, "self_intersecting_pairs_history": hist}))
