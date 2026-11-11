#!/usr/bin/env python
"""Remove self-intersections from a fixed-connectivity catmesh-1 with minimal vertex moves.

    python3 variants/200/untangle.py --mesh IN.json --out OUT.json [--max-steps 300]

Deterministic.  Intersection test = every edge vs every non-adjacent triangle (no shared
vertex), segment/triangle Moller-Trumbore, the same pair definition as the validator
(non-adjacent triangle pairs).  Repair: while pairs remain, for the vertices of the
intersecting triangles try moves towards the vertex's 1-ring centroid and towards the
proxy surface (catlib.proxy.closest_surface) at several step sizes; keep the move with the
fewest remaining pairs (ties: smallest displacement).  Sole vertices (z = 0) keep z = 0.
Topology, triangle count, labels and landmark attachments are unchanged (landmarks are
recomputed from their attachments).  The official evaluation decides whether it helped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # the case directory (examples/cat)
import os as _os  # noqa: E402
_os.environ["LOWPOLY_CASE"] = str(ROOT)  # catlib.paths reads the case from here
sys.path.insert(0, _os.environ.get("LOWPOLY_SCRIPTS",
                                   str(ROOT.parents[1] / "skills" / "image-to-lowpoly" / "scripts")))

import numpy as np  # noqa: E402

from catlib import meshio, proxy  # noqa: E402


def edges_of(T):
    E = {}
    for i, t in enumerate(T):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            E.setdefault((min(a, b), max(a, b)), []).append(i)
    return E


def intersecting_pairs(V, T, E=None):
    E = E or edges_of(T)
    ek = np.array(list(E.keys()))
    P0, P1 = V[ek[:, 0]], V[ek[:, 1]]
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1, e2 = B - A, C - A
    D = P1 - P0                                                     # (E,3)
    pairs = set()
    for j in range(len(T)):
        h = np.cross(D, e2[j])
        a = h @ e1[j]
        ok = np.abs(a) > 1e-14
        f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
        s = P0 - A[j]
        u = f * (s * h).sum(1)
        q = np.cross(s, e1[j])
        v = f * (q * D).sum(1)
        t = f * (q @ e2[j])
        hit = ok & (u > 1e-9) & (v > 1e-9) & (u + v < 1 - 1e-9) & (t > 1e-9) & (t < 1 - 1e-9)
        tj = set(T[j].tolist())
        for k in np.flatnonzero(hit):
            a_, b_ = ek[k]
            if a_ in tj or b_ in tj:
                continue
            for i in E[(a_, b_)]:
                if set(T[i].tolist()) & tj:
                    continue
                pairs.add((min(i, j), max(i, j)))
    return pairs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--freeze-verts", type=Path, default=None, help="JSON list of vertices never moved (head kit)")
    a = ap.parse_args(argv)
    d = json.loads(a.mesh.read_text())
    V = np.asarray(d["vertices"], float)
    T = np.asarray(d["triangles"], int)
    E = edges_of(T)
    nbr = [set() for _ in range(len(V))]
    for (x, y) in E:
        nbr[x].add(y); nbr[y].add(x)
    sole = V[:, 2] < 1e-3
    frozen = set(json.loads(a.freeze_verts.read_text())) if a.freeze_verts else set()
    V0 = V.copy()
    pairs = intersecting_pairs(V, T, E)
    print(f"start: {len(pairs)} intersecting pairs")
    steps = 0
    while pairs and steps < a.max_steps:
        steps += 1
        verts = sorted({int(x) for i, j in pairs for x in (*T[i], *T[j])} - frozen)
        best = None
        for vi in verts:
            ring_c = V[sorted(nbr[vi])].mean(0)
            surf = proxy.closest_surface(V[vi][None])[0] if hasattr(proxy, "closest_surface") else ring_c
            surf = np.asarray(surf, float).reshape(-1)[:3]
            for target in (ring_c, surf):
                for s in (0.15, 0.3, 0.5, 0.8):
                    p = V[vi] + s * (target - V[vi])
                    if sole[vi]:
                        p[2] = 0.0
                    old = V[vi].copy()
                    V[vi] = p
                    n = len(intersecting_pairs(V, T, E))
                    V[vi] = old
                    key = (n, float(np.linalg.norm(p - old)))
                    if best is None or key < best[0]:
                        best = (key, vi, p)
        if best is None or best[0][0] >= len(pairs):
            print(f"step {steps}: no single-vertex move reduces {len(pairs)} pairs; stopping")
            break
        V[best[1]] = best[2]
        pairs = intersecting_pairs(V, T, E)
        print(f"step {steps}: moved v{best[1]} by {best[0][1]*1000:.1f} mm -> {len(pairs)} pairs")
    disp = np.linalg.norm(V - V0, axis=1)
    print(f"done: {len(pairs)} pairs left; moved {int((disp > 0).sum())} vertices, max {disp.max()*1000:.1f} mm")
    d["vertices"] = V.tolist()
    d["notes"] = (d.get("notes", "") + f" [untangle.py: {steps} steps, {len(pairs)} pairs left]").strip()
    meshio.save(a.out, d)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
