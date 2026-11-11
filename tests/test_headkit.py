#!/usr/bin/env python3
"""Unit tests for catlib.headkit (templates + attach_head weld on a dummy body tube).

    python3 tests/test_headkit.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/
import _setup  # noqa: E402  (skill scripts on sys.path; LOWPOLY_CASE defaults to examples/cat)
REPO = _setup.REPO
ROOT = _setup.CASE

from catlib import headkit as hk, meshio, topo  # noqa: E402


def dummy_body(head, n):
    """Closed-bottom tube (torso stand-in) whose open top ring sits 15 mm below the head's
    neck loop."""
    V = np.asarray(head["vertices"])
    c = V[head["neck_loop"]].mean(0)
    verts = []
    for k, dz in enumerate((0.015, 0.08, 0.15)):
        for i in range(n):
            th = 2 * math.pi * i / n
            verts.append(c + [0.035 * math.cos(th), 0.035 * math.sin(th), -dz])
    verts.append(c + [0, 0, -0.16])
    T = []
    for k in range(2):
        for i in range(n):
            j = (i + 1) % n
            a, b, cc, d = k * n + i, k * n + j, (k + 1) * n + j, (k + 1) * n + i
            T += [[a, d, cc], [a, cc, b]]
    for i in range(n):
        T.append([2 * n + i, 3 * n, 2 * n + (i + 1) % n])
    lm = {k: list(verts[-1]) for k in meshio.LANDMARKS}
    d = meshio.to_dict(verts, T, ["torso"] * len(T), ["fur_orange"] * len(T), lm, 1000, "test_body")
    d["landmark_attach"] = {"paw_FR": {"vertex": 3 * n}}
    return topo.orient_outward(d), list(range(n))


def edge_counts(T):
    c = Counter()
    for t in T:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            c[(min(a, b), max(a, b))] += 1
    return c


def main():
    fails = 0
    for lod in hk.LODS:
        loc = json.loads((ROOT / "head" / "template" / f"head_L{lod}_local.json").read_text())
        chk = hk.check_lod(loc)
        assert chk["ok"] and chk["mirror_max_err_m"] < 1e-12 and chk["triangulation_symmetric"], (lod, chk)
        head = json.loads((ROOT / "head" / f"head_L{lod}.json").read_text())
        for n in (4, 5, 6, 8, 10, 12):
            body, ring = dummy_body(head, n)
            d, frozen = hk.attach_head(body, head, ring)
            nt = len(body["triangles"]) + head["n_triangles"] + hk.bridge_cost(lod, n)
            ok = len(d["triangles"]) == nt
            ec = edge_counts(d["triangles"])
            ok &= all(v == 2 for v in ec.values())                         # closed 2-manifold
            de = Counter((t[i], t[(i + 1) % 3]) for t in d["triangles"] for i in range(3))
            ok &= all(v == 1 for v in de.values())                         # consistent winding
            ok &= not meshio.validate_dict(d)
            nb = len(body["vertices"])
            loopset = set(i + nb for i in head["neck_loop"])
            ok &= len(frozen) == len(head["vertices"]) - len(loopset) and not (set(frozen) & loopset)
            lmh = meshio.landmarks_from_attach(d["vertices"], d["triangles"], d["landmark_attach"])
            for k in ("ear_tip_R", "ear_tip_L", "eye_R", "eye_L", "nose", "chin"):
                ok &= np.allclose(lmh[k], head["landmarks"][k], atol=1e-9)
            ok &= d["landmark_attach"]["paw_FR"] == body["landmark_attach"]["paw_FR"]
            ok &= d["tri_region"].count("neck") == hk.bridge_cost(lod, n)
            print(f"L{lod} body ring {n:2d}: {len(d['triangles'])} tris (head {head['n_triangles']} + bridge "
                  f"{hk.bridge_cost(lod, n)}) {'OK' if ok else 'FAIL'}")
            fails += not ok
    print("FAILURES", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
