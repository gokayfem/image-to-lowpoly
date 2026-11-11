#!/usr/bin/env python3
"""Fast smoke check, Python only (no Blender, no GPU; writes only __pycache__).

    python3 tests/smoke.py

- every script under skills/image-to-lowpoly/scripts, tests/ and examples/ compiles;
- the catlib package imports from this layout and resolves the worked case examples/cat;
- the four final meshes of the worked case load, and each has exactly its budget of triangles in
  mesh.json and in a plain parse of its .glb (counts A and D of references/counting.md);
- each final's metrics.json reports all four counts equal to the budget, and its mesh hash matches
  the shipped mesh.json.

The Blender suites (test_tooling, test_render, test_integration) and the longer CPU tests
(test_refine, test_headkit) are separate; see the README's Verification section.
"""
from __future__ import annotations

import compileall
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _setup  # noqa: E402

BUDGETS = (100, 200, 500, 1000)


def glb_triangles(path: Path) -> int:
    b = path.read_bytes()
    magic, _, _ = struct.unpack("<III", b[:12])
    assert magic == 0x46546C67, "not a GLB"
    jl, _ = struct.unpack("<II", b[12:20])
    g = json.loads(b[20:20 + jl])
    uses = {}
    for node in g.get("nodes", []):
        if "mesh" in node:
            uses[node["mesh"]] = uses.get(node["mesh"], 0) + 1
    n = 0
    for mi, mesh in enumerate(g["meshes"]):
        for prim in mesh["primitives"]:
            assert prim.get("mode", 4) == 4, "primitive not in TRIANGLES mode"
            n += g["accessors"][prim["indices"]]["count"] // 3 * uses.get(mi, 0)
    return n


def main() -> int:
    fails = 0
    ok = all(compileall.compile_dir(str(d), quiet=1) for d in
             (_setup.SCRIPTS, _setup.TESTS, _setup.REPO / "examples"))
    print(f"compileall: {'ok' if ok else 'FAIL'}")
    fails += not ok

    from catlib import camera, meshio, metrics, paths, raster, topo  # noqa: F401
    print(f"catlib imported from {Path(paths.CATLIB).relative_to(_setup.REPO)}; "
          f"case {Path(paths.ROOT).relative_to(_setup.REPO)}")

    for n in BUDGETS:
        fin = paths.VARIANTS / str(n) / "final"
        d = json.loads((fin / "mesh.json").read_text())
        a = len(d["triangles"])
        dd = glb_triangles(fin / f"cat_{n}.glb")
        m = json.loads((fin / "metrics.json").read_text())
        t = m["triangles"]
        four = (t["mesh_json"], t["blend_evaluated"], t["glb_reimported"], t["glb_parsed"])
        sha = hashlib.sha256((fin / "mesh.json").read_bytes()).hexdigest()
        good = a == n and dd == n and four == (n, n, n, n) and sha == m["provenance"]["mesh_sha256"]
        print(f"budget {n:4d}: mesh.json {a}, parsed glb {dd}, recorded four-way {four}, "
              f"mesh hash {'matches' if sha == m['provenance']['mesh_sha256'] else 'DIFFERS'}  "
              f"{'ok' if good else 'FAIL'}")
        fails += not good
    print("FAILURES", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
