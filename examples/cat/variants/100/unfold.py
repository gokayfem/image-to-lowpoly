"""Greedy fold repair (CPU): for every triangle seen from its back in the reference view, try
flipping each of its edges (catlib.topo.edge_flip, triangle count unchanged) and keep a flip
only if the total visible back-face area (reference view + the 8 novel views, 384 px) drops.
Optional FREEZE.json: edges whose two triangles touch a frozen (head-kit) vertex are never
flipped, so a rigid head keeps its connectivity.
    python3 variants/100/unfold.py IN.json OUT.json [FREEZE.json]"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # the case directory (examples/cat)
import os as _os  # noqa: E402
_os.environ["LOWPOLY_CASE"] = str(ROOT)  # catlib.paths reads the case from here
sys.path.insert(0, _os.environ.get("LOWPOLY_SCRIPTS",
                                   str(ROOT.parents[1] / "skills" / "image-to-lowpoly" / "scripts")))
import numpy as np
from catlib import meshio, raster, topo
from catlib.camera import Camera, novel_view_cameras

CAMS = [Camera.from_json().scaled((512, 512))] + list(novel_view_cameras(size=(384, 384)).values())


def backface(d):
    V = np.asarray(d["vertices"], float); T = np.asarray(d["triangles"], np.int64)
    tri = V[T]; c = tri.mean(1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    tot, per = 0, {}
    for k, cam in enumerate(CAMS):
        ids = raster.id_buffer(V, T, cam)
        ids = ids[0] if isinstance(ids, tuple) else ids
        vis, cnt = np.unique(ids[ids >= 0], return_counts=True)
        for i, m in zip(vis, cnt):
            if np.dot(n[i], cam.location - c[i]) < 0:
                tot += int(m) * (3 if k == 0 else 1)
                per[int(i)] = per.get(int(i), 0) + int(m)
    return tot, per


def main():
    d = json.load(open(sys.argv[1]))
    frozen = set(json.load(open(sys.argv[3]))) if len(sys.argv) > 3 else set()
    cur, per = backface(d)
    print("start back-face px", cur)
    flips = []
    improved = True
    while improved:
        improved = False
        for ti in sorted(per, key=lambda i: -per[i]):
            t = d["triangles"][ti]
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                if frozen and any(set(tt) & frozen for tt in d["triangles"] if a in tt and b in tt):
                    continue
                try:
                    d2 = topo.edge_flip(d, a, b)
                except ValueError:
                    continue
                v2, p2 = backface(d2)
                if v2 < cur:
                    d, cur, per = d2, v2, p2
                    flips.append([int(a), int(b)])
                    print("flip", a, b, "->", cur)
                    improved = True
                    break
            if improved:
                break
    meshio.save(sys.argv[2], d)
    Path(sys.argv[2]).with_suffix(".flips.json").write_text(json.dumps(flips))
    print("end back-face px", cur, "flips", flips)


if __name__ == "__main__":
    main()
