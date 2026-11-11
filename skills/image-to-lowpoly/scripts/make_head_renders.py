#!/usr/bin/env python3
"""Evidence renders of the fitted head kit welded onto a simple neck stub.

    python3 scripts/make_head_renders.py [--lods 0,1,2,3,4] [--no-render]

Per LOD: catlib.headkit.attach_head() welds head/head_L*.json onto a 2-ring neck stub
(a test body, NOT cat geometry), writes head/renders/stub_L*/mesh.json (catmesh-1, the
non-head landmarks are parked on the stub), assembles it with scripts/assemble_variant.py
(Blender, CPU) and renders it with the shared scripts/render_variant.py (it takes the GPU
lock itself) using head/renders/head_scene.json: same lights/materials as the contract
scene but the orbit centred on the head (radius 0.55 m) with views face_front (along the
head's own forward axis), side_R, side_L, top, threeq_L, plus the locked reference camera.
Then builds head/renders/L*_views.png sheets.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the folder that holds catlib/
from catlib import paths as _paths  # noqa: E402  (resolves the case: --case / LOWPOLY_CASE / cwd)
ROOT = _paths.ROOT

from catlib import headkit as hk, meshio  # noqa: E402

BLENDER = str(_paths.BLENDER)
PY = str(_paths.VENV_PYTHON)
RD = ROOT / "head" / "renders"


def neck_stub(head: dict, n_body: int = None, length=0.06):
    """Open tube below the head's neck loop: ring 0 (open, to be bridged) 12 mm below the
    loop, ring 1 `length` below, capped.  Returns (catmesh dict, ring0 indices)."""
    V = np.asarray(head["vertices"])
    loop = V[head["neck_loop"]]
    c = loop.mean(0)
    n = n_body or len(loop)
    # stub axis: straight down (the body worker's neck rises from the chest)
    down = np.array([0, 0, -1.0])
    rel = loop - c
    e1 = rel[0] / np.linalg.norm(rel[0])
    e1 = e1 - (e1 @ down) * down
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(down, e1)
    rx = np.abs(rel @ e1).max() * 0.95
    ry = np.abs(rel @ e2).max() * 0.95
    verts = []
    for k, dz in enumerate((0.012, length)):
        for i in range(n):
            th = 2 * math.pi * i / n
            verts.append(c + dz * down + rx * math.cos(th) * e1 + ry * math.sin(th) * e2)
    tip = c + (length + 0.01) * down
    verts.append(tip)
    T = []
    for i in range(n):
        j = (i + 1) % n
        T += [[i, n + i, n + j], [i, n + j, j]]
        T += [[n + i, 2 * n, n + j]]
    d = {"format": "catmesh-1", "budget": 1000, "name": "neckstub", "vertices": [list(map(float, v)) for v in verts],
         "triangles": T, "tri_region": ["torso"] * len(T), "tri_material": ["fur_orange"] * len(T),
         "landmarks": {}, "notes": "test neck stub for head-kit renders; not cat geometry"}
    return d, list(range(n))


def welded(lod):
    head = json.loads((ROOT / "head" / f"head_L{lod}.json").read_text())
    body, ring = neck_stub(head)
    from catlib import topo
    body = topo.orient_outward(body)
    d, frozen = hk.attach_head(body, head, ring)
    d = topo.orient_outward(d)
    # park the non-head landmarks on stub vertex 0 (test mesh only)
    att = d["landmark_attach"]
    for k in meshio.LANDMARKS:
        if k not in att:
            att[k] = {"vertex": len(body["vertices"]) - 1}
    meshio.sync_landmarks(d)
    d["name"] = f"headkit_L{lod}"
    d["budget"] = 1000
    for k in ("freeze_vertices", "head_kit"):
        d.pop(k, None)
    return d, head


def scene_for(head):
    scn = json.loads((ROOT / "contract" / "scene.json").read_text())
    V = np.asarray(head["vertices"])
    c = 0.5 * (V.min(0) + V.max(0))
    c[2] -= 0.01
    yaw = head["pose"]["yaw_deg"]
    scn["orbit_target"] = c.tolist()
    scn["orbit_radius_m"] = 0.55
    scn["views"] = {"face_front": {"az_deg": yaw, "el_deg": 5}, "side_R": {"az_deg": yaw - 90, "el_deg": 5},
                    "side_L": {"az_deg": yaw + 90, "el_deg": 5}, "threeq_L": {"az_deg": yaw + 45, "el_deg": 15},
                    "back": {"az_deg": yaw + 180, "el_deg": 10},
                    "top": {"straight_down": True, "image_up": [math.cos(math.radians(yaw)),
                                                                math.sin(math.radians(yaw)), 0]}}
    scn["view_resolution"] = [640, 640]
    scn["note"] = "head-kit evidence scene: contract/scene.json with the orbit centred on the head"
    return scn


def sheet(lod, out_dir, ntri):
    from PIL import Image, ImageDraw
    head_ref = json.loads((ROOT / "reference" / "derived" / "regions.json").read_text())["head"]
    u0, v0, u1, v1 = head_ref
    pad = 40
    tiles = []
    photo = Image.open(ROOT / "reference" / "cat.png").convert("RGB")
    box = (u0 - pad, v0 - pad, u1 + pad, v1 + pad + 40)
    for src in [photo, Image.open(out_dir / "ref_clay.png").convert("RGB"),
                Image.open(out_dir / "ref_wire.png").convert("RGB")]:
        tiles.append(src.crop(box).resize((420, int(420 * (box[3] - box[1]) / (box[2] - box[0])))))
    for v in ("face_front", "threeq_L", "side_R", "side_L", "top", "back"):
        for kind in ("clay", "wire"):
            p = out_dir / f"view_{v}_{kind}.png"
            if p.exists() and kind == "wire":
                tiles.append(Image.open(p).convert("RGB").resize((420, 420)))
    h = max(t.height for t in tiles)
    cols = 5
    rows = math.ceil(len(tiles) / cols)
    S = Image.new("RGB", (cols * 420, rows * h + 30), (255, 255, 255))
    names = ["photo", "ref clay", "ref wire", "face front", "3/4 left", "side R", "side L", "top", "back"]
    for i, t in enumerate(tiles):
        x, y = (i % cols) * 420, 30 + (i // cols) * h
        S.paste(t, (x, y))
        ImageDraw.Draw(S).text((x + 6, y + 4), names[i] if i < len(names) else "", fill=(200, 0, 0))
    ImageDraw.Draw(S).text((8, 8), f"Head kit L{lod}: {ntri} head triangles (+ bridge + test neck stub, "
                                   f"not cat geometry)", fill=(0, 0, 0))
    S.save(RD / f"L{lod}_views.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lods", default="0,1,2,3,4")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    for lod in [int(x) for x in a.lods.split(",")]:
        d, head = welded(lod)
        od = RD / f"stub_L{lod}"
        od.mkdir(parents=True, exist_ok=True)
        meshio.save(od / "mesh.json", d)
        (od / "head_scene.json").write_text(json.dumps(scene_for(head), indent=1))
        subprocess.run([BLENDER, "-b", "--factory-startup", "--python", str(_paths.SCRIPTS / "assemble_variant.py"),
                        "--", "--mesh", str(od / "mesh.json")], check=True, capture_output=True)
        if a.no_render:
            continue
        subprocess.run([BLENDER, "-b", "--factory-startup", "--python", str(_paths.SCRIPTS / "render_variant.py"),
                        "--", "--blend", str(od / f"{d['name']}.blend"), "--out", str(od),
                        "--passes", "clay,wire", "--views", "--scene", str(od / "head_scene.json")], check=True)
        sheet(lod, od, head["n_triangles"])
        print("rendered", lod)


if __name__ == "__main__":
    main()
