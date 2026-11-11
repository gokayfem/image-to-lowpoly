"""Synthetic fixtures for the tooling self-test (NOT a cat reconstruction; test data only).

blob_cat(): a closed low-poly 'blob cat' made of separate closed parts (torso ellipsoid,
head ellipsoid, 2 ear tetrahedra, 4 leg boxes, tail box) in the CONTRACT §2 frame, with
all 16 landmarks snapped onto the surface.  Parts interpenetrate (disclosed, not fatal).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/
import _setup  # noqa: E402  (skill scripts on sys.path; LOWPOLY_CASE defaults to examples/cat)
REPO = _setup.REPO
ROOT = _setup.CASE

from catlib import geom, meshio, raster  # noqa: E402
from catlib.camera import Camera  # noqa: E402


def ellipsoid(center, radii, seg=8, rings=5):
    """UV ellipsoid, closed, CCW from outside.  tris = 2*seg*(rings-1)."""
    cx, cy, cz = center
    V = [(cx, cy, cz + radii[2])]
    for i in range(1, rings):
        th = np.pi * i / rings
        for j in range(seg):
            ph = 2 * np.pi * j / seg
            V.append((cx + radii[0] * np.sin(th) * np.cos(ph), cy + radii[1] * np.sin(th) * np.sin(ph),
                      cz + radii[2] * np.cos(th)))
    V.append((cx, cy, cz - radii[2]))
    T = []
    for j in range(seg):
        T.append((0, 1 + j, 1 + (j + 1) % seg))
    for i in range(rings - 2):
        a0, b0 = 1 + i * seg, 1 + (i + 1) * seg
        for j in range(seg):
            j1 = (j + 1) % seg
            T.append((a0 + j, b0 + j, b0 + j1))
            T.append((a0 + j, b0 + j1, a0 + j1))
    last = len(V) - 1
    base = 1 + (rings - 2) * seg
    for j in range(seg):
        T.append((last, base + (j + 1) % seg, base + j))
    return np.array(V), np.array(T)


def box(p0, p1, half_w):
    """Square prism from p0 to p1 (closed, 12 triangles)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = p1 - p0
    ax /= np.linalg.norm(ax)
    h = np.array([1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1.0, 0])
    a = np.cross(ax, h); a /= np.linalg.norm(a)
    b = np.cross(ax, a)
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    V = [p0 + half_w * (s * a + t * b) for s, t in corners] + [p1 + half_w * (s * a + t * b) for s, t in corners]
    T = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7)]
    for k in range(4):
        k1 = (k + 1) % 4
        T += [(k, k1, 4 + k1), (k, 4 + k1, 4 + k)]
    V, T = np.array(V), np.array(T)
    return orient_outward(V, T)


def tetra(base_center, tip, r, normal_hint):
    bc = np.asarray(base_center, float)
    tip = np.asarray(tip, float)
    ax = tip - bc; ax /= np.linalg.norm(ax)
    a = np.cross(ax, normal_hint); a /= np.linalg.norm(a)
    b = np.cross(ax, a)
    V = [bc + r * (np.cos(t) * a + np.sin(t) * b) for t in (0, 2.1, 4.2)] + [tip]
    T = np.array([(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)])
    return orient_outward(np.array(V), T)


def orient_outward(V, T):
    vol = (V[T[:, 0]] * np.cross(V[T[:, 1]], V[T[:, 2]])).sum() / 6.0
    c = V.mean(0)
    vol = ((V[T[:, 0]] - c) * np.cross(V[T[:, 1]] - c, V[T[:, 2]] - c)).sum() / 6.0
    return (V, T[:, ::-1].copy()) if vol < 0 else (V, T)


PARTS = []


def blob_cat(budget: int | None = None, name: str = "cat_fixture"):
    parts = [
        ("torso", "fur_orange", ellipsoid((0.0, 0.0, 0.17), (0.20, 0.08, 0.075), 10, 6)),
        ("head", "fur_orange", ellipsoid((0.23, -0.02, 0.29), (0.06, 0.065, 0.06), 8, 5)),
        ("ears", "ear_inner", tetra((0.23, -0.055, 0.335), (0.235, -0.07, 0.40), 0.022, np.array([1.0, 0, 0]))),
        ("ears", "ear_inner", tetra((0.23, 0.015, 0.335), (0.235, 0.03, 0.40), 0.022, np.array([1.0, 0, 0]))),
        ("leg_FR", "fur_cream", box((0.14, -0.045, 0.0), (0.14, -0.045, 0.16), 0.018)),
        ("leg_FL", "fur_cream", box((0.10, 0.045, 0.0), (0.10, 0.045, 0.16), 0.018)),
        ("leg_HR", "fur_cream", box((-0.14, -0.045, 0.0), (-0.14, -0.045, 0.16), 0.02)),
        ("leg_HL", "fur_cream", box((-0.11, 0.045, 0.0), (-0.11, 0.045, 0.16), 0.02)),
        ("tail", "fur_stripe", box((-0.19, 0.0, 0.20), (-0.33, 0.0, 0.42), 0.014)),
    ]
    Vs, Ts, reg, mat = [], [], [], []
    off = 0
    for r, mt, (V, T) in parts:
        Vs.append(V); Ts.append(T + off); off += len(V)
        reg += [r] * len(T); mat += [mt] * len(T)
    V = np.concatenate(Vs); T = np.concatenate(Ts)
    rough = {
        "ear_tip_R": (0.235, -0.07, 0.40), "ear_tip_L": (0.235, 0.03, 0.40),
        "eye_R": (0.28, -0.045, 0.30), "eye_L": (0.28, 0.005, 0.30), "nose": (0.29, -0.02, 0.28),
        "paw_FR": (0.14, -0.045, 0.0), "paw_FL": (0.10, 0.045, 0.0),
        "paw_HR": (-0.15, -0.045, 0.0), "paw_HL": (-0.11, 0.045, 0.0),
        "tail_base": (-0.19, 0.0, 0.20), "tail_mid": (-0.26, -0.02, 0.31), "tail_tip": (-0.33, 0.0, 0.43),
        "shoulder_R": (0.13, -0.08, 0.2), "hip_R": (-0.13, -0.08, 0.18), "hock_HR": (-0.15, -0.07, 0.06),
        "chin": (0.25, -0.02, 0.23),
    }
    _, snapped = geom.point_mesh_distance(np.array(list(rough.values())), V, T)
    lms = dict(zip(rough, snapped))
    return meshio.to_dict(V, T, reg, mat, lms, budget if budget is not None else len(T), name,
                          notes="synthetic tooling-test fixture, not a reconstruction")


def synthetic_derived(mesh_dict: dict, out_dir: Path, cam: Camera, jitter_px: float = 3.0,
                      dilate_px: int = 2) -> dict:
    """Make mask.png / landmarks.json / regions.json / reference_bbox.json from a fixture so
    the metrics can be exercised before the real preflight files exist."""
    from PIL import Image
    from scipy.ndimage import binary_dilation
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    V = np.asarray(mesh_dict["vertices"]); T = np.asarray(mesh_dict["triangles"])
    m = raster.silhouette(V, T, cam)
    if dilate_px:
        m = binary_dilation(m, iterations=dilate_px)
    Image.fromarray((m * 255).astype(np.uint8), "L").save(out_dir / "mask.png")
    rng = np.random.default_rng(1)
    lm = {}
    for k, p in mesh_dict["landmarks"].items():
        uv = cam.project_uv(np.array([p]))[0] + rng.normal(0, jitter_px, 2)
        lm[k] = {"uv": uv.tolist(), "class": "sharp" if k in meshio.SHARP_LANDMARKS else "fuzzy",
                 "note": "synthetic"}
    (out_dir / "landmarks.json").write_text(json.dumps({"image_size": [cam.width, cam.height],
                                                        "landmarks": lm}, indent=1))
    ys, xs = np.nonzero(m)
    u0, u1, v0, v1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    mid_u = 0.5 * (u0 + u1)
    belly = v0 + 0.62 * (v1 - v0)
    regions = {"head": [mid_u + 0.25 * (u1 - u0), v0, u1, v0 + 0.45 * (v1 - v0)],
               "tail": [u0, v0, u0 + 0.3 * (u1 - u0), v0 + 0.6 * (v1 - v0)],
               "legs": [u0, belly, u1, v1],
               "torso": [u0 + 0.15 * (u1 - u0), v0 + 0.3 * (v1 - v0), u1 - 0.1 * (u1 - u0), belly]}
    (out_dir / "regions.json").write_text(json.dumps(regions, indent=1))
    bbox = {"u0": int(u0), "v0": int(v0), "u1": int(u1), "v1": int(v1),
            "diag_px": float(np.hypot(u1 - u0, v1 - v0)), "area_px": int(m.sum())}
    (out_dir / "reference_bbox.json").write_text(json.dumps(bbox))
    return {"dir": out_dir}
