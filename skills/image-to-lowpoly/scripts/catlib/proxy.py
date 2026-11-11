"""Read access to the inferred proxy cat (CONTRACT §11).  numpy-only; Blender-importable.

    from catlib import proxy
    V, T = proxy.load_mesh()
    d = proxy.sdf_lookup(points)                  # signed distance, metres, negative inside
    q, d = proxy.closest_surface(points)          # projection onto the zero level set
    labels = proxy.region_of(points)              # CONTRACT REGIONS label per point
    lm = proxy.landmarks3d()                      # {name: (3,) array} on the proxy surface

Files (written by scripts/build_proxy.py):
    proxy/proxy_mesh.npz   vertices (V,3) float64, triangles (T,3) int32
    proxy/sdf.npz          origin (3,), spacing (scalar), sdf (nx,ny,nz) float32
    proxy/part_labels.npz  origin, spacing, labels (nx,ny,nz) int8 -> part_names,
                           part_regions, special_points / special_radius (eyes, nose)
    proxy/landmarks3d.json {"landmarks": {name: [x,y,z]}, ...}

The proxy is INFERRED geometry fitted to one photo; it is a target/prior, not ground truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import paths as _paths  # noqa: E402
PROXY_DIR = _paths.PROXY
_CACHE: dict = {}


def _npz(name: str, path=None):
    p = Path(path) if path else PROXY_DIR / name
    key = str(p)
    if key not in _CACHE:
        if not p.is_file():
            raise FileNotFoundError(f"{p} missing: run scripts/build_proxy.py")
        with np.load(p, allow_pickle=False) as z:
            _CACHE[key] = {k: z[k] for k in z.files}
    return _CACHE[key]


def clear_cache():
    _CACHE.clear()


def load_mesh(path=None):
    z = _npz("proxy_mesh.npz", path)
    return np.asarray(z["vertices"], np.float64), np.asarray(z["triangles"], np.int64)


def load_sdf(path=None):
    z = _npz("sdf.npz", path)
    return np.asarray(z["origin"], float), float(z["spacing"]), z["sdf"]


def _trilinear(grid, origin, h, pts, outside_value=None):
    pts = np.asarray(pts, float).reshape(-1, 3)
    g = (pts - origin) / h
    n = np.array(grid.shape)
    gc = np.clip(g, 0, n - 1 - 1e-9)
    i0 = np.floor(gc).astype(np.int64)
    f = gc - i0
    i1 = np.minimum(i0 + 1, n - 1)
    out = np.zeros(len(pts))
    for dx in (0, 1):
        ix = i1[:, 0] if dx else i0[:, 0]
        wx = f[:, 0] if dx else 1 - f[:, 0]
        for dy in (0, 1):
            iy = i1[:, 1] if dy else i0[:, 1]
            wy = f[:, 1] if dy else 1 - f[:, 1]
            for dz in (0, 1):
                iz = i1[:, 2] if dz else i0[:, 2]
                wz = f[:, 2] if dz else 1 - f[:, 2]
                out += wx * wy * wz * grid[ix, iy, iz]
    # outside the grid: add the Euclidean distance to the grid box (keeps the sign positive)
    excess = np.linalg.norm((g - gc) * h, axis=1)
    return out + excess


def sdf_lookup(points, path=None) -> np.ndarray:
    """Trilinear signed distance (m, negative inside) at points (N,3)."""
    origin, h, grid = load_sdf(path)
    return _trilinear(grid, origin, h, points)


def sdf_gradient(points, path=None, eps=None) -> np.ndarray:
    origin, h, grid = load_sdf(path)
    eps = eps or 0.5 * h
    p = np.asarray(points, float).reshape(-1, 3)
    g = np.zeros_like(p)
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        g[:, k] = (_trilinear(grid, origin, h, p + d) - _trilinear(grid, origin, h, p - d)) / (2 * eps)
    return g


def closest_surface(points, iters: int = 6, path=None):
    """Project points onto the proxy zero level set by SDF-gradient steps.
    Returns (projected (N,3), signed distance of the input points (N,))."""
    p = np.asarray(points, float).reshape(-1, 3).copy()
    d0 = sdf_lookup(p, path)
    for _ in range(iters):
        d = sdf_lookup(p, path)
        g = sdf_gradient(p, path)
        gn = np.linalg.norm(g, axis=1, keepdims=True)
        p = p - d[:, None] * g / np.maximum(gn, 1e-9)
    return p, d0


def _labels(path=None):
    return _npz("part_labels.npz", path)


def region_of(points, path=None) -> list[str]:
    """CONTRACT REGIONS label for each point (N,3): nearest voxel of the proxy part-label
    grid (argmin of the per-part SDFs before the smooth union), with 'eyes' / 'nose'
    overriding inside small spheres around the proxy eye / nose landmarks."""
    z = _labels(path)
    origin, h = np.asarray(z["origin"], float), float(z["spacing"])
    lab = z["labels"]
    regions = [str(x) for x in z["part_regions"]]
    p = np.asarray(points, float).reshape(-1, 3)
    idx = np.rint((p - origin) / h).astype(np.int64)
    idx = np.clip(idx, 0, np.array(lab.shape) - 1)
    li = lab[idx[:, 0], idx[:, 1], idx[:, 2]]
    out = [regions[i] if i >= 0 else "torso" for i in li]
    sp = np.asarray(z["special_points"], float)
    sr = np.asarray(z["special_radius"], float)
    sn = [str(x) for x in z["special_regions"]]
    for c, r, name in zip(sp, sr, sn):
        hit = np.flatnonzero(np.linalg.norm(p - c, axis=1) <= r)
        for i in hit:
            out[i] = name
    return out


def part_of(points, path=None) -> list[str]:
    """Skeleton part name (torso, neck, head, ear_R, ..., tail) for each point."""
    z = _labels(path)
    origin, h = np.asarray(z["origin"], float), float(z["spacing"])
    lab = z["labels"]
    names = [str(x) for x in z["part_names"]]
    p = np.asarray(points, float).reshape(-1, 3)
    idx = np.clip(np.rint((p - origin) / h).astype(np.int64), 0, np.array(lab.shape) - 1)
    return [names[i] if i >= 0 else "torso" for i in lab[idx[:, 0], idx[:, 1], idx[:, 2]]]


def landmarks3d(path=None) -> dict:
    p = Path(path) if path else PROXY_DIR / "landmarks3d.json"
    d = json.loads(p.read_text())
    return {k: np.asarray(v, float) for k, v in d["landmarks"].items()}
