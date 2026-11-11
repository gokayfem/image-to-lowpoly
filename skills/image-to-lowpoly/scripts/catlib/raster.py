"""Fast exact CPU rasteriser (numpy only; importable inside Blender).

Coverage semantics: a pixel (column i, row j) is covered by a triangle iff its centre
(i + 0.5, j + 0.5) lies inside or on the triangle's edges, in the CONTRACT §3 pixel
convention.  This approximates a binarised (>= 50 % coverage) anti-aliased render.

Method: vectorised scanline.  For every (triangle, pixel row) pair the horizontal line
through the row's pixel centres is intersected with the triangle's three edges, giving a
span [x_lo, x_hi]; covered columns are ceil(x_lo - .5) .. floor(x_hi - .5).  The union of
spans is accumulated with a difference array (np.bincount) and a cumulative sum.  No
Python loop over triangles.
"""
from __future__ import annotations

import numpy as np

from .camera import Camera


def _spans(P: np.ndarray, W: int, H: int):
    """P (T,3,2) pixel-convention vertex positions.  Returns (tri, row, c0, c1) int arrays,
    one entry per non-empty span, c0 <= c1, all inside the image."""
    T = len(P)
    if T == 0:
        z = np.zeros(0, np.int64)
        return z, z, z, z
    ys = P[:, :, 1]
    r0 = np.ceil(ys.min(1) - 0.5).astype(np.int64)
    r1 = np.floor(ys.max(1) - 0.5).astype(np.int64)
    r0 = np.maximum(r0, 0)
    r1 = np.minimum(r1, H - 1)
    n = r1 - r0 + 1
    n[n < 0] = 0
    total = int(n.sum())
    if total == 0:
        z = np.zeros(0, np.int64)
        return z, z, z, z
    tri = np.repeat(np.arange(T), n)
    start = np.cumsum(n) - n
    rows = r0[tri] + (np.arange(total) - np.repeat(start, n))
    yc = rows + 0.5
    lo = np.full(total, np.inf)
    hi = np.full(total, -np.inf)
    for e in range(3):
        a = P[:, e]
        b = P[:, (e + 1) % 3]
        ay, by = a[:, 1][tri], b[:, 1][tri]
        ax, bx = a[:, 0][tri], b[:, 0][tri]
        dy = by - ay
        ok = (dy != 0) & (yc >= np.minimum(ay, by)) & (yc <= np.maximum(ay, by))
        with np.errstate(divide="ignore", invalid="ignore"):
            x = ax + (yc - ay) / dy * (bx - ax)
        lo = np.where(ok, np.minimum(lo, x), lo)
        hi = np.where(ok, np.maximum(hi, x), hi)
    c0 = np.ceil(lo - 0.5)
    c1 = np.floor(hi - 0.5)
    good = np.isfinite(lo) & np.isfinite(hi)
    c0 = np.where(good, c0, 1).astype(np.int64)
    c1 = np.where(good, c1, 0).astype(np.int64)
    c0 = np.maximum(c0, 0)
    c1 = np.minimum(c1, W - 1)
    keep = c1 >= c0
    return tri[keep], rows[keep], c0[keep], c1[keep]


def fill_screen(P: np.ndarray, W: int, H: int) -> np.ndarray:
    """Union coverage mask (H,W) bool of screen-space triangles P (T,3,2)."""
    tri, rows, c0, c1 = _spans(P, W, H)
    if len(rows) == 0:
        return np.zeros((H, W), bool)
    Wp = W + 1
    m = Wp * H
    acc = np.bincount(rows * Wp + c0, minlength=m) - np.bincount(rows * Wp + c1 + 1, minlength=m)
    return (np.cumsum(acc.reshape(H, Wp), axis=1)[:, :W] > 0)


def project_triangles(vertices, triangles, camera: Camera, size=None):
    """-> (P (T,3,2), depth (T,3), cam) with camera scaled to `size` (int or (W,H))."""
    cam = camera.scaled(size) if size is not None else camera
    V = np.asarray(vertices, float).reshape(-1, 3)
    T = np.asarray(triangles, np.int64).reshape(-1, 3)
    u, v, d = cam.project(V)
    uv = np.stack([u, v], 1)
    return uv[T], d[T], cam


def silhouette(vertices, triangles, camera: Camera, size=None, return_info: bool = False):
    """Binary silhouette (H,W) bool of a triangle mesh through `camera` at `size`
    (int = square, or (W,H); default = the camera's own resolution).  All triangles count
    (no back-face culling).  Triangles with any vertex at depth <= clip_start are dropped
    and reported in info['dropped_behind_camera'] (no near-plane clipping)."""
    P, depth, cam = project_triangles(vertices, triangles, camera, size)
    front = (depth > cam.clip_start).all(1)
    mask = fill_screen(P[front], cam.width, cam.height)
    if return_info:
        return mask, {"dropped_behind_camera": int((~front).sum()), "size": [cam.width, cam.height]}
    return mask


def id_buffer_screen(P: np.ndarray, key: np.ndarray, W: int, H: int):
    """Z-buffered triangle-ID image.  P (T,3,2) screen positions, key (T,3) per-vertex
    value interpolated linearly in screen space; the fragment with the LARGEST key wins
    (pass 1/depth for perspective, -depth for orthographic).
    Returns (ids (H,W) int64 with -1 = empty, keymap (H,W) float)."""
    ids = np.full(H * W, -1, np.int64)
    kmap = np.full(H * W, -np.inf)
    tri, rows, c0, c1 = _spans(P, W, H)
    if len(rows) == 0:
        return ids.reshape(H, W), kmap.reshape(H, W)
    n = c1 - c0 + 1
    total = int(n.sum())
    ft = np.repeat(tri, n)
    fr = np.repeat(rows, n)
    start = np.cumsum(n) - n
    fc = np.repeat(c0, n) + (np.arange(total) - np.repeat(start, n))
    px = fc + 0.5
    py = fr + 0.5
    a, b, c = P[ft, 0], P[ft, 1], P[ft, 2]
    den = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    den = np.where(den == 0, 1e-30, den)
    wb = ((px - a[:, 0]) * (c[:, 1] - a[:, 1]) - (py - a[:, 1]) * (c[:, 0] - a[:, 0])) / den
    wc = ((b[:, 0] - a[:, 0]) * (py - a[:, 1]) - (b[:, 1] - a[:, 1]) * (px - a[:, 0])) / den
    wa = 1.0 - wb - wc
    k = key[ft]
    fk = wa * k[:, 0] + wb * k[:, 1] + wc * k[:, 2]
    pix = fr * W + fc
    order = np.lexsort((-fk, pix))
    pix_s = pix[order]
    first = np.ones(len(order), bool)
    first[1:] = pix_s[1:] != pix_s[:-1]
    sel = order[first]
    ids[pix[sel]] = ft[sel]
    kmap[pix[sel]] = fk[sel]
    return ids.reshape(H, W), kmap.reshape(H, W)


def id_buffer(vertices, triangles, camera: Camera, size=None):
    """Perspective z-buffered triangle-ID image through `camera`.  Returns
    (ids (H,W), depth (H,W) with inf where empty)."""
    P, depth, cam = project_triangles(vertices, triangles, camera, size)
    front = (depth > cam.clip_start).all(1)
    idx = np.flatnonzero(front)
    if len(idx) == 0:
        return np.full((cam.height, cam.width), -1, np.int64), np.full((cam.height, cam.width), np.inf)
    ids, k = id_buffer_screen(P[front], 1.0 / depth[front], cam.width, cam.height)
    out = np.where(ids >= 0, idx[np.maximum(ids, 0)], -1)
    with np.errstate(divide="ignore"):
        d = np.where(ids >= 0, 1.0 / k, np.inf)
    return out, d


def ortho_id_buffer(vertices, triangles, direction, size: int = 512, margin: float = 0.03):
    """Orthographic z-buffered ID image looking along -direction (i.e. the viewer sits at
    +direction, infinitely far), framing the mesh bounding sphere.  Returns ids (size,size)."""
    V = np.asarray(vertices, float)
    Tr = np.asarray(triangles, np.int64)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(helper, d)
    x /= np.linalg.norm(x)
    y = np.cross(d, x)
    c = 0.5 * (V.min(0) + V.max(0))
    r = np.linalg.norm(V - c, axis=1).max() * (1 + margin)
    q = V - c
    s = size / (2 * r)
    P2 = np.stack([size / 2 + s * (q @ x), size / 2 - s * (q @ y)], 1)
    key = q @ d                     # larger = nearer to the viewer
    ids, _ = id_buffer_screen(P2[Tr], key[Tr], size, size)
    return ids
