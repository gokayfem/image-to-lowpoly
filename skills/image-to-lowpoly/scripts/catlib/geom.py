"""Mesh geometry / topology helpers (numpy only; importable inside Blender)."""
from __future__ import annotations

import numpy as np

WELD_TOL = 1e-6          # m: coincident-position tolerance (CONTRACT §6 duplicates)
DEGEN_AREA = 1e-6        # m^2 (1 mm^2)
DEGEN_EDGE = 1e-3        # m (1 mm)
FLAT_ANGLE_DEG = 0.5     # fan normals within this angle -> flat_redundant_vertex


def tri_cross(V, T):
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    return np.cross(B - A, C - A)


def tri_areas(V, T):
    return 0.5 * np.linalg.norm(tri_cross(V, T), axis=1)


def tri_normals(V, T):
    n = tri_cross(V, T)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(ln == 0, 1, ln)


def weld_ids(V, tol: float = WELD_TOL) -> np.ndarray:
    """Map each vertex to a welded id: positions equal after rounding to a `tol` grid."""
    if len(V) == 0:
        return np.zeros(0, np.int64)
    q = np.round(np.asarray(V, float) / tol).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    return inv.reshape(-1)


def _components(n_vert: int, T: np.ndarray) -> np.ndarray:
    """Connected components of triangles sharing a vertex index -> label per triangle."""
    parent = np.arange(n_vert)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b, c in T:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[find(rc)] = ra
    roots = np.array([find(t[0]) for t in T], np.int64)
    _, lab = np.unique(roots, return_inverse=True)
    return lab.reshape(-1)


def closest_point_triangles(p, A, B, C):
    """Closest points on triangles (A,B,C each (N,3)) to one point p (3,).
    Ericson, Real-Time Collision Detection §5.1.5, vectorised.  Returns (N,3)."""
    p = np.asarray(p, float)
    ab, ac, ap = B - A, C - A, p - A
    d1 = (ab * ap).sum(1); d2 = (ac * ap).sum(1)
    bp = p - B
    d3 = (ab * bp).sum(1); d4 = (ac * bp).sum(1)
    cp = p - C
    d5 = (ab * cp).sum(1); d6 = (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    out = np.empty_like(A)
    done = np.zeros(len(A), bool)

    def put(mask, val):
        nonlocal done
        m = mask & ~done
        out[m] = val[m]
        done |= m

    eps = 1e-30
    put((d1 <= 0) & (d2 <= 0), A)
    put((d3 >= 0) & (d4 <= d3), B)
    v = d1 / np.where(np.abs(d1 - d3) < eps, eps, d1 - d3)
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), A + v[:, None] * ab)
    put((d6 >= 0) & (d5 <= d6), C)
    w = d2 / np.where(np.abs(d2 - d6) < eps, eps, d2 - d6)
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), A + w[:, None] * ac)
    w2 = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) < eps, eps, (d4 - d3) + (d5 - d6))
    put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), B + w2[:, None] * (C - B))
    den = va + vb + vc
    den = np.where(np.abs(den) < eps, eps, den)
    vv = vb / den
    ww = vc / den
    put(np.ones(len(A), bool), A + ab * vv[:, None] + ac * ww[:, None])
    return out


def point_mesh_distance(points, V, T):
    """Unsigned distance from each point (N,3) to the triangle mesh, plus closest point."""
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    pts = np.asarray(points, float).reshape(-1, 3)
    dist = np.empty(len(pts))
    cpt = np.empty_like(pts)
    for i, p in enumerate(pts):
        q = closest_point_triangles(p, A, B, C)
        d = np.linalg.norm(q - p, axis=1)
        j = int(np.argmin(d))
        dist[i], cpt[i] = d[j], q[j]
    return dist, cpt


def ray_hits(origin, direction, V, T, eps: float = 1e-12) -> np.ndarray:
    """Möller–Trumbore, both faces.  Returns t per triangle (inf = no hit, t > 0 only)."""
    o = np.asarray(origin, float)
    d = np.asarray(direction, float)
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1, e2 = B - A, C - A
    pv = np.cross(d, e2)
    det = (e1 * pv).sum(1)
    ok = np.abs(det) > eps
    inv = 1.0 / np.where(ok, det, 1.0)
    tv = o - A
    u = (tv * pv).sum(1) * inv
    qv = np.cross(tv, e1)
    v = (qv @ d) * inv
    t = (e2 * qv).sum(1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return np.where(hit, t, np.inf)


def sphere_directions(n_sub: int = 2) -> np.ndarray:
    """Icosphere vertex directions: 12, 42, 162 (n_sub=2), 642 ...  Deterministic."""
    t = (1 + 5 ** 0.5) / 2
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
             (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
             (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
             (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(v, float) / np.linalg.norm(v) for v in verts]
    for _ in range(n_sub):
        cache = {}
        nf = []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[k] = len(verts) - 1
            return cache[k]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nf
    return np.array(verts)


def topology_report(V, T, flat_angle_deg: float = FLAT_ANGLE_DEG) -> dict:
    """Hard-failure facts (degenerate, duplicates, min z, billboard) and disclosed
    findings (CONTRACT §6) computed from vertex/triangle arrays."""
    V = np.asarray(V, float)
    T = np.asarray(T, np.int64)
    nT = len(T)
    out: dict = {}
    # ---- degenerate
    areas = tri_areas(V, T)
    e = np.stack([np.linalg.norm(V[T[:, 1]] - V[T[:, 0]], axis=1),
                  np.linalg.norm(V[T[:, 2]] - V[T[:, 1]], axis=1),
                  np.linalg.norm(V[T[:, 0]] - V[T[:, 2]], axis=1)], 1)
    min_edge = e.min(1)
    degen = np.flatnonzero((areas < DEGEN_AREA) | (min_edge < DEGEN_EDGE))
    out["degenerate_triangles"] = degen.tolist()
    out["min_triangle_area_m2"] = float(areas.min()) if nT else None
    out["min_edge_m"] = float(min_edge.min()) if nT else None
    # ---- duplicates (same index set; or same welded-position set)
    wid = weld_ids(V)
    dup_idx, dup_pos = [], []
    for key_arr, sink in ((np.sort(T, 1), dup_idx), (np.sort(wid[T], 1), dup_pos)):
        _, first, inv, cnt = np.unique(key_arr, axis=0, return_index=True, return_inverse=True,
                                       return_counts=True)
        inv = inv.reshape(-1)
        for t in np.flatnonzero(cnt[inv] > 1):
            if first[inv[t]] != t:
                sink.append([int(first[inv[t]]), int(t)])
    out["duplicate_triangles_same_indices"] = dup_idx
    out["duplicate_triangles_coincident_positions"] = [p for p in dup_pos if p not in dup_idx]
    # ---- unwelded coincident vertices
    used = np.unique(T)
    _, cnt = np.unique(wid[used], return_counts=True)
    out["unwelded_coincident_vertex_groups"] = int((cnt > 1).sum())
    out["unused_vertices"] = int(len(V) - len(used))
    # ---- components
    lab = _components(len(V), T) if nT else np.zeros(0, int)
    ncomp = int(lab.max() + 1) if nT else 0
    comp_tris = np.bincount(lab, minlength=ncomp).tolist() if nT else []
    labw = _components(int(wid.max()) + 1, wid[T]) if nT else np.zeros(0, int)
    out["components"] = ncomp
    out["component_triangles"] = comp_tris
    out["components_after_weld"] = int(labw.max() + 1) if nT else 0
    # ---- edges (on welded vertices so that unwelded seams do not look open)
    Tw = wid[T]
    de = np.concatenate([Tw[:, [0, 1]], Tw[:, [1, 2]], Tw[:, [2, 0]]])       # directed
    ue = np.sort(de, 1)
    _, uinv, ucnt = np.unique(ue, axis=0, return_inverse=True, return_counts=True)
    uinv = uinv.reshape(-1)
    out["edges"] = int(len(ucnt))
    out["boundary_edges"] = int((ucnt == 1).sum())
    out["non_manifold_edges"] = int((ucnt > 2).sum())
    _, dcnt = np.unique(de, axis=0, return_counts=True)
    out["inconsistent_winding_edges"] = int((dcnt > 1).sum())
    out["closed_manifold"] = out["boundary_edges"] == 0 and out["non_manifold_edges"] == 0
    # ---- signed volume per (welded) component
    vol = (V[T[:, 0]] * np.cross(V[T[:, 1]], V[T[:, 2]])).sum(1) / 6.0
    comp_vol = np.bincount(labw, weights=vol, minlength=out["components_after_weld"]) if nT else []
    out["signed_volume_m3"] = float(vol.sum())
    out["component_signed_volume_m3"] = [float(x) for x in comp_vol]
    out["components_with_negative_volume"] = int((np.asarray(comp_vol) < 0).sum()) if nT else 0
    # ---- flat redundant interior vertices
    normals = tri_normals(V, T)
    cos_tol = np.cos(np.radians(flat_angle_deg))
    boundary_verts = set(np.unique(ue[(ucnt[uinv] != 2)]).tolist())
    flat = []
    order = np.argsort(Tw.ravel(), kind="stable")
    tv_sorted = Tw.ravel()[order]
    tri_of = order // 3
    splits = np.flatnonzero(np.diff(tv_sorted)) + 1
    for grp_v, grp in zip(tv_sorted[np.r_[0, splits]] if len(tv_sorted) else [],
                          np.split(tri_of, splits) if len(tv_sorted) else []):
        if grp_v in boundary_verts or len(grp) < 3:
            continue
        n = normals[grp]
        if (n @ n[0] >= cos_tol).all():
            flat.append(int(grp_v))
    out["flat_redundant_vertices"] = len(flat)
    out["flat_angle_tol_deg"] = flat_angle_deg
    # ---- extents
    lo, hi = V[used].min(0), V[used].max(0)
    ext = hi - lo
    out["bbox_min"] = lo.tolist()
    out["bbox_max"] = hi.tolist()
    out["bbox_extent_m"] = ext.tolist()
    out["min_extent_ratio"] = float(ext.min() / ext.max()) if ext.max() > 0 else 0.0
    out["min_z"] = float(lo[2])
    return out


def hidden_triangles(V, T, n_sub: int = 2, size: int = 512, extra_visible=None) -> dict:
    """Triangles that win no pixel in an orthographic z-buffer from any of the icosphere
    directions (162 for n_sub=2).  `extra_visible` (bool per triangle, e.g. from Blender
    centroid ray casts) is OR-ed in so tiny triangles are not falsely called hidden."""
    from .raster import ortho_id_buffer
    V = np.asarray(V, float)
    T = np.asarray(T, np.int64)
    dirs = sphere_directions(n_sub)
    seen = np.zeros(len(T), bool)
    for d in dirs:
        ids = ortho_id_buffer(V, T, d, size)
        seen[np.unique(ids[ids >= 0])] = True
    raster_seen = seen.copy()
    if extra_visible is not None:
        seen |= np.asarray(extra_visible, bool)
    hidden = np.flatnonzero(~seen)
    return {"directions": int(len(dirs)), "raster_size": size,
            "hidden_triangles": int(len(hidden)), "hidden_triangle_ids": hidden.tolist(),
            "hidden_by_raster_only": int((~raster_seen).sum())}
