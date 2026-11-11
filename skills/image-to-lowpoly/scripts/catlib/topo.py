"""Small topology edits on catmesh-1 dicts (CONTRACT §12).  numpy-only.

Every function takes a catmesh-1 dict and returns a NEW dict (the input is not modified)
in which `tri_region`, `tri_material` and `landmark_attach` stay consistent with the
triangles, and `landmarks` are recomputed from `landmark_attach` when present.

Landmark rule: an attachment whose triangle is untouched keeps its (tri, bary) with the
triangle index remapped; an attachment whose triangle is replaced or removed is
re-attached to the closest point on the triangles that replaced it (for splits/flips the
surface is unchanged so the landmark does not move; for collapses it lands on the
surviving fan).  Vertex attachments follow vertex merges.  Choosing where to apply these
edits is the budget worker's job; these are only tools.

    edge_flip(d, a, b)                         2 tris -> 2 tris (count unchanged)
    edge_split(d, a, b, t=0.5, pos=None)       +2 tris for an interior edge (+1 boundary)
    edge_collapse(d, a, b, pos=None, t=0.5)    -2 tris for an interior edge (link checked)
    weld_vertices(d, pairs=None, tol=1e-6)     merge vertices, drop collapsed triangles
    bridge_rings(d, idx_a, idx_b, ...)         len(a)+len(b) tris (2n when equal)
    cap_fan(d, ring, ...)                      len(ring)-2 tris, no new vertex
    cap_tip(d, ring, tip=None, ...)            len(ring) tris, one new apex vertex
    orient_outward(d)                          consistent winding, positive volume per part
    triangle_count(d)
"""
from __future__ import annotations

import copy
from collections import defaultdict, deque

import numpy as np

from . import meshio

DEFAULT_REGION = "torso"
DEFAULT_MATERIAL = "fur_orange"


# ---- helpers -------------------------------------------------------------------------------

def triangle_count(d: dict) -> int:
    return int(len(d["triangles"]))


def _unpack(d):
    V = np.asarray(d["vertices"], float).reshape(-1, 3).copy()
    T = np.asarray(d["triangles"], np.int64).reshape(-1, 3).copy()
    return V, T, list(d["tri_region"]), list(d["tri_material"])


def _landmark_positions(d) -> dict:
    att = d.get("landmark_attach")
    if att:
        return meshio.landmarks_from_attach(d["vertices"], d["triangles"], att)
    return {k: np.asarray(v, float) for k, v in d.get("landmarks", {}).items()}


def _pack(d, V, T, reg, mat, att=None) -> dict:
    out = {k: copy.deepcopy(v) for k, v in d.items()
           if k not in ("vertices", "triangles", "tri_region", "tri_material", "landmark_attach")}
    out["vertices"] = [[float(c) for c in v] for v in V]
    out["triangles"] = [[int(c) for c in t] for t in T]
    out["tri_region"] = list(reg)
    out["tri_material"] = list(mat)
    if att:
        out["landmark_attach"] = att
        meshio.sync_landmarks(out)
    return out


def _remap_attach(d, V, T, vert_map=None, tri_map=None, candidates=None):
    """Remap d['landmark_attach'] after an edit.  vert_map: old vertex -> new vertex.
    tri_map: old tri -> new tri or -1 (replaced/removed).  Replaced-tri attachments are
    re-attached at the old landmark position among `candidates` (new tri ids; default all)."""
    att = d.get("landmark_attach")
    if not att:
        return None
    old_pos = _landmark_positions(d)
    out = {}
    for k, a in att.items():
        if "vertex" in a:
            v = int(a["vertex"])
            out[k] = {"vertex": int(vert_map[v]) if vert_map is not None else v}
            continue
        t = int(a["tri"])
        nt = int(tri_map[t]) if tri_map is not None else t
        if nt >= 0:
            out[k] = {"tri": nt, "bary": list(a["bary"])}
        else:
            out[k] = meshio.attach_point(old_pos[k], V, T, candidates)
    return out


def _directed_edges(T) -> set:
    return {(int(a), int(b)) for t in T for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}


def _tris_with_edge(T, a, b):
    has = ((T == a).any(1)) & ((T == b).any(1))
    return np.flatnonzero(has)


def _rot_to_edge(t, a, b):
    """Rotate triangle t so that it reads (a, b, c) or (b, a, c); returns (x, y, c) with
    x->y the directed edge as it appears in t."""
    t = [int(x) for x in t]
    for r in range(3):
        x, y, c = t[r], t[(r + 1) % 3], t[(r + 2) % 3]
        if {x, y} == {a, b}:
            return x, y, c
    raise ValueError("triangle does not contain the edge")


def _orient_like_neighbours(T_existing, new_tris):
    """Flip all new triangles if they disagree with the existing winding along shared
    edges (a shared edge must appear with opposite directions)."""
    if len(new_tris) == 0 or len(T_existing) == 0:
        return new_tris
    ex = _directed_edges(T_existing)
    agree = disagree = 0
    for t in new_tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            if (a, b) in ex:
                disagree += 1
            elif (b, a) in ex:
                agree += 1
    if disagree > agree:
        return [[t[0], t[2], t[1]] for t in new_tris]
    return new_tris


def _labels_near(T, reg, mat, verts, region, material):
    if region is not None and material is not None:
        return region, material
    vs = set(int(v) for v in verts)
    for i, t in enumerate(T):
        if vs & set(int(x) for x in t):
            return region or reg[i], material or mat[i]
    return region or DEFAULT_REGION, material or DEFAULT_MATERIAL


# ---- edits ---------------------------------------------------------------------------------

def edge_flip(d: dict, a: int, b: int) -> dict:
    """Replace the two triangles sharing edge (a,b) by the two sharing the other diagonal.
    Raises ValueError if the edge is not shared by exactly two consistently wound
    triangles or the new diagonal already exists (would make a non-manifold edge)."""
    V, T, reg, mat = _unpack(d)
    tt = _tris_with_edge(T, a, b)
    if len(tt) != 2:
        raise ValueError(f"edge ({a},{b}) is shared by {len(tt)} triangles, need 2")
    x1, y1, c = _rot_to_edge(T[tt[0]], a, b)
    x2, y2, e = _rot_to_edge(T[tt[1]], a, b)
    if (x1, y1) != (y2, x2):
        raise ValueError("inconsistent winding across the edge")
    if c == e or len(_tris_with_edge(T, c, e)):
        raise ValueError(f"diagonal ({c},{e}) already exists")
    # t1 = (x1,y1,c), t2 = (y1,x1,e); quad cycle x1 -> e -> y1 -> c
    T[tt[0]] = [x1, e, c]
    T[tt[1]] = [e, y1, c]
    tri_map = np.arange(len(T))
    tri_map[tt] = -1
    att = _remap_attach(d, V, T, None, tri_map, candidates=tt)
    return _pack(d, V, T, reg, mat, att)


def edge_split(d: dict, a: int, b: int, t: float = 0.5, pos=None) -> dict:
    """Insert a vertex on edge (a,b) (at a + t (b-a), or at `pos`) and split every adjacent
    triangle in two.  Count +2 for an interior edge, +1 for a boundary edge.  The new
    vertex is the last one; new triangles are appended."""
    V, T, reg, mat = _unpack(d)
    tt = _tris_with_edge(T, a, b)
    if len(tt) == 0:
        raise ValueError(f"no triangle has edge ({a},{b})")
    m = len(V)
    p = np.asarray(pos, float) if pos is not None else V[a] + t * (V[b] - V[a])
    V = np.vstack([V, p])
    new_rows, new_reg, new_mat = [], [], []
    for ti in tt:
        x, y, c = _rot_to_edge(T[ti], a, b)
        T[ti] = [x, m, c]
        new_rows.append([m, y, c]); new_reg.append(reg[ti]); new_mat.append(mat[ti])
    first_new = len(T)
    T = np.vstack([T, np.asarray(new_rows, np.int64)])
    reg += new_reg; mat += new_mat
    tri_map = np.arange(first_new)
    tri_map[tt] = -1
    cand = np.concatenate([tt, np.arange(first_new, len(T))])
    att = _remap_attach(d, V, T, None, tri_map, candidates=cand)
    return _pack(d, V, T, reg, mat, att)


def _drop_vertex(V, T, drop):
    keep = np.ones(len(V), bool)
    keep[drop] = False
    new_id = np.cumsum(keep) - 1
    return V[keep], new_id[T], new_id


def edge_collapse(d: dict, a: int, b: int, pos=None, t: float = 0.5, check_link: bool = True) -> dict:
    """Merge vertex b into a (a moves to `pos`, default a + t (b-a)); triangles containing
    both are removed (-2 for an interior edge).  With check_link (default) a collapse
    that violates the link condition (would create a non-manifold edge) raises."""
    V, T, reg, mat = _unpack(d)
    tt = _tris_with_edge(T, a, b)
    if len(tt) == 0:
        raise ValueError(f"no triangle has edge ({a},{b})")
    if check_link:
        def nbrs(v):
            return set(T[(T == v).any(1)].ravel().tolist()) - {v}
        def link_edges(v):
            return {tuple(sorted(int(x) for x in t if x != v)) for t in T[(T == v).any(1)]
                    if not ({a, b} <= set(int(x) for x in t))}
        common = nbrs(a) & nbrs(b)
        opp = {int(x) for ti in tt for x in T[ti] if x not in (a, b)}
        if common != opp or (link_edges(a) & link_edges(b)):
            raise ValueError(f"link condition violated for ({a},{b}): common {sorted(common)} vs {sorted(opp)}")
    p = np.asarray(pos, float) if pos is not None else V[a] + t * (V[b] - V[a])
    keep_t = np.ones(len(T), bool)
    keep_t[tt] = False
    T[T == b] = a
    V[a] = p
    old_pos = _landmark_positions(d)
    T2 = T[keep_t]
    reg2 = [r for r, k in zip(reg, keep_t) if k]
    mat2 = [m_ for m_, k in zip(mat, keep_t) if k]
    tri_map = np.where(keep_t, np.cumsum(keep_t) - 1, -1)
    V2, T2, vid = _drop_vertex(V, T2, b)
    vert_map = vid.copy()
    vert_map[b] = vid[a]
    att = None
    if d.get("landmark_attach"):
        fan = np.flatnonzero((T2 == vid[a]).any(1))
        att = {}
        for k, at in d["landmark_attach"].items():
            if "vertex" in at:
                att[k] = {"vertex": int(vert_map[int(at["vertex"])])}
            elif tri_map[int(at["tri"])] >= 0:
                att[k] = {"tri": int(tri_map[int(at["tri"])]), "bary": list(at["bary"])}
            else:
                att[k] = meshio.attach_point(old_pos[k], V2, T2, fan if len(fan) else None)
    return _pack(d, V2, T2, reg2, mat2, att)


def weld_vertices(d: dict, pairs=None, tol: float = 1e-6) -> dict:
    """Merge vertices.  pairs = [(keep, drop), ...] (drop is merged into keep, keep's
    position is kept), or None = merge every group of positions equal within `tol`.
    Triangles that become degenerate (repeated index) are removed; unused vertices are
    dropped.  The returned dict has 'weld_report' {'merged', 'removed_triangles'} in notes."""
    V, T, reg, mat = _unpack(d)
    n = len(V)
    target = np.arange(n)
    if pairs is None:
        from .geom import weld_ids
        wid = weld_ids(V, tol)
        first = {}
        for i, w in enumerate(wid):
            first.setdefault(int(w), i)
        target = np.array([first[int(w)] for w in wid])
    else:
        for keep, drop in pairs:
            target[int(drop)] = int(keep)
        for i in range(n):                                    # resolve chains
            j = i
            while target[j] != j:
                j = target[j]
            target[i] = j
    T = target[T]
    ok = (T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])
    tri_map = np.where(ok, np.cumsum(ok) - 1, -1)
    T = T[ok]
    reg = [r for r, k in zip(reg, ok) if k]
    mat = [m_ for m_, k in zip(mat, ok) if k]
    used = np.zeros(n, bool)
    used[np.unique(T)] = True
    new_id = np.cumsum(used) - 1
    V2 = V[used]
    T2 = new_id[T]
    vert_map = new_id[target]
    att = None
    if d.get("landmark_attach"):
        old_pos = _landmark_positions(d)
        att = {}
        for k, at in d["landmark_attach"].items():
            if "vertex" in at:
                att[k] = {"vertex": int(vert_map[int(at["vertex"])])}
            elif tri_map[int(at["tri"])] >= 0:
                att[k] = {"tri": int(tri_map[int(at["tri"])]), "bary": list(at["bary"])}
            else:
                att[k] = meshio.attach_point(old_pos[k], V2, T2)
    out = _pack(d, V2, T2, reg, mat, att)
    out["notes"] = (d.get("notes", "") + f" [weld: merged {int((target != np.arange(n)).sum())} "
                    f"vertices, removed {int((~ok).sum())} triangles]").strip()
    return out


def _align_ring(A, B, Va, Vb):
    """Rotate/reverse ring B so that it runs alongside ring A (same sense, close start)."""
    best = None
    for rev in (False, True):
        b = B[::-1] if rev else B
        for s in range(len(b)):
            bb = np.roll(b, -s)
            # compare at matching normalised parameters
            ia = np.linspace(0, len(A), 16, endpoint=False).astype(int) % len(A)
            ib = np.linspace(0, len(bb), 16, endpoint=False).astype(int) % len(bb)
            cost = np.linalg.norm(Va[A[ia]] - Vb[bb[ib]], axis=1).sum()
            if best is None or cost < best[0]:
                best = (cost, bb)
    return best[1]


def bridge_rings(d: dict, idx_a, idx_b, region=None, material=None, align: bool = True) -> dict:
    """Connect two closed vertex loops with a band of triangles.  Equal sizes n: 2n
    triangles (quads split along the shorter diagonal); unequal sizes n, m: n + m
    triangles by a shortest-diagonal zipper.  Ring B is rotated/reversed to run alongside
    ring A (align=True).  Winding follows the neighbouring triangles when the rings are
    open boundaries of existing geometry (else call orient_outward afterwards).  Labels
    default to those of a triangle adjacent to ring A."""
    V, T, reg, mat = _unpack(d)
    A = np.asarray(idx_a, np.int64)
    B = np.asarray(idx_b, np.int64)
    if align:
        B = _align_ring(A, B, V, V)
    n, m = len(A), len(B)
    new = []
    if n == m:
        for i in range(n):
            a0, a1, b0, b1 = A[i], A[(i + 1) % n], B[i], B[(i + 1) % n]
            if np.linalg.norm(V[a0] - V[b1]) <= np.linalg.norm(V[a1] - V[b0]):
                new += [[a0, a1, b1], [a0, b1, b0]]
            else:
                new += [[a0, a1, b0], [a1, b1, b0]]
    else:
        i = j = 0
        while i < n or j < m:
            ai, ai1 = A[i % n], A[(i + 1) % n]
            bj, bj1 = B[j % m], B[(j + 1) % m]
            if i >= n:
                adv_a = False
            elif j >= m:
                adv_a = True
            else:
                adv_a = np.linalg.norm(V[ai1] - V[bj]) <= np.linalg.norm(V[ai] - V[bj1])
            if adv_a:
                new.append([ai, ai1, bj]); i += 1
            else:
                new.append([ai, bj1, bj]); j += 1
    new = _orient_like_neighbours(T, [[int(x) for x in t] for t in new])
    r, mt = _labels_near(T, reg, mat, A, region, material)
    T = np.vstack([T, np.asarray(new, np.int64)]) if len(T) else np.asarray(new, np.int64)
    reg += [r] * len(new); mat += [mt] * len(new)
    att = _remap_attach(d, V, T)
    return _pack(d, V, T, reg, mat, att)


def cap_fan(d: dict, ring, region=None, material=None) -> dict:
    """Close a vertex loop with a fan from ring[0] (len(ring) - 2 triangles, no new
    vertex).  Suitable for small, roughly planar convex loops."""
    V, T, reg, mat = _unpack(d)
    R = [int(x) for x in ring]
    new = [[R[0], R[k], R[k + 1]] for k in range(1, len(R) - 1)]
    new = _orient_like_neighbours(T, new)
    r, mt = _labels_near(T, reg, mat, R, region, material)
    T = np.vstack([T, np.asarray(new, np.int64)])
    reg += [r] * len(new); mat += [mt] * len(new)
    return _pack(d, V, T, reg, mat, _remap_attach(d, V, T))


def cap_tip(d: dict, ring, tip=None, region=None, material=None) -> dict:
    """Close a vertex loop with a cone to a new apex vertex (len(ring) triangles).  tip =
    apex position (default: ring centroid).  The apex is the last vertex."""
    V, T, reg, mat = _unpack(d)
    R = [int(x) for x in ring]
    p = np.asarray(tip, float) if tip is not None else V[R].mean(0)
    k = len(V)
    V = np.vstack([V, p])
    new = [[R[i], R[(i + 1) % len(R)], k] for i in range(len(R))]
    new = _orient_like_neighbours(T, new)
    r, mt = _labels_near(T, reg, mat, R, region, material)
    T = np.vstack([T, np.asarray(new, np.int64)])
    reg += [r] * len(new); mat += [mt] * len(new)
    return _pack(d, V, T, reg, mat, _remap_attach(d, V, T))


def orient_outward(d: dict) -> dict:
    """Make winding consistent inside every edge-connected component (BFS over manifold
    edges), then flip any component whose signed volume is negative.  Labels and
    attachments are unchanged (a flipped triangle keeps its index; barycentric weights
    are permuted accordingly)."""
    V, T, reg, mat = _unpack(d)
    nT = len(T)
    edge_tris = defaultdict(list)
    for i, t in enumerate(T):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_tris[(min(a, b), max(a, b))].append(i)
    flipped = np.zeros(nT, bool)
    seen = np.zeros(nT, bool)
    comp = np.full(nT, -1)
    nc = 0
    for s in range(nT):
        if seen[s]:
            continue
        seen[s] = True
        comp[s] = nc
        q = deque([s])
        while q:
            i = q.popleft()
            t = T[i]
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                for j in edge_tris[(min(a, b), max(a, b))]:
                    if seen[j]:
                        continue
                    # neighbour must traverse the shared edge as b->a
                    tj = T[j]
                    same_dir = any((tj[k], tj[(k + 1) % 3]) == (a, b) for k in range(3))
                    if same_dir:
                        T[j] = tj[[0, 2, 1]]
                        flipped[j] = ~flipped[j]
                    seen[j] = True
                    comp[j] = nc
                    q.append(j)
        nc += 1
    for c in range(nc):
        ids = np.flatnonzero(comp == c)
        cen = V[np.unique(T[ids])].mean(0)
        A, B, C = V[T[ids, 0]] - cen, V[T[ids, 1]] - cen, V[T[ids, 2]] - cen
        if (A * np.cross(B, C)).sum() < 0:
            T[ids] = T[ids][:, [0, 2, 1]]
            flipped[ids] = ~flipped[ids]
    att = None
    if d.get("landmark_attach"):
        att = {}
        for k, a in d["landmark_attach"].items():
            if "tri" in a and flipped[int(a["tri"])]:
                b = a["bary"]
                att[k] = {"tri": int(a["tri"]), "bary": [b[0], b[2], b[1]]}
            else:
                att[k] = copy.deepcopy(a)
    return _pack(d, V, T, reg, mat, att)
