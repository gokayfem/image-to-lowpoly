"""Generalised-cylinder cat skeleton, format ``catskel-1`` (CONTRACT §11).

numpy-only; importable inside Blender's bundled Python and in the analysis venv.

File layout (``proxy/skeleton.json``)::

    {"format": "catskel-1", "units": "m", "frame": "...",
     "parts": {name: {"region": "torso", "caps": {"start": "round", "end": "round"},
                      "cap_depth": {"start": d0, "end": d1},          # optional, metres
                      "rings": [{"c":[x,y,z], "u":[..], "v":[..], "ru":r, "rv":r, "p":2.0}, ...]}},
     ...}

A ring point at angle theta is ``c + ru*sgn(cos)|cos|^(2/p)*u + rv*sgn(sin)|sin|^(2/p)*v``.
``u x v`` is the ring-plane normal; rings of a part are ordered from ``t = 0`` to ``t = 1``.

Interpolation between authored rings (what every budget worker resamples):
  * centre: centripetal Catmull-Rom (Barry-Goldman) through the ring centres;
  * u, v: linear in the knot parameter, then re-orthonormalised (n = u x v, v = n x u);
  * ru, rv: uniform Catmull-Rom in the knot parameter, clamped to [0.5*min, max] of the
    bracketing rings (smooth, crease-free); p: linear in the knot parameter;
  * ``t`` is the normalised arc length of the centre curve.

Caps: ``"round"`` = elliptical dome along the end ring's normal (depth = cap_depth, default
mean(ru, rv) of that ring), ``"flat"`` = fan to the ring centre, ``"none"`` = open.
``tube_mesh`` returns triangles counter-clockwise seen from outside (checked by signed volume).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import paths as _paths

FORMAT = "catskel-1"
PART_ORDER = ("torso", "neck", "head", "ear_R", "ear_L", "leg_FR", "leg_FL", "leg_HR",
              "leg_HL", "tail", "muzzle")
DEFAULT_PATH = _paths.PROXY / "skeleton.json"
_DENSE = 48            # samples per knot segment for the arc-length table


def _cr_knots(P: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    d = np.maximum(d, 1e-9) ** alpha
    return np.concatenate([[0.0], np.cumsum(d)])


def _catmull_rom(P: np.ndarray, s: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Centripetal Catmull-Rom through points P (K,3) evaluated at knot-index parameter
    s in [0, K-1] (segment i = [i, i+1]).  Phantom end points are reflected."""
    K = len(P)
    s = np.clip(np.asarray(s, float), 0.0, K - 1)
    if K == 1:
        return np.repeat(P, len(s), 0)
    if K == 2:
        f = s[:, None]
        return P[0] * (1 - f) + P[1] * f
    Q = np.concatenate([2 * P[:1] - P[1:2], P, 2 * P[-1:] - P[-2:-1]])      # K+2
    kn = _cr_knots(Q, alpha)
    i = np.minimum(np.floor(s).astype(int), K - 2)
    f = (s - i)[:, None]
    p0, p1, p2, p3 = Q[i], Q[i + 1], Q[i + 2], Q[i + 3]
    t0, t1, t2, t3 = (kn[i + k][:, None] for k in range(4))
    t = t1 + f * (t2 - t1)

    def lerp(a, b, ta, tb):
        den = np.where(np.abs(tb - ta) < 1e-12, 1e-12, tb - ta)
        return a * ((tb - t) / den) + b * ((t - ta) / den)

    a1 = lerp(p0, p1, t0, t1)
    a2 = lerp(p1, p2, t1, t2)
    a3 = lerp(p2, p3, t2, t3)
    b1 = lerp(a1, a2, t0, t2)
    b2 = lerp(a2, a3, t1, t3)
    return lerp(b1, b2, t1, t2)


def _lin(A: np.ndarray, s: np.ndarray) -> np.ndarray:
    K = len(A)
    s = np.clip(np.asarray(s, float), 0.0, K - 1)
    i = np.minimum(np.floor(s).astype(int), max(K - 2, 0))
    f = s - i
    if K == 1:
        return np.repeat(A[:1], len(s), 0)
    f = f.reshape((-1,) + (1,) * (A.ndim - 1))
    return A[i] * (1 - f) + A[i + 1] * f


def _smooth_scalar(A: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Uniform Catmull-Rom of a positive scalar sequence at knot parameter s, clamped
    between 0.5*min and max of the bracketing values (no overshoot, no negatives)."""
    K = len(A)
    s = np.clip(np.asarray(s, float), 0.0, K - 1)
    if K < 3:
        return _lin(A, s)
    Q = np.concatenate([[2 * A[0] - A[1]], A, [2 * A[-1] - A[-2]]])
    i = np.minimum(np.floor(s).astype(int), K - 2)
    f = s - i
    p0, p1, p2, p3 = Q[i], Q[i + 1], Q[i + 2], Q[i + 3]
    out = 0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f ** 2
                 + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)
    lo = 0.5 * np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    return np.clip(out, lo, hi)


def superellipse_unit(n: int, p, phase: float = 0.0):
    """(cos-like, sin-like) coefficients of n ring points; p scalar or (M,) -> (M,n)."""
    th = phase + 2 * np.pi * np.arange(n) / n
    c, s = np.cos(th), np.sin(th)
    p = np.atleast_1d(np.asarray(p, float))[:, None]
    e = 2.0 / p
    return np.sign(c) * np.abs(c) ** e, np.sign(s) * np.abs(s) ** e


class Part:
    """One generalised cylinder.  Arrays: c, u, v (K,3); ru, rv, p (K,)."""

    def __init__(self, name: str, rings, caps=None, cap_depth=None, region: str | None = None,
                 meta: dict | None = None):
        self.name = name
        if isinstance(rings, dict):                       # arrays form
            self.c = np.asarray(rings["c"], float)
            self.u = np.asarray(rings["u"], float)
            self.v = np.asarray(rings["v"], float)
            self.ru = np.asarray(rings["ru"], float)
            self.rv = np.asarray(rings["rv"], float)
            self.p = np.asarray(rings.get("p", np.full(len(self.c), 2.0)), float)
        else:
            self.c = np.array([r["c"] for r in rings], float)
            self.u = np.array([r["u"] for r in rings], float)
            self.v = np.array([r["v"] for r in rings], float)
            self.ru = np.array([r["ru"] for r in rings], float)
            self.rv = np.array([r["rv"] for r in rings], float)
            self.p = np.array([r.get("p", 2.0) for r in rings], float)
        caps = caps or {}
        self.caps = {"start": caps.get("start", "round"), "end": caps.get("end", "round")}
        cd = cap_depth or {}
        self.cap_depth = {"start": cd.get("start"), "end": cd.get("end")}
        self.region = region or name
        self.meta = meta or {}
        self._build_arclength()

    # ---- parameterisation ------------------------------------------------------------
    @property
    def n_rings(self) -> int:
        return len(self.c)

    def _build_arclength(self):
        K = self.n_rings
        if K < 2:
            self._s_tab = np.zeros(1)
            self._t_tab = np.zeros(1)
            self.length = 0.0
            return
        s = np.linspace(0, K - 1, (K - 1) * _DENSE + 1)
        pts = _catmull_rom(self.c, s)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        L = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(L[-1])
        self._s_tab = s
        self._t_tab = L / max(L[-1], 1e-12)

    def s_of_t(self, t) -> np.ndarray:
        t = np.clip(np.atleast_1d(np.asarray(t, float)), 0.0, 1.0)
        if self.n_rings < 2:
            return np.zeros_like(t)
        return np.interp(t, self._t_tab, self._s_tab)

    def t_of_ring(self) -> np.ndarray:
        """Arc-length parameter t of each authored ring."""
        if self.n_rings < 2:
            return np.zeros(1)
        return np.interp(np.arange(self.n_rings, dtype=float), self._s_tab, self._t_tab)

    def center(self, t) -> np.ndarray:
        scalar = np.ndim(t) == 0
        out = _catmull_rom(self.c, self.s_of_t(t))
        return out[0] if scalar else out

    def frame(self, t):
        """-> (u, v, n) each (N,3) (or (3,) for scalar t); n = u x v."""
        scalar = np.ndim(t) == 0
        s = self.s_of_t(t)
        u = _lin(self.u, s)
        v = _lin(self.v, s)
        n = np.cross(u, v)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        u = u - (u * n).sum(1, keepdims=True) * n
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        v = np.cross(n, u)
        if scalar:
            return u[0], v[0], n[0]
        return u, v, n

    def radii(self, t):
        """-> (ru, rv, p) arrays (or scalars for scalar t).  ru, rv: uniform Catmull-Rom in
        the knot index (smooth, no creases at authored rings), clamped to
        [0.5*min, 1.0*max] of the two neighbouring rings; p: linear."""
        scalar = np.ndim(t) == 0
        s = self.s_of_t(t)
        ru, rv = _smooth_scalar(self.ru, s), _smooth_scalar(self.rv, s)
        p = _lin(self.p, s)
        if scalar:
            return float(ru[0]), float(rv[0]), float(p[0])
        return ru, rv, p

    def rings(self, ts, n: int, phase: float = 0.0) -> np.ndarray:
        """(len(ts), n, 3) ring points."""
        ts = np.atleast_1d(np.asarray(ts, float))
        c = self.center(ts)
        u, v, _ = self.frame(ts)
        ru, rv, p = self.radii(ts)
        a, b = superellipse_unit(n, p, phase)                # (M,n)
        return (c[:, None, :] + (ru[:, None] * a)[..., None] * u[:, None, :]
                + (rv[:, None] * b)[..., None] * v[:, None, :])

    def ring(self, t, n: int, phase: float = 0.0) -> np.ndarray:
        return self.rings([float(t)], n, phase)[0]

    def tangent(self, t) -> np.ndarray:
        scalar = np.ndim(t) == 0
        tt = np.atleast_1d(np.asarray(t, float))
        eps = 1e-3
        a = self.center(np.clip(tt - eps, 0, 1))
        b = self.center(np.clip(tt + eps, 0, 1))
        d = b - a
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        return d[0] if scalar else d

    def end_cap_depth(self, which: str) -> float:
        d = self.cap_depth.get(which)
        if d is not None:
            return float(d)
        k = 0 if which == "start" else -1
        return float(0.5 * (self.ru[k] + self.rv[k]))

    def to_dict(self) -> dict:
        rings = []
        for k in range(self.n_rings):
            rings.append({"c": [round(float(x), 6) for x in self.c[k]],
                          "u": [round(float(x), 6) for x in self.u[k]],
                          "v": [round(float(x), 6) for x in self.v[k]],
                          "ru": round(float(self.ru[k]), 6), "rv": round(float(self.rv[k]), 6),
                          "p": round(float(self.p[k]), 4)})
        d = {"region": self.region, "caps": dict(self.caps), "rings": rings}
        if any(v is not None for v in self.cap_depth.values()):
            d["cap_depth"] = {k: (None if v is None else round(float(v), 6))
                              for k, v in self.cap_depth.items()}
        if self.meta:
            d["meta"] = self.meta
        return d


class Skeleton:
    def __init__(self, parts: dict[str, Part], meta: dict | None = None):
        self.parts = parts
        self.meta = meta or {}

    def __getitem__(self, k) -> Part:
        return self.parts[k]

    def __iter__(self):
        return iter(self.parts)

    def items(self):
        return self.parts.items()

    def to_dict(self) -> dict:
        d = {"format": FORMAT, "units": "m",
             "frame": "CONTRACT §2: nose +X, cat's left +Y, up +Z, ground z=0",
             "ring_formula": "c + ru*sgn(cos t)|cos t|^(2/p)*u + rv*sgn(sin t)|sin t|^(2/p)*v",
             "interpolation": "centre: centripetal Catmull-Rom; ru,rv: uniform Catmull-Rom in knot "
                              "index clamped to [0.5*min,max] of the bracketing rings; u,v,p: linear "
                              "in knot index (u,v re-orthonormalised); t = normalised arc length",
             "parts": {k: p.to_dict() for k, p in self.parts.items()}}
        d.update({k: v for k, v in self.meta.items() if k not in d})
        return d


def skeleton_from_dict(d: dict) -> Skeleton:
    if d.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} file: format={d.get('format')!r}")
    parts = {}
    for name, pd in d["parts"].items():
        parts[name] = Part(name, pd["rings"], pd.get("caps"), pd.get("cap_depth"),
                           pd.get("region"), pd.get("meta"))
    meta = {k: v for k, v in d.items() if k not in ("format", "parts")}
    return Skeleton(parts, meta)


def load_skeleton(path: Path | str | None = None) -> Skeleton:
    return skeleton_from_dict(json.loads(Path(path or DEFAULT_PATH).read_text()))


def save_skeleton(skel: Skeleton, path: Path | str | None = None) -> Path:
    path = Path(path or DEFAULT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(skel.to_dict(), indent=1))
    return path


# ---- meshing -----------------------------------------------------------------------------

def _cap(part: Part, which: str, kind: str, ring_pts: np.ndarray, base: int, n: int,
         dome_rings: int):
    """Vertices/triangles closing one end.  ring_pts (n,3) = the end ring (indices base..)."""
    t = 0.0 if which == "start" else 1.0
    c = part.center(t)
    _, _, nrm = part.frame(t)
    tang = part.tangent(t)
    sign = -1.0 if which == "start" else 1.0
    axis = nrm if np.dot(nrm, tang) >= 0 else -nrm
    axis = sign * axis
    V, T = [], []
    if kind == "none":
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    prev = np.arange(base, base + n)
    nxt_id = base + n
    rel = ring_pts - c
    if kind == "round":
        depth = part.end_cap_depth(which)
        for k in range(1, dome_rings + 1):
            phi = 0.5 * np.pi * k / (dome_rings + 1)
            pts = c + rel * np.cos(phi) + axis * depth * np.sin(phi)
            V.append(pts)
            cur = np.arange(nxt_id, nxt_id + n)
            nxt_id += n
            for j in range(n):
                j1 = (j + 1) % n
                T.append((prev[j], prev[j1], cur[j1]))
                T.append((prev[j], cur[j1], cur[j]))
            prev = cur
        apex = c + axis * depth
    else:
        apex = c
    V.append(apex[None])
    for j in range(n):
        T.append((prev[j], prev[(j + 1) % n], nxt_id))
    return np.concatenate(V), np.asarray(T, np.int64)


def tube_mesh(part: Part, ts, n: int, cap_start: str | None = None, cap_end: str | None = None,
              phase: float = 0.0, dome_rings: int = 3):
    """Closed (unless a cap is "none") triangle mesh of a part sampled at rings ``ts``
    (normalised arc length, increasing) with ``n`` points per ring.  Returns (V, T) with
    outward (CCW) winding."""
    ts = np.asarray(ts, float)
    R = part.rings(ts, n, phase)                        # (M,n,3)
    M = len(ts)
    V = [R.reshape(-1, 3)]
    T = []
    j = np.arange(n)
    j1 = (j + 1) % n
    for i in range(M - 1):
        a = i * n + j
        b = i * n + j1
        c = (i + 1) * n + j1
        d = (i + 1) * n + j
        T.append(np.stack([a, b, c], 1))
        T.append(np.stack([a, c, d], 1))
    T = [np.concatenate(T)] if T else []
    nv = M * n
    cs = cap_start or part.caps["start"]
    ce = cap_end or part.caps["end"]
    # start cap: built with reversed orientation, so flip its triangles afterwards
    Vs, Ts = _cap(part, "start", cs, R[0], 0, n, dome_rings)
    if len(Ts):
        Ts = Ts.copy()
        Ts[Ts >= n] += nv - n                            # new verts appended after rings
        T.append(Ts[:, ::-1])
        V.append(Vs)
        nv += len(Vs)
    Ve, Te = _cap(part, "end", ce, R[-1], (M - 1) * n, n, dome_rings)
    if len(Te):
        Te = Te.copy()
        Te[Te >= M * n] += nv - M * n
        T.append(Te)
        V.append(Ve)
    V = np.concatenate(V)
    T = np.concatenate(T).astype(np.int64)
    # orientation check: signed volume must be positive for outward winding
    vol = (V[T[:, 0]] * np.cross(V[T[:, 1]], V[T[:, 2]])).sum()
    if vol < 0:
        T = T[:, ::-1].copy()
    return V, T


def default_ts(part: Part, per_segment: int = 4) -> np.ndarray:
    """Authored rings plus `per_segment - 1` arc-length samples between them."""
    tr = part.t_of_ring()
    out = []
    for a, b in zip(tr[:-1], tr[1:]):
        out.extend(np.linspace(a, b, per_segment, endpoint=False).tolist())
    out.append(1.0)
    return np.asarray(out)


def skeleton_mesh(skel: Skeleton, sides=16, per_segment=4, parts=None):
    """Concatenated unfused part tubes -> (V, T, part_index per triangle, names)."""
    Vs, Ts, lab, names = [], [], [], []
    base = 0
    for k, (name, part) in enumerate(skel.items()):
        if parts and name not in parts:
            continue
        n = sides[name] if isinstance(sides, dict) else sides
        ps = per_segment[name] if isinstance(per_segment, dict) else per_segment
        V, T = tube_mesh(part, default_ts(part, ps), n)
        Vs.append(V)
        Ts.append(T + base)
        lab.append(np.full(len(T), len(names)))
        names.append(name)
        base += len(V)
    return np.concatenate(Vs), np.concatenate(Ts), np.concatenate(lab), names
