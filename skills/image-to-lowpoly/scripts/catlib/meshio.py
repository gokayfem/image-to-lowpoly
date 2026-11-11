"""catmesh-1 load / save / validate (CONTRACT §5).  numpy-only; importable inside Blender."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FORMAT = "catmesh-1"
REGIONS = ("head", "ears", "eyes", "nose", "neck", "torso",
           "leg_FR", "leg_FL", "leg_HR", "leg_HL", "tail")
MATERIALS = ("fur_orange", "fur_cream", "fur_stripe", "ear_inner", "eye", "pupil", "nose")
SHARP_LANDMARKS = ("ear_tip_R", "ear_tip_L", "eye_R", "eye_L", "nose", "paw_FR", "paw_FL",
                   "paw_HR", "paw_HL", "tail_base", "tail_mid", "tail_tip")
FUZZY_LANDMARKS = ("shoulder_R", "hip_R", "hock_HR", "chin")
LANDMARKS = SHARP_LANDMARKS + FUZZY_LANDMARKS
BUDGETS = (100, 200, 500, 1000)


class CatMesh:
    """Arrays view of a mesh.json.  Attributes: budget, name, vertices (V,3) float64,
    triangles (T,3) int64, tri_region (list[str]), tri_material (list[str]),
    landmarks {name: (3,) float64}, notes, raw (the parsed dict), path."""

    def __init__(self, d: dict, path: Path | None = None):
        self.raw = d
        self.path = path
        self.budget = d.get("budget")
        self.name = d.get("name")
        self.vertices = np.asarray(d.get("vertices", []), dtype=np.float64).reshape(-1, 3)
        self.triangles = np.asarray(d.get("triangles", []), dtype=np.int64)
        self.tri_region = list(d.get("tri_region", []))
        self.tri_material = list(d.get("tri_material", []))
        self.landmarks = {k: np.asarray(v, float) for k, v in d.get("landmarks", {}).items()}
        self.notes = d.get("notes", "")
        # CONTRACT §12: attached landmarks are derived from geometry, never read as stored.
        self.landmark_attach = d.get("landmark_attach") or None
        if self.landmark_attach and len(self.vertices) and len(self.triangles):
            self.landmarks.update(landmarks_from_attach(self.vertices, self.triangles,
                                                        self.landmark_attach))

    @property
    def n_triangles(self) -> int:
        return int(len(self.triangles))

    def region_ids(self) -> np.ndarray:
        return np.array([REGIONS.index(r) if r in REGIONS else -1 for r in self.tri_region], np.int32)

    def tri_vertices(self) -> np.ndarray:
        return self.vertices[self.triangles]


def load(path: Path | str) -> CatMesh:
    path = Path(path)
    return CatMesh(json.loads(path.read_text()), path)


# ---- landmark attachment (CONTRACT §12) ----------------------------------------------------

def landmarks_from_attach(vertices, triangles, attach: dict) -> dict:
    """{name: (3,)} from {name: {"tri": i, "bary": [a,b,c]} | {"vertex": i}}."""
    V = np.asarray(vertices, float).reshape(-1, 3)
    T = np.asarray(triangles, np.int64).reshape(-1, 3)
    out = {}
    for k, a in attach.items():
        if "vertex" in a:
            out[k] = V[int(a["vertex"])].copy()
        else:
            b = np.asarray(a["bary"], float)
            out[k] = b @ V[T[int(a["tri"])]]
    return out


def _closest_bary(p, A, B, C):
    """Closest point on each triangle (A,B,C (N,3)) to p, as barycentric coords (N,3)."""
    from .geom import closest_point_triangles
    q = closest_point_triangles(p, A, B, C)
    v0, v1, v2 = B - A, C - A, q - A
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    b1 = (d11 * d20 - d01 * d21) / den
    b2 = (d00 * d21 - d01 * d20) / den
    bary = np.stack([1 - b1 - b2, b1, b2], 1)
    return q, np.clip(bary, 0.0, 1.0) / np.clip(bary, 0.0, 1.0).sum(1, keepdims=True)


def attach_point(p, vertices, triangles, tri_ids=None) -> dict:
    """Attachment {"tri", "bary"} of the closest surface point to p (optionally restricted
    to the triangles `tri_ids`)."""
    V = np.asarray(vertices, float).reshape(-1, 3)
    T = np.asarray(triangles, np.int64).reshape(-1, 3)
    ids = np.arange(len(T)) if tri_ids is None else np.asarray(tri_ids, np.int64)
    A, B, C = V[T[ids, 0]], V[T[ids, 1]], V[T[ids, 2]]
    q, bary = _closest_bary(np.asarray(p, float), A, B, C)
    j = int(np.argmin(np.linalg.norm(q - np.asarray(p, float), axis=1)))
    return {"tri": int(ids[j]), "bary": [float(x) for x in bary[j]]}


def attach_landmarks(mesh, names=None, vertex_tol: float = 1e-7) -> dict:
    """Derive `landmark_attach` from the existing landmark xyz (closest triangle +
    barycentric; a landmark coinciding with a vertex attaches to that vertex).
    `mesh` = CatMesh or catmesh-1 dict.  A dict is updated in place (landmark_attach set,
    landmarks snapped to the attached points); a CatMesh gets .landmark_attach and
    .landmarks updated.  Returns the attach dict."""
    if isinstance(mesh, CatMesh):
        V, T, lm = mesh.vertices, mesh.triangles, mesh.landmarks
    else:
        V = np.asarray(mesh["vertices"], float)
        T = np.asarray(mesh["triangles"], np.int64)
        lm = {k: np.asarray(v, float) for k, v in mesh["landmarks"].items()}
    att = {}
    for k in (names or [n for n in LANDMARKS if n in lm] + [n for n in lm if n not in LANDMARKS]):
        p = np.asarray(lm[k], float)
        dv = np.linalg.norm(V - p, axis=1)
        j = int(np.argmin(dv))
        if dv[j] <= vertex_tol:
            att[k] = {"vertex": j}
        else:
            att[k] = attach_point(p, V, T)
    new = landmarks_from_attach(V, T, att)
    if isinstance(mesh, CatMesh):
        mesh.landmark_attach = {**(mesh.landmark_attach or {}), **att}
        mesh.landmarks.update(new)
        mesh.raw["landmark_attach"] = mesh.landmark_attach
    else:
        mesh["landmark_attach"] = {**(mesh.get("landmark_attach") or {}), **att}
        mesh["landmarks"].update({k: [float(c) for c in v] for k, v in new.items()})
    return att


def sync_landmarks(d: dict) -> dict:
    """Recompute d['landmarks'] from d['landmark_attach'] in place (no-op without one)."""
    att = d.get("landmark_attach")
    if att:
        new = landmarks_from_attach(d["vertices"], d["triangles"], att)
        d.setdefault("landmarks", {}).update({k: [float(c) for c in v] for k, v in new.items()})
    return d


def to_dict(vertices, triangles, tri_region, tri_material, landmarks, budget, name=None,
            notes: str = "", landmark_attach: dict | None = None) -> dict:
    d = {
        "format": FORMAT,
        "budget": int(budget),
        "name": name or f"cat_{int(budget)}",
        "vertices": [[float(c) for c in v] for v in np.asarray(vertices, float)],
        "triangles": [[int(c) for c in t] for t in np.asarray(triangles, int)],
        "tri_region": [str(r) for r in tri_region],
        "tri_material": [str(m) for m in tri_material],
        "landmarks": {k: [float(c) for c in np.asarray(v, float)] for k, v in landmarks.items()},
        "notes": notes,
    }
    if landmark_attach:
        d["landmark_attach"] = landmark_attach
        sync_landmarks(d)
    return d


_CORE_KEYS = ("format", "budget", "name", "vertices", "triangles", "tri_region", "tri_material",
              "landmarks", "notes")


def save(path: Path | str, d: dict) -> Path:
    """Write a mesh.json (compact rows, one vertex/triangle per line).  Validates first;
    schema errors raise ValueError, budget overrun does not (the validator reports it).
    If `landmark_attach` is present, `landmarks` are recomputed from it first (in place).
    Extra top-level keys (landmark_attach, groups, ...) are written after the core keys."""
    sync_landmarks(d)
    errs = [e for e in validate_dict(d) if not e.startswith("over_budget")]
    if errs:
        raise ValueError("invalid catmesh-1: " + "; ".join(errs[:10]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["{"]
    lines.append(f'  "format": {json.dumps(d["format"])},')
    lines.append(f'  "budget": {int(d["budget"])},')
    lines.append(f'  "name": {json.dumps(d["name"])},')
    lines.append('  "vertices": [\n' + ",\n".join(
        "    [" + ", ".join(f"{c:.7f}" for c in v) + "]" for v in d["vertices"]) + "\n  ],")
    lines.append('  "triangles": [\n' + ",\n".join(
        "    [" + ", ".join(str(int(c)) for c in t) + "]" for t in d["triangles"]) + "\n  ],")
    lines.append(f'  "tri_region": {json.dumps(d["tri_region"])},')
    lines.append(f'  "tri_material": {json.dumps(d["tri_material"])},')
    lines.append('  "landmarks": {\n' + ",\n".join(
        f'    {json.dumps(k)}: [' + ", ".join(f"{c:.7f}" for c in v) + "]"
        for k, v in d["landmarks"].items()) + "\n  },")
    extra = [k for k in d if k not in _CORE_KEYS]
    lines.append(f'  "notes": {json.dumps(d.get("notes", ""))}' + ("," if extra else ""))
    for i, k in enumerate(extra):
        v = d[k]
        if k == "landmark_attach":
            body = "{\n" + ",\n".join(f"    {json.dumps(n)}: {json.dumps(a)}" for n, a in v.items()) + "\n  }"
        else:
            body = json.dumps(v)
        lines.append(f"  {json.dumps(k)}: {body}" + ("," if i < len(extra) - 1 else ""))
    lines.append("}")
    path.write_text("\n".join(lines) + "\n")
    return path


def validate_dict(d: dict) -> list[str]:
    """Schema errors of a catmesh-1 dict (empty list = valid).  Every entry is a hard
    failure.  Budget overrun is reported as 'over_budget: ...'."""
    errs: list[str] = []
    if d.get("format") != FORMAT:
        errs.append(f"format: expected {FORMAT!r}, got {d.get('format')!r}")
    b = d.get("budget")
    if not isinstance(b, int) or isinstance(b, bool) or b <= 0:
        errs.append(f"budget: must be a positive int, got {b!r}")
    if not isinstance(d.get("name"), str) or not d.get("name"):
        errs.append("name: missing")
    try:
        V = np.asarray(d.get("vertices"), dtype=np.float64)
        if V.ndim != 2 or V.shape[1] != 3 or len(V) == 0:
            errs.append(f"vertices: expected (V,3), got shape {V.shape}")
        elif not np.isfinite(V).all():
            errs.append("vertices: non-finite values")
    except (TypeError, ValueError) as e:
        errs.append(f"vertices: not a numeric (V,3) array ({e})")
        V = np.zeros((0, 3))
    tris = d.get("triangles")
    T = None
    if not isinstance(tris, list) or not tris:
        errs.append("triangles: missing or empty")
    else:
        bad = [i for i, t in enumerate(tris) if not isinstance(t, (list, tuple)) or len(t) != 3]
        if bad:
            errs.append(f"triangles: {len(bad)} faces are not triangles (e.g. index {bad[0]}: "
                        f"{tris[bad[0]]!r}); only triangles are allowed")
        else:
            try:
                T = np.asarray(tris, dtype=np.int64)
                if np.any(np.asarray(tris, dtype=np.float64) != T):
                    errs.append("triangles: non-integer indices")
            except (TypeError, ValueError):
                errs.append("triangles: non-integer indices")
                T = None
    if T is not None and len(V):
        if T.min() < 0 or T.max() >= len(V):
            errs.append(f"triangles: index out of range [0,{len(V) - 1}] (min {T.min()}, max {T.max()})")
        rep = (T[:, 0] == T[:, 1]) | (T[:, 1] == T[:, 2]) | (T[:, 0] == T[:, 2])
        if rep.any():
            errs.append(f"triangles: {int(rep.sum())} triangles repeat a vertex index "
                        f"(e.g. {int(np.flatnonzero(rep)[0])})")
        if isinstance(b, int) and len(T) > b:
            errs.append(f"over_budget: {len(T)} triangles > budget {b}")
    nt = len(tris) if isinstance(tris, list) else 0
    for key, vocab in (("tri_region", REGIONS), ("tri_material", MATERIALS)):
        lab = d.get(key)
        if not isinstance(lab, list) or len(lab) != nt:
            errs.append(f"{key}: expected {nt} labels, got "
                        f"{len(lab) if isinstance(lab, list) else type(lab).__name__}")
        else:
            unk = sorted({x for x in lab if x not in vocab})
            if unk:
                errs.append(f"{key}: unknown labels {unk}; allowed {list(vocab)}")
    lm = d.get("landmarks")
    if not isinstance(lm, dict):
        errs.append("landmarks: missing")
    else:
        missing = [k for k in LANDMARKS if k not in lm]
        extra = [k for k in lm if k not in LANDMARKS]
        if missing:
            errs.append(f"landmarks: missing {missing}")
        if extra:
            errs.append(f"landmarks: unknown names {extra}")
        for k, v in lm.items():
            a = np.asarray(v, dtype=np.float64) if isinstance(v, (list, tuple)) else None
            if a is None or a.shape != (3,) or not np.isfinite(a).all():
                errs.append(f"landmarks.{k}: expected [x,y,z], got {v!r}")
    att = d.get("landmark_attach")
    if att is not None:
        if not isinstance(att, dict):
            errs.append("landmark_attach: must be a dict {name: {tri,bary}|{vertex}}")
        else:
            nv = len(V)
            for k, a in att.items():
                if k not in LANDMARKS:
                    errs.append(f"landmark_attach: unknown landmark {k!r}")
                if not isinstance(a, dict):
                    errs.append(f"landmark_attach.{k}: not a dict")
                elif "vertex" in a:
                    if not isinstance(a["vertex"], int) or not 0 <= a["vertex"] < nv:
                        errs.append(f"landmark_attach.{k}: vertex {a['vertex']!r} out of range")
                elif "tri" in a and "bary" in a:
                    b = np.asarray(a["bary"], float) if isinstance(a["bary"], (list, tuple)) else None
                    if not isinstance(a["tri"], int) or not 0 <= a["tri"] < nt:
                        errs.append(f"landmark_attach.{k}: tri {a['tri']!r} out of range")
                    if b is None or b.shape != (3,) or not np.isfinite(b).all() \
                            or abs(b.sum() - 1) > 1e-6 or (b < -1e-9).any():
                        errs.append(f"landmark_attach.{k}: bary {a['bary']!r} not a convex weight triple")
                else:
                    errs.append(f"landmark_attach.{k}: needs 'vertex' or 'tri'+'bary'")
    return errs


def validate(path_or_dict) -> list[str]:
    d = json.loads(Path(path_or_dict).read_text()) if not isinstance(path_or_dict, dict) else path_or_dict
    return validate_dict(d)
