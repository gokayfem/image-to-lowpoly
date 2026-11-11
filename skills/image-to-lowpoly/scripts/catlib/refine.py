"""Fixed-connectivity vertex refinement (CONTRACT §12).  CPU only, deterministic per seed
when stopped by a proposal count (--iters); a wall-clock stop (--minutes) is not.

The objective is a SEARCH HEURISTIC ONLY.  Official numbers come from scripts/evaluate.py
(Blender render through the locked camera, CONTRACT §7) and decide what is kept.

Objective  E = w_iou (1 - IoU_ref) + w_bnd B + w_lm L + w_sdf S + w_nv (1 - IoU_nv)
               + w_reg Q + w_sym Y
  IoU_ref  reference-view silhouette (catlib.raster coverage at the working resolution) vs
           reference/derived/mask.png block-downsampled to that resolution.
  B        symmetric mean boundary distance / reference bbox diagonal (same definition as
           the official boundary_mean_px, computed inside a crop window around the cat).
  L        sharp-landmark reprojection error / diag, weighted by 1/sigma_px; landmarks are
           evaluated through `landmark_attach`, so they move with the geometry.
  S        mean |SDF| of the proxy over vertices + face centroids + edge midpoints, in cm.
  IoU_nv   mean silhouette IoU vs the proxy over the fixed novel views at low resolution
           (coverage term: stops parts shrinking onto the proxy surface).
  Q        mean (1 - q) triangle quality, q = 4 sqrt(3) A / sum(l^2) in (0, 1].
  Y        mean mirror-pair mismatch |v_i - mirror_y(v_j)| in cm (optional pairs).
Hard constraints (a move is rejected): z >= 0; vertices with z < 1 mm at the start are
pinned to z = 0 (soles); no affected triangle may newly fall below area 2e-6 m^2 or edge
1.5 mm (or get worse if it already was); no face normal may turn by > 80 deg in one move.

Search: greedy accept-if-better, coarse-to-fine resolutions, rounds of
  (i)  directed silhouette moves of contour vertices along the reference mask's signed
       distance gradient, back-projected at the vertex depth, with step halving;
  (ii) SDF moves towards the proxy surface, preferring the camera depth axis;
  (iii) stochastic per-vertex Gaussian moves with per-vertex adaptive step (1/5 success
       rule) plus rigid translations/rotations of vertex groups;
  (iv) optional edge flips (same-region, same-material pairs, not hosting a landmark).
Incremental evaluation: per-view coverage COUNT images are updated in the bounding box of
the changed triangles only, so a proposal costs O(changed area), not O(mesh).
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import meshio, paths, raster
from .camera import Camera, load_scene, novel_view_cameras

DEFAULT_WEIGHTS = {"iou": 1.0, "bnd": 8.0, "lm": 3.0, "sdf": 0.15, "nv": 0.5, "reg": 0.02, "sym": 0.0}
MIN_AREA = 2e-6
MIN_EDGE = 1.5e-3
MAX_NORMAL_TURN_DEG = 80.0
PIN_Z = 1e-3
SIGMA_MIN, SIGMA_MAX = 3e-4, 3e-2


# ---- data sources (injectable) ------------------------------------------------------------

@dataclass
class Sources:
    derived_dir: Path = paths.DERIVED            # mask.png, landmarks.json, reference_bbox.json
    camera: Path = paths.CAMERA_JSON
    scene: Path = paths.SCENE_JSON
    proxy_dir: Path = paths.ROOT / "proxy"       # proxy_mesh.npz, sdf.npz

    @property
    def mask(self):
        return Path(self.derived_dir) / "mask.png"

    @property
    def landmarks(self):
        return Path(self.derived_dir) / "landmarks.json"

    @property
    def bbox(self):
        return Path(self.derived_dir) / "reference_bbox.json"

    @property
    def proxy_mesh(self):
        return Path(self.proxy_dir) / "proxy_mesh.npz"

    @property
    def proxy_sdf(self):
        return Path(self.proxy_dir) / "sdf.npz"


class Proxy:
    """§11 interface through catlib.proxy, with injectable file paths."""

    def __init__(self, src: Sources):
        from . import proxy as P
        self.P = P
        self.sdf_path = src.proxy_sdf
        self.mesh_path = src.proxy_mesh
        self.ok = self.sdf_path.is_file()
        self.has_mesh = self.mesh_path.is_file()

    def sdf(self, pts):
        return self.P.sdf_lookup(pts, self.sdf_path)

    def grad(self, pts):
        return self.P.sdf_gradient(pts, self.sdf_path)

    def mesh(self):
        return self.P.load_mesh(self.mesh_path)


# ---- small numeric helpers -----------------------------------------------------------------

def _dt(src_zero: np.ndarray, precise: bool = True) -> np.ndarray:
    """Euclidean distance of every pixel to the nearest True pixel of src_zero
    (precise=False: OpenCV's 3x3 chamfer approximation of L2, ~2x faster)."""
    import cv2
    if not src_zero.any():
        return np.full(src_zero.shape, 1e6, np.float32)
    return cv2.distanceTransform((~src_zero).astype(np.uint8), cv2.DIST_L2,
                                 cv2.DIST_MASK_PRECISE if precise else cv2.DIST_MASK_3)


def _boundary(m):
    p = np.pad(m, 1, constant_values=False)
    inner = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~inner


def _downsample(mask: np.ndarray, res: int) -> np.ndarray:
    H, W = mask.shape
    if W % res == 0 and H % res == 0:
        k = W // res
        return mask.reshape(res, k, res, k).mean((1, 3)) >= 0.5
    from PIL import Image
    im = Image.fromarray(mask.astype(np.uint8) * 255).resize((res, res), Image.BOX)
    return np.asarray(im) >= 128


def _bilinear(img, u, v):
    """Sample img at pixel-convention coords (u right, v down; centres at +0.5)."""
    H, W = img.shape
    x = np.clip(np.asarray(u) - 0.5, 0, W - 1.001)
    y = np.clip(np.asarray(v) - 0.5, 0, H - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    fx = x - x0; fy = y - y0
    return ((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x0 + 1]
            + (1 - fx) * fy * img[y0 + 1, x0] + fx * fy * img[y0 + 1, x0 + 1])


def count_raster(P, W, H) -> np.ndarray:
    tri, rows, c0, c1 = raster._spans(P, W, H)
    out = np.zeros((H, W), np.int32)
    if len(rows) == 0:
        return out
    Wp = W + 1
    m = Wp * H
    acc = np.bincount(rows * Wp + c0, minlength=m) - np.bincount(rows * Wp + c1 + 1, minlength=m)
    return np.cumsum(acc.reshape(H, Wp), axis=1)[:, :W].astype(np.int32)


def span_delta(Pold, Pnew, W, H):
    """Coverage-count change from replacing screen triangles Pold by Pnew.
    Returns (r0, c0, delta (h,w) int) or None if nothing is covered."""
    no = len(Pold)
    tri, rows, c0s, c1s = raster._spans(np.concatenate([Pold, Pnew]), W, H)
    if not len(rows):
        return None
    sg = np.where(tri < no, -1.0, 1.0)
    r0, r1 = int(rows.min()), int(rows.max())
    cc0, cc1 = int(c0s.min()), int(c1s.max())
    h, w = r1 - r0 + 1, cc1 - cc0 + 1
    Wp = w + 1
    a = np.bincount((rows - r0) * Wp + (c0s - cc0), weights=sg, minlength=h * Wp) \
        - np.bincount((rows - r0) * Wp + (c1s - cc0 + 1), weights=sg, minlength=h * Wp)
    delta = np.rint(np.cumsum(a.reshape(h, Wp), axis=1)[:, :w]).astype(np.int32)
    return r0, cc0, delta


def tri_quality(V, T):
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    l2 = ((B - A) ** 2).sum(1) + ((C - B) ** 2).sum(1) + ((A - C) ** 2).sum(1)
    return 4 * math.sqrt(3) * area / np.maximum(l2, 1e-30)


def _tri_geom(V, T):
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    n = np.cross(B - A, C - A)
    area = 0.5 * np.linalg.norm(n, axis=1)
    e = np.stack([np.linalg.norm(B - A, axis=1), np.linalg.norm(C - B, axis=1),
                  np.linalg.norm(A - C, axis=1)], 1).min(1)
    return n, area, e


# ---- views -----------------------------------------------------------------------------------

class View:
    """One camera + target mask + coverage count image."""

    def __init__(self, cam: Camera, target: np.ndarray, name: str):
        self.cam, self.name = cam, name
        self.W, self.H = cam.width, cam.height
        self.target = target
        self.t_area = int(target.sum())
        self.counts = None
        self.uv = None
        w2c = cam.world_to_cam
        self._R, self._t = w2c[:3, :3].T.copy(), w2c[:3, 3].copy()

    def project(self, V):
        pc = np.asarray(V, float).reshape(-1, 3) @ self._R + self._t
        dep = -pc[:, 2]
        return np.stack([self.W / 2.0 + self.cam.f_px * pc[:, 0] / dep,
                         self.H / 2.0 - self.cam.f_px * pc[:, 1] / dep], 1)

    def full(self, V, T):
        self.uv = self.project(V)
        self.counts = count_raster(self.uv[T], self.W, self.H)
        m = self.counts > 0
        self.inter = int((m & self.target).sum())
        self.union = int((m | self.target).sum())

    def iou(self, inter=None, union=None):
        i = self.inter if inter is None else inter
        u = self.union if union is None else union
        return i / u if u else 1.0

    def trial(self, T_old_rows, T_new_rows, uv_new_full):
        """uv_new_full: (n,2) uv with changed vertices replaced."""
        d = span_delta(self.uv[T_old_rows], uv_new_full[T_new_rows], self.W, self.H)
        if d is None:
            return None, self.inter, self.union
        r0, c0, delta = d
        h, w = delta.shape
        C = self.counts[r0:r0 + h, c0:c0 + w]
        newC = C + delta
        tg = self.target[r0:r0 + h, c0:c0 + w]
        om, nm = C > 0, newC > 0
        inter = self.inter + int((nm & tg).sum()) - int((om & tg).sum())
        union = self.union + int((nm | tg).sum()) - int((om | tg).sum())
        return (r0, c0, newC), inter, union

    def commit(self, patch, inter, union, uv_new_full):
        if patch is not None:
            r0, c0, newC = patch
            h, w = newC.shape
            self.counts[r0:r0 + h, c0:c0 + w] = newC
        self.inter, self.union = inter, union
        self.uv = uv_new_full


class RefView(View):
    """Reference view with the boundary term and the signed-distance field for directed moves."""

    def __init__(self, cam: Camera, R: np.ndarray, diag_px: float):
        super().__init__(cam, R, "reference")
        self.diag = diag_px
        self.bR = _boundary(R)
        self.dt_bR = _dt(self.bR)
        inside = _dt(~R)          # distance to nearest non-R pixel (for R pixels)
        outside = _dt(R)          # distance to nearest R pixel (for non-R pixels)
        self.sdist = np.where(R, -inside, outside).astype(np.float64)
        gy, gx = np.gradient(self.sdist)
        self.gx, self.gy = gx, gy
        ys, xs = np.nonzero(R)
        pad = int(0.06 * self.W)
        self.crop = (max(ys.min() - pad, 0), min(ys.max() + pad + 1, self.H),
                     max(xs.min() - pad, 0), min(xs.max() + pad + 1, self.W))
        r0, r1, c0, c1 = self.crop
        self.bR_crop = self.bR[r0:r1, c0:c1]
        self.dt_bR_crop = self.dt_bR[r0:r1, c0:c1]
        # reverse direction (R boundary -> M boundary) on a 2x subsampled grid above 512 px:
        # it mainly has to notice missing parts, which needs no sub-pixel precision.
        self.rev_step = 2 if self.W > 512 else 1
        k = self.rev_step
        self.bR_rev = _boundary(R[r0:r1, c0:c1][::k, ::k])

    def boundary_term(self, counts_patch=None):
        r0, r1, c0, c1 = self.crop
        if counts_patch is None:
            M = self.counts[r0:r1, c0:c1] > 0
        else:
            pr, pc, newC = counts_patch
            M = self.counts[r0:r1, c0:c1] > 0
            h, w = newC.shape
            a0, a1 = max(pr, r0), min(pr + h, r1)
            b0, b1 = max(pc, c0), min(pc + w, c1)
            if a0 < a1 and b0 < b1:
                M = M.copy()
                M[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = newC[a0 - pr:a1 - pr, b0 - pc:b1 - pc] > 0
        bM = _boundary(M)
        if not bM.any():
            return 1.0
        fwd = float(self.dt_bR_crop[bM].mean())
        k = self.rev_step
        bMk = bM if k == 1 else _boundary(M[::k, ::k])
        rev = k * float(_dt(bMk, precise=False)[self.bR_rev].mean())
        return 0.5 * (fwd + rev) / self.diag

    def contour_distance(self):
        """Full-resolution distance (crop coordinates) to the current rendered boundary."""
        r0, r1, c0, c1 = self.crop
        bM = _boundary(self.counts[r0:r1, c0:c1] > 0)
        return _dt(bM) if bM.any() else None


# ---- the refiner --------------------------------------------------------------------------------

@dataclass
class Settings:
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    res: tuple = (384, 768)
    nv_res: int = 160
    iters: int | None = 20000
    minutes: float | None = None
    flips: bool = False
    seed: int = 0
    freeze: tuple = ()
    sym_pairs: tuple = ()
    group_frac: float = 0.1
    verbose: bool = True
    snapshot_dir: str | None = None   # CONTRACT §13 (release film); off by default
    snapshot_every: int = 0


class Refiner:
    def __init__(self, d: dict, src: Sources | None = None, settings: Settings | None = None):
        try:
            import cv2
            cv2.setNumThreads(1)      # small images: threading overhead dominates
        except ImportError:
            pass
        self.src = src or Sources()
        self.s = settings or Settings()
        self.w = {**DEFAULT_WEIGHTS, **self.s.weights}
        self.d = json.loads(json.dumps(d))
        self.notes = []
        if not self.d.get("landmark_attach"):
            meshio.attach_landmarks(self.d)
            self.notes.append("landmark_attach derived from input landmark xyz (closest triangle)")
        self.V = np.asarray(self.d["vertices"], float).copy()
        self.T = np.asarray(self.d["triangles"], np.int64).copy()
        self.reg = list(self.d["tri_region"])
        self.mat = list(self.d["tri_material"])
        self.att = self.d["landmark_attach"]
        self.n = len(self.V)
        self.rng = np.random.default_rng(self.s.seed)
        # pins / frozen
        self.pinned = self.V[:, 2] < PIN_Z
        self.V[self.pinned, 2] = 0.0
        self.frozen = np.zeros(self.n, bool)
        if len(self.s.freeze):
            self.frozen[np.asarray(self.s.freeze, int)] = True
        self.sym = np.asarray(self.s.sym_pairs, int).reshape(-1, 2) if len(self.s.sym_pairs) else None
        # reference data
        self.cam_full = Camera.from_json(self.src.camera)
        from .metrics import load_mask, reference_diag
        self.R_full = load_mask(self.src.mask)
        self.diag_full = reference_diag(self.src.bbox, self.R_full)
        lmj = json.loads(Path(self.src.landmarks).read_text())["landmarks"]
        self.lm_names = [n for n in meshio.SHARP_LANDMARKS if n in lmj and n in self.att]
        self.lm_ref = np.array([lmj[n]["uv"] for n in self.lm_names], float)
        sig = np.array([float(lmj[n].get("sigma_px", 4.0)) for n in self.lm_names])
        self.lm_w = (1.0 / np.maximum(sig, 1.0))
        self.lm_w /= self.lm_w.sum()
        self._build_lm_index()
        # proxy
        self.proxy = Proxy(self.src)
        if not self.proxy.ok:
            self.w["sdf"] = 0.0
            self.notes.append(f"proxy SDF missing ({self.src.proxy_sdf}): sdf term disabled")
        if not self.proxy.has_mesh:
            self.w["nv"] = 0.0
            self.notes.append(f"proxy mesh missing ({self.src.proxy_mesh}): novel-view term disabled")
        self.scene = load_scene(self.src.scene)
        self.counters = {}
        self.trace = []
        self.proposals = 0
        self._snap_last = -1
        self.t0 = time.time()
        self._build_topology()
        self.sigma = 0.15 * self._mean_incident_edge()
        self.groups = self._groups()
        self.gsig_t = {k: 0.004 for k in self.groups}
        self.gsig_r = {k: math.radians(3.0) for k in self.groups}

    # ---------------------------------------------------------------- structure
    def _build_topology(self):
        T = self.T
        E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
        E = np.sort(E, 1)
        self.E, inv = np.unique(E, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        nT = len(T)
        self.edge_tris = [[] for _ in range(len(self.E))]
        for k, e in enumerate(inv):
            self.edge_tris[e].append(k % nT)
        self.vt = [[] for _ in range(self.n)]
        for i, t in enumerate(T):
            for v in t:
                self.vt[v].append(i)
        self.vt = [np.array(x, np.int64) for x in self.vt]
        self.ve = [[] for _ in range(self.n)]
        for i, (a, b) in enumerate(self.E):
            self.ve[a].append(i); self.ve[b].append(i)
        self.ve = [np.array(x, np.int64) for x in self.ve]
        self.lm_tris = {int(a["tri"]) for a in self.att.values() if "tri" in a}
        self.lm_verts = {int(a["vertex"]) for a in self.att.values() if "vertex" in a}

    def _build_lm_index(self):
        I = np.zeros((len(self.lm_names), 3), np.int64)
        Wb = np.zeros((len(self.lm_names), 3))
        for k, n in enumerate(self.lm_names):
            a = self.att[n]
            if "vertex" in a:
                I[k] = a["vertex"]; Wb[k] = [1, 0, 0]
            else:
                I[k] = self.T[int(a["tri"])]; Wb[k] = a["bary"]
        self.lm_I, self.lm_B = I, Wb

    def _mean_incident_edge(self):
        L = np.linalg.norm(self.V[self.E[:, 0]] - self.V[self.E[:, 1]], axis=1)
        s = np.zeros(self.n); c = np.zeros(self.n)
        np.add.at(s, self.E[:, 0], L); np.add.at(s, self.E[:, 1], L)
        np.add.at(c, self.E[:, 0], 1); np.add.at(c, self.E[:, 1], 1)
        return s / np.maximum(c, 1)

    def _groups(self):
        g = self.d.get("groups")
        if not g:
            try:
                nj = json.loads(self.d.get("notes", ""))
                g = nj.get("groups") if isinstance(nj, dict) else None
            except (json.JSONDecodeError, TypeError):
                g = None
        if g:
            return {k: np.asarray(v, np.int64) for k, v in g.items() if len(v) >= 2}
        out = {}
        for r in sorted(set(self.reg)):
            if r == "torso":
                continue
            ids = np.unique(self.T[[i for i, x in enumerate(self.reg) if x == r]])
            if len(ids) >= 3:
                out[r] = ids
        return out

    # ---------------------------------------------------------------- per-resolution setup
    def setup(self, res: int):
        self.res = res
        cam = self.cam_full.scaled(res)
        R = _downsample(self.R_full, res)
        self.ref = RefView(cam, R, self.diag_full * res / self.cam_full.width)
        self.views = [self.ref]
        if self.w["nv"] > 0:
            if not hasattr(self, "_nv_targets"):
                PV, PT = self.proxy.mesh()
                cams = novel_view_cameras(self.cam_full, self.scene, size=self.s.nv_res)
                self._nv_targets = {k: (c, raster.silhouette(PV, PT, c)) for k, c in cams.items()}
            for k, (c, tgt) in self._nv_targets.items():
                self.views.append(View(c, tgt, k))
        for v in self.views:
            v.full(self.V, self.T)
        self.state = self._full_terms()
        self.E_cur = self._energy(self.state)

    def _full_terms(self):
        st = {}
        st["iou"] = 1.0 - self.ref.iou()
        st["bnd"] = self.ref.boundary_term()
        st["lm"] = self._lm_term(self.V)
        if self.w["sdf"] > 0:
            self.sdf_v = np.abs(self.proxy.sdf(self.V))
            self.sdf_c = np.abs(self.proxy.sdf(self.V[self.T].mean(1)))
            self.sdf_e = np.abs(self.proxy.sdf(0.5 * (self.V[self.E[:, 0]] + self.V[self.E[:, 1]])))
            self.sdf_sum = self.sdf_v.sum() + self.sdf_c.sum() + self.sdf_e.sum()
            self.sdf_n = len(self.sdf_v) + len(self.sdf_c) + len(self.sdf_e)
            st["sdf"] = self.sdf_sum / self.sdf_n / 0.01
        else:
            st["sdf"] = 0.0
        nv = [v.iou() for v in self.views[1:]]
        st["nv"] = 1.0 - float(np.mean(nv)) if nv else 0.0
        self.q = tri_quality(self.V, self.T)
        st["reg"] = float((1 - self.q).mean())
        st["sym"] = self._sym_term(self.V)
        return st

    def _energy(self, st):
        return float(sum(self.w[k] * st[k] for k in st))

    def _lm_term(self, V):
        if not len(self.lm_names):
            return 0.0
        P = (self.lm_B[:, :, None] * V[self.lm_I]).sum(1)
        uv = self.cam_full.project_uv(P)
        e = np.linalg.norm(uv - self.lm_ref, axis=1)
        return float((self.lm_w * e).sum() / self.diag_full)

    def _sym_term(self, V):
        if self.sym is None or not len(self.sym):
            return 0.0
        a, b = V[self.sym[:, 0]], V[self.sym[:, 1]] * np.array([1, -1, 1])
        return float(np.linalg.norm(a - b, axis=1).mean() / 0.01)

    # ---------------------------------------------------------------- proposals
    def _count(self, kind, key):
        c = self.counters.setdefault(kind, {"proposed": 0, "accepted": 0, "rejected_constraint": 0,
                                            "rejected_worse": 0})
        c[key] += 1

    def try_move(self, idx, newpos, kind) -> bool:
        ok = self._try_move(idx, newpos, kind)
        if self.s.snapshot_every:
            self._maybe_snapshot()
        return ok

    def _try_move(self, idx, newpos, kind) -> bool:
        idx = np.asarray(idx, np.int64)
        newpos = np.array(newpos, float).reshape(-1, 3)
        keep = ~self.frozen[idx]
        idx, newpos = idx[keep], newpos[keep]
        self.proposals += 1
        self._count(kind, "proposed")
        if not len(idx):
            self._count(kind, "rejected_constraint")
            return False
        pin = self.pinned[idx]
        newpos[pin, 2] = 0.0
        if (newpos[:, 2] < 0).any():
            self._count(kind, "rejected_constraint")
            return False
        aff = np.unique(np.concatenate([self.vt[i] for i in idx]))
        if not len(aff):
            self._count(kind, "rejected_constraint")
            return False
        Vn = self.V.copy()
        Vn[idx] = newpos
        Ta = self.T[aff]
        n0, a0, e0 = _tri_geom(self.V, Ta)
        n1, a1, e1 = _tri_geom(Vn, Ta)
        bad0 = (a0 < MIN_AREA) | (e0 < MIN_EDGE)
        bad1 = (a1 < MIN_AREA) | (e1 < MIN_EDGE)
        worse = bad1 & (~bad0 | (a1 < a0) | (e1 < e0))
        cosang = (n0 * n1).sum(1) / np.maximum(np.linalg.norm(n0, axis=1) * np.linalg.norm(n1, axis=1), 1e-30)
        if worse.any() or (cosang < math.cos(math.radians(MAX_NORMAL_TURN_DEG))).any():
            self._count(kind, "rejected_constraint")
            return False
        affE = np.unique(np.concatenate([self.ve[i] for i in idx]))
        return self._evaluate_and_commit(Vn, self.T, aff, idx, affE, kind)

    def _evaluate_and_commit(self, Vn, Tn, aff, idx, affE, kind, topo_change=False) -> bool:
        st = dict(self.state)
        patches = []
        for v in self.views:
            uvn = v.uv.copy()
            if len(idx):
                uvn[idx] = v.project(Vn[idx])
            patch, inter, union = v.trial(self.T[aff], Tn[aff], uvn)
            patches.append((patch, inter, union, uvn))
        r = patches[0]
        st["iou"] = 1.0 - self.ref.iou(r[1], r[2])
        st["bnd"] = self.ref.boundary_term(r[0])
        if len(self.views) > 1:
            st["nv"] = 1.0 - float(np.mean([v.iou(p[1], p[2]) for v, p in zip(self.views[1:], patches[1:])]))
        st["lm"] = self._lm_term(Vn) if not topo_change else self.state["lm"]
        if self.w["sdf"] > 0:
            sv = np.abs(self.proxy.sdf(Vn[idx])) if len(idx) else np.zeros(0)
            sc = np.abs(self.proxy.sdf(Vn[Tn[aff]].mean(1)))
            Ea = self._E_trial[affE] if topo_change else self.E[affE]
            se = np.abs(self.proxy.sdf(0.5 * (Vn[Ea[:, 0]] + Vn[Ea[:, 1]])))
            ssum = (self.sdf_sum - self.sdf_v[idx].sum() + sv.sum() - self.sdf_c[aff].sum() + sc.sum()
                    - self.sdf_e[affE].sum() + se.sum())
            st["sdf"] = ssum / self.sdf_n / 0.01
        qn = tri_quality(Vn, Tn[aff])
        st["reg"] = float(self.state["reg"] + ((self.q[aff] - qn).sum()) / len(self.T))
        st["sym"] = self._sym_term(Vn) if self.sym is not None else 0.0
        En = self._energy(st)
        if En < self.E_cur - 1e-12:
            for v, (patch, inter, union, uvn) in zip(self.views, patches):
                v.commit(patch, inter, union, uvn)
            self.V = Vn
            self.q[aff] = qn
            if self.w["sdf"] > 0:
                if len(idx):
                    self.sdf_v[idx] = sv
                self.sdf_c[aff] = sc
                self.sdf_e[affE] = se
                self.sdf_sum = ssum
            self.state, self.E_cur = st, En
            self._count(kind, "accepted")
            return True
        self._count(kind, "rejected_worse")
        return False

    # ---------------------------------------------------------------- move generators
    def _cam_axes(self):
        mw = self.cam_full.matrix_world
        return mw[:3, 0], mw[:3, 1], -mw[:3, 2]

    def pass_directed(self, budget):
        used = 0
        dM = self.ref.contour_distance()
        if dM is None:
            return 0
        r0, r1, c0, c1 = self.ref.crop
        right, up, fwd = self._cam_axes()
        u, v, depth = self.ref.cam.project(self.V)
        f = self.ref.cam.f_px
        order = self.rng.permutation(self.n)
        for i in order:
            if used >= budget:
                break
            if self.frozen[i] or not (c0 <= u[i] < c1 - 1 and r0 <= v[i] < r1 - 1):
                continue
            if _bilinear(dM, u[i] - c0, v[i] - r0) > 1.5:
                continue
            s = float(_bilinear(self.ref.sdist, u[i], v[i]))
            if abs(s) < 0.5:
                continue
            g = np.array([float(_bilinear(self.ref.gx, u[i], v[i])), float(_bilinear(self.ref.gy, u[i], v[i]))])
            gn = np.linalg.norm(g)
            if gn < 1e-6:
                continue
            step = -s * g / gn                          # pixels at working resolution
            uu, vv, dd = self.ref.cam.project(self.V[i:i + 1])
            for fac in (1.0, 0.5, 0.25):
                du, dv = fac * step
                dw = (du * dd[0] / f) * right - (dv * dd[0] / f) * up
                used += 1
                if self.try_move([i], self.V[i] + dw, "directed_silhouette"):
                    break
                if used >= budget:
                    break
        return used

    def pass_sdf(self, budget):
        if self.w["sdf"] <= 0:
            return 0
        used = 0
        _, _, fwd = self._cam_axes()
        d = self.proxy.sdf(self.V)
        g = self.proxy.grad(self.V)
        order = self.rng.permutation(self.n)
        for i in order:
            if used >= budget:
                break
            if self.frozen[i] or abs(d[i]) < 5e-4:
                continue
            n = g[i] / max(np.linalg.norm(g[i]), 1e-9)
            c = float(n @ fwd)
            cands = []
            if abs(c) > 0.25:
                step = float(np.clip(-d[i] / c, -0.03, 0.03))
                cands += [step * fwd, 0.5 * step * fwd]
            cands += [-d[i] * n, -0.5 * d[i] * n]
            for dw in cands:
                used += 1
                if self.try_move([i], self.V[i] + dw, "sdf"):
                    break
        return used

    def pass_stochastic(self, budget):
        used = 0
        groups = list(self.groups)
        while used < budget:
            used += 1
            if groups and self.rng.random() < self.s.group_frac:
                k = groups[self.rng.integers(len(groups))]
                ids = self.groups[k]
                if self.rng.random() < 0.5:
                    dv = self.rng.normal(0, self.gsig_t[k], 3)
                    ok = self.try_move(ids, self.V[ids] + dv, "group_translate")
                    self.gsig_t[k] = float(np.clip(self.gsig_t[k] * (1.5 if ok else 1.5 ** -0.25),
                                                   SIGMA_MIN, SIGMA_MAX))
                else:
                    ax = self.rng.normal(size=3); ax /= np.linalg.norm(ax)
                    ang = self.rng.normal(0, self.gsig_r[k])
                    c = self.V[ids].mean(0)
                    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                    Rm = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
                    ok = self.try_move(ids, (self.V[ids] - c) @ Rm.T + c, "group_rotate")
                    self.gsig_r[k] = float(np.clip(self.gsig_r[k] * (1.5 if ok else 1.5 ** -0.25),
                                                   math.radians(0.2), math.radians(20)))
                continue
            i = int(self.rng.integers(self.n))
            if self.frozen[i]:
                continue
            step = self.rng.normal(0, self.sigma[i], 3)
            ok = self.try_move([i], self.V[i] + step, "stochastic")
            self.sigma[i] = float(np.clip(self.sigma[i] * (1.5 if ok else 1.5 ** -0.25), SIGMA_MIN, SIGMA_MAX))
        return used

    def pass_flips(self, budget):
        used = 0
        order = self.rng.permutation(len(self.E))
        cos_max = math.cos(math.radians(MAX_NORMAL_TURN_DEG))
        for ei in order:
            if used >= budget:
                break
            if ei >= len(self.E):
                continue
            tt = self.edge_tris[ei]
            if len(tt) != 2:
                continue
            t1, t2 = tt
            if t1 in self.lm_tris or t2 in self.lm_tris:
                continue
            if self.reg[t1] != self.reg[t2] or self.mat[t1] != self.mat[t2]:
                continue
            a, b = (int(x) for x in self.E[ei])
            try:
                x1, y1, c = _rot(self.T[t1], a, b)
                x2, y2, e = _rot(self.T[t2], a, b)
            except ValueError:
                continue
            if (x1, y1) != (y2, x2) or c == e:
                continue
            ce = (min(c, e), max(c, e))
            if any(ce[0] == p and ce[1] == q for p, q in self.E[self.ve[c]]):
                continue
            Tn = self.T.copy()
            Tn[t1] = [x1, e, c]
            Tn[t2] = [e, y1, c]
            aff = np.array([t1, t2])
            n0, _, _ = _tri_geom(self.V, self.T[aff])
            n1, a1, e1 = _tri_geom(self.V, Tn[aff])
            nav = n0.sum(0)
            nav /= max(np.linalg.norm(nav), 1e-30)
            if (a1 < MIN_AREA).any() or (e1 < MIN_EDGE).any():
                continue
            if ((n1 @ nav) / np.maximum(np.linalg.norm(n1, axis=1), 1e-30) < cos_max).any():
                continue
            if (n1[0] @ n1[1]) <= 0:
                continue
            used += 1
            self.proposals += 1
            self._count("flip", "proposed")
            self._E_trial = self.E.copy()
            self._E_trial[ei] = ce
            if self._evaluate_and_commit(self.V.copy(), Tn, aff, np.zeros(0, np.int64),
                                         np.array([ei]), "flip", topo_change=True):
                self.T = Tn
                self._build_topology()          # re-sorts the edge list
                self._build_lm_index()
                self.state = self._full_terms()  # re-index per-edge SDF samples
                self.E_cur = self._energy(self.state)
            if self.s.snapshot_every:
                self._maybe_snapshot()
        return used

    # ---------------------------------------------------------------- snapshots (§13)
    def _maybe_snapshot(self, force: bool = False):
        """Write the current vertices every `snapshot_every` proposals (read-only w.r.t. the search:
        no RNG draw, no state change)."""
        p = self.proposals
        if p == self._snap_last or (not force and p % self.s.snapshot_every):
            return
        self._snap_last = p
        d = Path(self.s.snapshot_dir)
        d.mkdir(parents=True, exist_ok=True)
        snap = {"proposal": int(p), "vertices": [[float(c) for c in v] for v in self.V],
                "objective": float(self.E_cur), "stage_res": int(getattr(self, "res", 0)),
                "terms": {k: float(v) for k, v in self.state.items()}}
        if self.s.flips:
            snap["triangles"] = [[int(c) for c in t] for t in self.T]
        tmp = d / f".snap_{p:07d}.json.tmp"
        tmp.write_text(json.dumps(snap))
        os.replace(tmp, d / f"snap_{p:07d}.json")

    # ---------------------------------------------------------------- driver
    def _log(self, stage, note=""):
        row = {"t_s": round(time.time() - self.t0, 3), "stage_res": self.res, "proposals": self.proposals,
               "E": self.E_cur, "terms": {k: float(v) for k, v in self.state.items()},
               "counters": json.loads(json.dumps(self.counters)), "note": note}
        self.trace.append(row)
        if self.s.verbose:
            t = self.state
            print(f"[{row['t_s']:7.1f}s res {self.res} #{self.proposals:6d}] E {self.E_cur:.5f}  "
                  f"1-IoU {t['iou']:.4f} bnd {t['bnd']:.5f} lm {t['lm']:.5f} sdf {t['sdf']:.4f} "
                  f"nv {t['nv']:.4f} reg {t['reg']:.4f} {note}", flush=True)

    def _resync(self):
        """Full recount of every view (guards against incremental drift)."""
        drift = 0
        for v in self.views:
            old = v.counts.copy()
            v.full(self.V, self.T)
            drift += int((old != v.counts).sum())
        st = self._full_terms()
        self.state, self.E_cur = st, self._energy(st)
        return drift

    def run(self):
        res_list = list(self.s.res)
        total_iters = self.s.iters
        total_s = self.s.minutes * 60 if self.s.minutes else None
        for si, res in enumerate(res_list):
            self.setup(res)
            self._log(si, "stage start")
            if si == 0 and self.s.snapshot_every:
                self._maybe_snapshot(force=True)
            frac = (si + 1) / len(res_list)
            it_end = None if total_iters is None else int(round(total_iters * frac))
            t_end = None if total_s is None else self.t0 + total_s * frac

            def left():
                a = math.inf if it_end is None else it_end - self.proposals
                if t_end is not None and time.time() >= t_end:
                    return 0
                return max(a, 0)

            rnd = 0
            while left() > 0:
                chunk = max(self.n, 50)
                self.pass_directed(min(left(), 3 * self.n))
                if left() > 0:
                    self.pass_sdf(min(left(), 2 * self.n))
                if left() > 0:
                    self.pass_stochastic(min(left(), 4 * chunk))
                if self.s.flips and left() > 0:
                    self.pass_flips(min(left(), len(self.E)))
                drift = self._resync()
                rnd += 1
                self._log(si, f"round {rnd}" + (f" resync drift {drift}px" if drift else ""))
        if self.s.snapshot_every:
            self._maybe_snapshot(force=True)
        return self.result()

    def result(self) -> dict:
        d = json.loads(json.dumps(self.d))
        d["vertices"] = [[float(c) for c in v] for v in self.V]
        d["triangles"] = [[int(c) for c in t] for t in self.T]
        d["landmark_attach"] = self.att
        meshio.sync_landmarks(d)
        return d


def _rot(t, a, b):
    t = [int(x) for x in t]
    for r in range(3):
        x, y, c = t[r], t[(r + 1) % 3], t[(r + 2) % 3]
        if {x, y} == {a, b}:
            return x, y, c
    raise ValueError


def unofficial_metrics(d: dict, src: Sources) -> dict:
    """catlib.metrics at full resolution through catlib.raster: UNOFFICIAL."""
    from . import metrics
    mesh = meshio.CatMesh(d)
    cam = Camera.from_json(src.camera)
    derived = paths.derived_paths(src.derived_dir)
    proxy = src.proxy_mesh if src.proxy_mesh.is_file() else None
    met, _ = metrics.compute_metrics(mesh, cam, derived, None, proxy, load_scene(src.scene))
    sec = met.get("secondary", {})
    return {"UNOFFICIAL": "catlib.raster @%dpx, not a Blender render" % cam.width,
            "iou": met["iou"], "boundary_mean_px": met["boundary_mean_px"],
            "boundary_hd95_px": met["boundary_hd95_px"], "legs_gap_iou": met["legs_gap_iou"],
            "region_iou": met["region_iou"],
            "landmark_sharp_mean_px": met["landmark_sharp_mean_px"],
            "landmark_sharp_max_px": met["landmark_sharp_max_px"],
            "landmark_flags": met["landmark_flags"],
            "novel_view_iou_vs_proxy": sec.get("novel_view_iou_vs_proxy"),
            "chamfer_mm_vs_proxy": sec.get("chamfer_mm_vs_proxy")}
