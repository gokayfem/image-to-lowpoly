"""Official metrics of CONTRACT §7.

Module import is numpy-only (Blender-safe); PIL / scipy are imported inside the functions
that need them (venv only).

Conventions documented here once:
- Masks are bool (H,W); files are binarised with >= 128 on the luminance of the RGB
  channels (alpha ignored).
- A window [u0, v0, u1, v1] (pixel convention, CONTRACT §3) contains pixel (i, j) iff its
  centre (i+.5, j+.5) lies in the closed box, i.e. columns ceil(u0-.5) .. floor(u1-.5).
- Boundary pixels: mask pixels with at least one 4-neighbour outside the mask (pixels
  outside the image count as outside).  Distances are Euclidean between pixel centres.
- IoU of two empty sets is reported as null.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import geom, meshio, raster
from .camera import Camera, load_scene, novel_view_cameras

OFF_SURFACE_M = 0.002
OCCLUDED_M = 0.025
OCCLUDED_PAW_M = 0.060      # paw_*: sole centre sits behind the paw's own front surface (§7)


def occluded_tolerance(name: str) -> float:
    """CONTRACT §7: 60 mm for paw_* landmarks, 25 mm otherwise."""
    return OCCLUDED_PAW_M if name.startswith("paw_") else OCCLUDED_M
SECONDARY_VIEW_SIZE = 512
CHAMFER_SAMPLES = 20000
CHAMFER_SEED = 0


# ---- io -----------------------------------------------------------------------------------

def load_mask(path) -> np.ndarray:
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA").convert("RGB")
    if im.mode == "I;16" or im.mode == "I":
        a = np.asarray(im).astype(np.float64)
        return a >= (128 * 257 if a.max() > 255 else 128)
    a = np.asarray(im.convert("L"))
    return a >= 128


def save_mask(path, mask: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray((np.asarray(mask, bool) * 255).astype(np.uint8), "L").save(path)


def load_regions(path) -> dict[str, list[float]]:
    d = json.loads(Path(path).read_text())
    if isinstance(d.get("regions"), dict):
        d = d["regions"]
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and "box" in v:
            v = v["box"]
        if isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v):
            out[k] = [float(x) for x in v]
    return out


def load_ref_landmarks(path) -> dict[str, dict]:
    d = json.loads(Path(path).read_text())
    lm = d.get("landmarks", d)
    out = {}
    for k, v in lm.items():
        if isinstance(v, dict) and "uv" in v:
            out[k] = {"uv": [float(v["uv"][0]), float(v["uv"][1])],
                      "class": v.get("class", "sharp" if k in meshio.SHARP_LANDMARKS else "fuzzy")}
    return out


def mask_bbox(mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"u0": 0, "v0": 0, "u1": 0, "v1": 0, "diag_px": 0.0, "area_px": 0}
    u0, u1, v0, v1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    return {"u0": u0, "v0": v0, "u1": u1, "v1": v1,
            "diag_px": float(np.hypot(u1 - u0, v1 - v0)), "area_px": int(mask.sum())}


def reference_diag(bbox_path, ref_mask) -> float:
    p = Path(bbox_path) if bbox_path else None
    if p and p.is_file():
        return float(json.loads(p.read_text())["diag_px"])
    return mask_bbox(ref_mask)["diag_px"]


# ---- mask metrics -------------------------------------------------------------------------

def iou(a: np.ndarray, b: np.ndarray):
    u = np.logical_or(a, b).sum()
    return None if u == 0 else float(np.logical_and(a, b).sum() / u)


def boundary(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, bool)
    p = np.pad(m, 1, constant_values=False)
    inner = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~inner


def boundary_distances(R: np.ndarray, M: np.ndarray) -> dict:
    from scipy.ndimage import distance_transform_edt
    bR, bM = boundary(R), boundary(M)
    if not bR.any() or not bM.any():
        return {"boundary_mean_px": None, "boundary_hd95_px": None, "boundary_max_px": None,
                "boundary_R_to_M_mean_px": None, "boundary_M_to_R_mean_px": None}
    dR = distance_transform_edt(~bM)[bR]     # each R-boundary pixel -> nearest M-boundary
    dM = distance_transform_edt(~bR)[bM]
    return {
        "boundary_mean_px": float(0.5 * (dR.mean() + dM.mean())),
        "boundary_hd95_px": float(max(np.percentile(dR, 95), np.percentile(dM, 95))),
        "boundary_max_px": float(max(dR.max(), dM.max())),
        "boundary_R_to_M_mean_px": float(dR.mean()),
        "boundary_M_to_R_mean_px": float(dM.mean()),
    }


def window_slices(box, W: int, H: int):
    u0, v0, u1, v1 = box
    c0 = max(int(np.ceil(u0 - 0.5)), 0)
    c1 = min(int(np.floor(u1 - 0.5)), W - 1)
    r0 = max(int(np.ceil(v0 - 0.5)), 0)
    r1 = min(int(np.floor(v1 - 0.5)), H - 1)
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


def mask_metrics(R: np.ndarray, M: np.ndarray, regions: dict | None, diag_px: float) -> dict:
    if R.shape != M.shape:
        raise ValueError(f"mask shapes differ: reference {R.shape} vs render {M.shape}")
    H, W = R.shape
    out = {
        "iou": iou(R, M),
        "ref_uncovered_frac": float((R & ~M).sum() / max(R.sum(), 1)),
        "render_outside_frac": float((M & ~R).sum() / max(M.sum(), 1)) if M.any() else None,
        "ref_area_px": int(R.sum()),
        "render_area_px": int(M.sum()),
    }
    b = boundary_distances(R, M)
    out.update(b)
    for k in ("boundary_mean_px", "boundary_hd95_px", "boundary_max_px"):
        out[k.replace("_px", "_norm")] = None if b[k] is None or not diag_px else b[k] / diag_px
    out["diag_px"] = diag_px
    ri = {}
    gap = None
    for name, box in (regions or {}).items():
        sl = window_slices(box, W, H)
        ri[name] = iou(R[sl], M[sl])
        if name == "legs":
            gap = iou(~R[sl], ~M[sl])
    out["region_iou"] = ri
    out["legs_gap_iou"] = gap
    return out


# ---- landmarks -----------------------------------------------------------------------------

def landmark_metrics(mesh: meshio.CatMesh, cam: Camera, ref_lm: dict, diag_px: float) -> dict:
    V, T = mesh.vertices, mesh.triangles
    names = [n for n in meshio.LANDMARKS if n in mesh.landmarks]
    table = {}
    if names:
        P = np.stack([mesh.landmarks[n] for n in names])
        u, v, depth = cam.project(P)
        dist, _ = geom.point_mesh_distance(P, V, T)
        o = cam.location
        for i, n in enumerate(names):
            d = P[i] - o
            L = float(np.linalg.norm(d))
            t = geom.ray_hits(o, d / L, V, T)
            first = float(t.min()) if np.isfinite(t).any() else None
            row = {"proj_uv": [float(u[i]), float(v[i])], "depth_m": float(depth[i]),
                   "surface_dist_mm": float(dist[i] * 1000),
                   "off_surface": bool(dist[i] > OFF_SURFACE_M),
                   "first_hit_m": first,
                   "occluded": bool(first is not None and first < L - occluded_tolerance(n)),
                   "occluded_tol_m": occluded_tolerance(n),
                   "class": "sharp" if n in meshio.SHARP_LANDMARKS else "fuzzy"}
            if n in ref_lm:
                ru = ref_lm[n]["uv"]
                row["ref_uv"] = ru
                row["error_px"] = float(np.hypot(u[i] - ru[0], v[i] - ru[1]))
            else:
                row["ref_uv"] = None
                row["error_px"] = None
            table[n] = row

    def agg(cls_names):
        e = [table[n]["error_px"] for n in cls_names if n in table and table[n]["error_px"] is not None]
        return (float(np.mean(e)) if e else None, float(np.max(e)) if e else None, len(e))

    sm, sx, sn = agg(meshio.SHARP_LANDMARKS)
    fm, fx, fn = agg(meshio.FUZZY_LANDMARKS)
    return {
        "landmark_sharp_mean_px": sm, "landmark_sharp_max_px": sx,
        "landmark_sharp_mean_norm": None if sm is None or not diag_px else sm / diag_px,
        "landmark_sharp_n": sn,
        "landmark_fuzzy_mean_px": fm, "landmark_fuzzy_max_px": fx, "landmark_fuzzy_n": fn,
        "landmark_flags": {
            "off_surface": [n for n, r in table.items() if r["off_surface"]],
            "occluded": [n for n, r in table.items() if r["occluded"]],
            "missing_in_reference": [n for n in meshio.LANDMARKS if n not in ref_lm],
            "missing_in_mesh": [n for n in meshio.LANDMARKS if n not in mesh.landmarks],
        },
        "landmarks": table,
    }


# ---- secondary metrics vs the inferred proxy ----------------------------------------------

def sample_surface(V, T, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = geom.tri_areas(V, T)
    p = a / a.sum()
    idx = rng.choice(len(T), size=n, p=p)
    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    A, B, C = V[T[idx, 0]], V[T[idx, 1]], V[T[idx, 2]]
    return (1 - r1)[:, None] * A + (r1 * (1 - r2))[:, None] * B + (r1 * r2)[:, None] * C


def load_proxy(path):
    p = Path(path)
    if not p.is_file():
        return None
    z = np.load(p)
    return np.asarray(z["vertices"], float), np.asarray(z["triangles"], np.int64)


def secondary_metrics(mesh: meshio.CatMesh, cam: Camera, proxy_path, scene: dict | None = None) -> dict:
    proxy = load_proxy(proxy_path) if proxy_path else None
    base = {"proxy_path": str(proxy_path) if proxy_path else None,
            "note": "inferred proxy geometry, NOT ground truth"}
    if proxy is None:
        return {**base, "novel_view_iou_vs_proxy": None, "novel_view_iou_per_view": None,
                "chamfer_mm_vs_proxy": None}
    from scipy.spatial import cKDTree
    PV, PT = proxy
    views = novel_view_cameras(cam, scene or load_scene(), size=SECONDARY_VIEW_SIZE)
    per = {}
    for name, c in views.items():
        a = raster.silhouette(mesh.vertices, mesh.triangles, c)
        b = raster.silhouette(PV, PT, c)
        per[name] = iou(a, b)
    vals = [x for x in per.values() if x is not None]
    A = sample_surface(mesh.vertices, mesh.triangles, CHAMFER_SAMPLES, CHAMFER_SEED)
    B = sample_surface(PV, PT, CHAMFER_SAMPLES, CHAMFER_SEED + 1)
    dab = cKDTree(B).query(A)[0]
    dba = cKDTree(A).query(B)[0]
    return {**base,
            "novel_view_iou_vs_proxy": float(np.mean(vals)) if vals else None,
            "novel_view_iou_per_view": per,
            "novel_view_size": SECONDARY_VIEW_SIZE,
            "chamfer_mm_vs_proxy": float(500.0 * (dab.mean() + dba.mean())),
            "chamfer_mesh_to_proxy_mm": float(1000 * dab.mean()),
            "chamfer_proxy_to_mesh_mm": float(1000 * dba.mean()),
            "chamfer_samples": CHAMFER_SAMPLES, "chamfer_seed": CHAMFER_SEED,
            "chamfer_definition": "0.5*(mean NN dist mesh->proxy + proxy->mesh) between "
                                  "area-weighted surface point samples, mm"}


# ---- all together --------------------------------------------------------------------------

def compute_metrics(mesh: meshio.CatMesh, cam: Camera, derived: dict, rendered_mask=None,
                    proxy_path=None, scene: dict | None = None) -> dict:
    """derived = catlib.paths.derived_paths(...).  rendered_mask = Blender silhouette (bool)
    for official numbers, or None -> unofficial numbers from catlib.raster."""
    R = load_mask(derived["mask"])
    H, W = R.shape
    if (W, H) != (cam.width, cam.height):
        raise ValueError(f"reference mask {W}x{H} != camera {cam.width}x{cam.height}")
    diag = reference_diag(derived.get("bbox"), R)
    regions = load_regions(derived["regions"]) if Path(derived["regions"]).is_file() else {}
    ref_lm = load_ref_landmarks(derived["landmarks"]) if Path(derived["landmarks"]).is_file() else {}
    rast, info = raster.silhouette(mesh.vertices, mesh.triangles, cam, return_info=True)
    official = rendered_mask is not None
    M = np.asarray(rendered_mask, bool) if official else rast
    out = {"official": official,
           "silhouette_source": "blender_render" if official else "catlib.raster (UNOFFICIAL)",
           "camera_status": cam.status,
           "image_size": [W, H]}
    out.update(mask_metrics(R, M, regions, diag))
    out["raster_vs_blender_iou"] = iou(rast, M) if official else None
    out["raster_dropped_behind_camera"] = info["dropped_behind_camera"]
    out.update(landmark_metrics(mesh, cam, ref_lm, diag))
    out["secondary"] = secondary_metrics(mesh, cam, proxy_path, scene)
    return out, M
