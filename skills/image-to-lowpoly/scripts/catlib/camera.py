"""Pinhole camera of CONTRACT §3 and the orbit / novel-view cameras of CONTRACT §8.

numpy-only; importable inside Blender.

Pixel convention (CONTRACT §3; same as the earlier project's gbuffers.camera_model,
adapted from the Blockout 2 Real project (gokayfem/blockout-to-real), 
blockout-to-real/skills/blockout-to-real/scripts/blender/gbuffers.py, read-only):
u grows right, v grows down, origin at the top-left corner of the top-left pixel, pixel
centres at integer + 0.5.  Blender camera axes: +X right, +Y up, -Z forward.
    u = cx + f * xc / (-zc),   v = cy - f * yc / (-zc),   depth = -zc
with cx = W/2, cy = H/2 (no shift).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import paths

WORLD_UP = np.array([0.0, 0.0, 1.0])


def look_at_matrix(location, look_at, roll_deg: float = 0.0, up_hint=None) -> np.ndarray:
    """matrix_world = [right | up | -fwd | location] (columns), CONTRACT §3.

    fwd = normalize(look_at - location); right = normalize(fwd x up_hint); up = right x fwd.
    up_hint defaults to world +Z.  Roll rotates right/up about fwd (positive = counter-
    clockwise as seen by the camera, i.e. Blender's local-Z rotation)."""
    loc = np.asarray(location, float)
    fwd = np.asarray(look_at, float) - loc
    fwd = fwd / np.linalg.norm(fwd)
    hint = WORLD_UP if up_hint is None else np.asarray(up_hint, float)
    right = np.cross(fwd, hint)
    n = np.linalg.norm(right)
    if n < 1e-9:
        raise ValueError("look direction parallel to the up hint; pass up_hint")
    right /= n
    up = np.cross(right, fwd)
    if roll_deg:
        r = math.radians(roll_deg)
        c, s = math.cos(r), math.sin(r)
        right, up = c * right + s * up, -s * right + c * up
    m = np.eye(4)
    m[:3, 0] = right
    m[:3, 1] = up
    m[:3, 2] = -fwd
    m[:3, 3] = loc
    return m


@dataclass
class Camera:
    width: int
    height: int
    f_px: float
    matrix_world: np.ndarray          # 4x4 camera-to-world
    focal_length_mm: float = 85.0
    sensor_width_mm: float = 36.0
    clip_start: float = 0.05
    clip_end: float = 50.0
    name: str = "reference"
    status: str = "unknown"
    meta: dict = field(default_factory=dict)

    # ---- construction -------------------------------------------------------------
    @classmethod
    def from_json(cls, path: Path | str | None = None) -> "Camera":
        path = Path(path) if path else paths.CAMERA_JSON
        d = json.loads(Path(path).read_text())
        W, H = (int(x) for x in d["resolution"])
        if d.get("sensor_fit", "HORIZONTAL") != "HORIZONTAL":
            raise ValueError("only sensor_fit HORIZONTAL is supported (CONTRACT §3)")
        f_px = d["focal_length_mm"] / d["sensor_width_mm"] * W
        mw = look_at_matrix(d["location"], d["look_at"], d.get("roll_deg", 0.0))
        cam = cls(W, H, f_px, mw, d["focal_length_mm"], d["sensor_width_mm"],
                  d.get("clip_start", 0.05), d.get("clip_end", 50.0), "reference",
                  d.get("status", "unknown"), {"source": str(path), "json": d})
        return cam

    @property
    def location(self) -> np.ndarray:
        return self.matrix_world[:3, 3].copy()

    @property
    def world_to_cam(self) -> np.ndarray:
        return np.linalg.inv(self.matrix_world)

    def scaled(self, size) -> "Camera":
        """Same camera at another resolution. size = int (square) or (W, H) with the same
        aspect.  Intrinsics scale with the image so that a pixel-convention coordinate
        scales exactly by size/W (the convention's origin is the image corner)."""
        if isinstance(size, (int, np.integer)):
            W2, H2 = int(size), int(round(size * self.height / self.width))
        else:
            W2, H2 = (int(x) for x in size)
        if abs(W2 / self.width - H2 / self.height) > 1e-9:
            raise ValueError("scaled() must keep the aspect ratio")
        s = W2 / self.width
        return Camera(W2, H2, self.f_px * s, self.matrix_world.copy(), self.focal_length_mm,
                      self.sensor_width_mm, self.clip_start, self.clip_end, self.name,
                      self.status, dict(self.meta))

    # ---- projection -----------------------------------------------------------------
    def to_camera(self, points_world) -> np.ndarray:
        p = np.asarray(points_world, float).reshape(-1, 3)
        w2c = self.world_to_cam
        return p @ w2c[:3, :3].T + w2c[:3, 3]

    def project(self, points_world):
        """-> (u, v, depth) arrays in the CONTRACT §3 pixel convention.  depth = -zc
        (positive in front of the camera)."""
        pc = self.to_camera(points_world)
        depth = -pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.width / 2.0 + self.f_px * pc[:, 0] / depth
            v = self.height / 2.0 - self.f_px * pc[:, 1] / depth
        return u, v, depth

    def project_uv(self, points_world) -> np.ndarray:
        u, v, _ = self.project(points_world)
        return np.stack([u, v], 1)

    def pixel_rays(self, uv) -> tuple[np.ndarray, np.ndarray]:
        """World-space unit ray directions through pixel-convention points (N,2)."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        xc = (uv[:, 0] - self.width / 2.0) / self.f_px
        yc = -(uv[:, 1] - self.height / 2.0) / self.f_px
        d_cam = np.stack([xc, yc, -np.ones_like(xc)], 1)
        d = d_cam @ self.matrix_world[:3, :3].T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return np.broadcast_to(self.location, d.shape).copy(), d

    def check_stored_matrix(self, tol: float = 1e-6) -> dict:
        """Stored camera.json matrix_world vs the matrix derived from location/look_at/roll."""
        stored = self.meta.get("json", {}).get("matrix_world")
        if stored is None:
            return {"ok": False, "reason": "no stored matrix_world"}
        err = float(np.abs(np.asarray(stored, float) - self.matrix_world).max())
        return {"ok": err <= tol, "max_abs_diff": err, "tol": tol}

    def to_dict(self) -> dict:
        return {"name": self.name, "width": self.width, "height": self.height,
                "f_px": self.f_px, "matrix_world": self.matrix_world.tolist(),
                "status": self.status}


# ---- novel views (CONTRACT §8) ------------------------------------------------------------

def load_scene(path: Path | str | None = None) -> dict:
    return json.loads(Path(path or paths.SCENE_JSON).read_text())


def orbit_camera(ref: Camera, az_deg: float, el_deg: float, target=None, radius=None,
                 size=None, name: str = "orbit") -> Camera:
    """Camera orbiting `target` (default: reference look_at) at the reference camera
    distance, same focal length.  az 0 = on +X (in front of the cat), 90 = on +Y (cat's
    left), 180 = behind, -90 = cat's right.  el = degrees above the horizontal."""
    tgt = np.asarray(target if target is not None else ref.meta["json"]["look_at"], float)
    r = float(radius if radius is not None else np.linalg.norm(ref.location - np.asarray(
        ref.meta["json"]["look_at"], float)))
    a, e = math.radians(az_deg), math.radians(el_deg)
    loc = tgt + r * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    mw = look_at_matrix(loc, tgt)
    cam = Camera(ref.width, ref.height, ref.f_px, mw, ref.focal_length_mm, ref.sensor_width_mm,
                 ref.clip_start, ref.clip_end, name, ref.status, {"json": {"look_at": tgt.tolist()}})
    return cam.scaled(size) if size else cam


def top_camera(ref: Camera, target=None, radius=None, image_up=(1.0, 0.0, 0.0), size=None,
               name: str = "top") -> Camera:
    """Straight down onto `target`; image-up = `image_up` (CONTRACT §8: +X)."""
    tgt = np.asarray(target if target is not None else ref.meta["json"]["look_at"], float)
    r = float(radius if radius is not None else np.linalg.norm(ref.location - np.asarray(
        ref.meta["json"]["look_at"], float)))
    loc = tgt + np.array([0.0, 0.0, r])
    mw = look_at_matrix(loc, tgt, up_hint=np.asarray(image_up, float))
    cam = Camera(ref.width, ref.height, ref.f_px, mw, ref.focal_length_mm, ref.sensor_width_mm,
                 ref.clip_start, ref.clip_end, name, ref.status, {"json": {"look_at": tgt.tolist()}})
    return cam.scaled(size) if size else cam


def orbit_radius(ref: Camera, scene: dict) -> float:
    """scene.json `orbit_radius_m` (explicit, used for every novel view and the turntable);
    falls back to the reference camera's distance to its look_at only if the key is absent."""
    if "orbit_radius_m" in scene:
        return float(scene["orbit_radius_m"])
    return float(np.linalg.norm(ref.location - np.asarray(ref.meta["json"]["look_at"], float)))


def novel_view_cameras(ref: Camera | None = None, scene: dict | None = None,
                       size=None) -> dict[str, Camera]:
    """The fixed views of scene.json, in file order, at `size` (default view_resolution)."""
    ref = ref or Camera.from_json()
    scene = scene or load_scene()
    tgt = scene.get("orbit_target", ref.meta["json"]["look_at"])
    radius = orbit_radius(ref, scene)
    size = size or tuple(scene.get("view_resolution", [ref.width, ref.height]))
    out = {}
    for name, spec in scene["views"].items():
        if spec.get("straight_down"):
            out[name] = top_camera(ref, tgt, radius, spec.get("image_up", (1, 0, 0)), size, name)
        else:
            out[name] = orbit_camera(ref, spec["az_deg"], spec["el_deg"], tgt, radius, size, name)
    return out
