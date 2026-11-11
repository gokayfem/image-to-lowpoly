"""Render one cat variant with the shared camera and presentation scene (CONTRACT sections 3, 8).

    blender -b --factory-startup \
        --python scripts/render_variant.py -- --blend variants/100/.../cat_100.blend \
        --out variants/100/.../renders --passes silhouette,clay,wire,beauty \
        [--views] [--turntable] [--keep-frames] \
        [--camera contract/camera.json] [--scene contract/scene.json]

* Takes the exclusive GPU lock (catlib/gpulock.py) first thing; holds it until exit.
* Opens the variant .blend read-only (it is never saved). "The cat" = every mesh object in a
  collection whose name starts with CAT_. Their evaluated meshes are triangulated into ONE
  in-memory render mesh (world space, flat shaded, per-triangle id + material-slot name).
* Novel views and the turntable orbit scene.json `orbit_target` at `orbit_radius_m` (same
  focal length as the reference camera); the reference view uses camera.json unchanged.
* Everything presentational (camera, lights, world, shadow-catcher ground, materials,
  wireframe annotation) is built in memory from contract/camera.json + contract/scene.json.
* Shaded passes (clay, beauty) and the id pass are rendered to 32-bit EXR in a temp dir and
  encoded to 8-bit sRGB PNG here (exact sRGB OETF = Blender's Standard view transform, no
  dither). The silhouette is written by Blender's own PNG writer (Standard, dither 0).
* Silhouette / id passes: 1 sample per pixel, BOX filter of width 0.01 px, i.e. every pixel is
  a point sample at its centre -> pixel-centre coverage of the projected triangles, no blur.
* Wire: per-triangle id image at `supersample` x resolution; pixels whose id differs from a
  4-neighbour form the triangle edges (hidden lines removed by construction, silhouette
  included); dilated with a disc to `line_px` output pixels, box-downsampled to alpha and
  composited over the clay render. Constant line width in pixels at every budget.

Only Blender's bundled Python (numpy) is used. Writes render_meta.json next to the images.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent  # scripts/ (holds catlib/)


def _case_root() -> Path:
    """The case directory (set by catlib.paths in the calling process)."""
    return Path(os.environ.get("LOWPOLY_CASE") or os.getcwd())

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


def _load_gpulock():
    """Import catlib/gpulock.py by path (does not depend on catlib/__init__.py)."""
    spec = importlib.util.spec_from_file_location("catlib_gpulock", CODE_DIR / "catlib" / "gpulock.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def script_args(argv=None):
    argv = sys.argv if argv is None else argv
    return argv[argv.index("--") + 1:] if "--" in argv else []


def log(msg: str) -> None:
    print(f"[render_variant {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- pure numpy helpers
import numpy as np  # noqa: E402  (Blender bundles numpy)


def camera_matrix(location, look_at, roll_deg: float = 0.0) -> np.ndarray:
    """CONTRACT section 3: fwd = norm(look_at - loc), right = norm(fwd x Z), up = right x fwd;
    matrix_world columns = [right | up | -fwd | location]."""
    loc = np.asarray(location, float)
    fwd = np.asarray(look_at, float) - loc
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("look direction parallel to Z: use top_camera_matrix")
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    if roll_deg:
        r = math.radians(roll_deg)
        right, up = (math.cos(r) * right + math.sin(r) * up, -math.sin(r) * right + math.cos(r) * up)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, up, -fwd, loc
    return m


def top_camera_matrix(target, dist: float, image_up=(1.0, 0.0, 0.0)) -> np.ndarray:
    up = np.asarray(image_up, float)
    up /= np.linalg.norm(up)
    back = np.array([0.0, 0.0, 1.0])            # camera +Z (it looks along -Z = straight down)
    right = np.cross(up, back)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = right, up, back
    m[:3, 3] = np.asarray(target, float) + back * dist
    return m


def project(matrix_world: np.ndarray, f_px: float, W: int, H: int, pts) -> np.ndarray:
    """CONTRACT section 3 pixel convention: (u, v, depth), v down, pixel centres at +0.5."""
    p = np.atleast_2d(np.asarray(pts, float))
    w2c = np.linalg.inv(matrix_world)
    pc = p @ w2c[:3, :3].T + w2c[:3, 3]
    d = -pc[:, 2]
    return np.stack([W / 2 + f_px * pc[:, 0] / d, H / 2 - f_px * pc[:, 1] / d, d], 1)


def srgb_encode(lin: np.ndarray) -> np.ndarray:
    """Scene-linear -> 8-bit sRGB (piecewise sRGB OETF, = Blender 'Standard'), clipped, rounded."""
    x = np.clip(lin, 0.0, 1.0)
    s = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return np.clip(np.round(s * 255.0), 0, 255).astype(np.uint8)


def write_png(path: Path, img: np.ndarray) -> None:
    """Minimal PNG writer (8-bit gray or RGB, top row first)."""
    img = np.ascontiguousarray(img)
    if img.ndim == 2:
        ctype, h, w = 0, img.shape[0], img.shape[1]
    else:
        ctype, (h, w) = 2, img.shape[:2]
        img = img[:, :, :3]
    raw = np.concatenate([np.zeros((h, 1), np.uint8), img.reshape(h, -1)], 1).tobytes()

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)


def disc_offsets(radius: float):
    r = int(math.ceil(radius))
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dx * dx + dy * dy <= radius * radius + 1e-9]


def wire_alpha(ids: np.ndarray, ss: int, line_px: float) -> np.ndarray:
    """ids: (H*ss, W*ss) int, top row first. Returns (H, W) float alpha of triangle-edge lines."""
    e = np.zeros(ids.shape, bool)
    dx = ids[:, 1:] != ids[:, :-1]
    dy = ids[1:, :] != ids[:-1, :]
    # one-sided marks sit on the boundary; the disc dilation below centres the band on it
    e[:, :-1] |= dx
    e[:-1, :] |= dy
    radius = max(0.0, (line_px * ss - 1.0) / 2.0)
    if radius > 0:
        out = np.zeros_like(e)
        H2, W2 = e.shape
        for oy, ox in disc_offsets(radius):
            ys, yd = (slice(0, H2 - oy), slice(oy, H2)) if oy >= 0 else (slice(-oy, H2), slice(0, H2 + oy))
            xs, xd = (slice(0, W2 - ox), slice(ox, W2)) if ox >= 0 else (slice(-ox, W2), slice(0, W2 + ox))
            out[yd, xd] |= e[ys, xs]
        e = out
    H, W = ids.shape[0] // ss, ids.shape[1] // ss
    return e.reshape(H, ss, W, ss).mean(axis=(1, 3)).astype(np.float32)


# --------------------------------------------------------------------------- Blender
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402


# setup_device adapted from blockout-to-real/scripts/blender/render_view.py (earlier room project)
def setup_device(scene) -> str:
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        names = []
        for d in prefs.devices:
            d.use = d.type == backend
            if d.use:
                names.append(d.name)
        if names:
            scene.cycles.device = "GPU"
            return f"GPU {backend}: " + ", ".join(names)
    scene.cycles.device = "CPU"
    return "CPU (no GPU device found)"


def set_format(settings, fmt: str, depth: str, mode: str) -> None:
    if hasattr(settings, "media_type"):
        settings.media_type = "IMAGE"
    settings.file_format = fmt
    settings.color_mode = mode
    settings.color_depth = depth
    if fmt == "OPEN_EXR":
        settings.exr_codec = "ZIP"


def collect_cat(scene_name_hint: str = "") -> tuple[list, dict]:
    """Every mesh object in collections named CAT_* -> one triangle soup in world space."""
    colls = [c for c in bpy.data.collections if c.name.startswith("CAT_")]
    objs = []
    for c in colls:
        for o in c.all_objects:
            if o.type == "MESH" and o not in objs:
                objs.append(o)
    if not objs:
        raise RuntimeError(f"no mesh objects in collections named CAT_*; collections: "
                           f"{[c.name for c in bpy.data.collections]}")
    dg = bpy.context.evaluated_depsgraph_get()
    verts, tris, mats = [], [], []
    per_obj = {}
    base = 0
    for o in objs:
        oe = o.evaluated_get(dg)
        me = oe.to_mesh()
        me.calc_loop_triangles()
        n = len(me.vertices)
        co = np.empty(n * 3, np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        mw = np.array(o.matrix_world, float)
        co = co @ mw[:3, :3].T + mw[:3, 3]
        nt = len(me.loop_triangles)
        tv = np.empty(nt * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", tv)
        mi = np.empty(nt, np.int64)
        me.loop_triangles.foreach_get("material_index", mi)
        slot_names = [(s.material.name if s.material else s.name) or "" for s in o.material_slots]
        for k in mi.tolist():
            mats.append(slot_names[k] if 0 <= k < len(slot_names) else "")
        verts.append(co)
        tris.append(tv.reshape(-1, 3) + base)
        base += n
        per_obj[o.name] = {"collection": [c.name for c in o.users_collection], "triangles": nt}
        oe.to_mesh_clear()
    return [np.concatenate(verts), np.concatenate(tris), mats], {"objects": per_obj, "collections": [c.name for c in colls]}


def build_render_mesh(V, T, mats):
    names = sorted(set(mats))
    me = bpy.data.meshes.new("__cat_render")
    me.from_pydata(V.tolist(), [], T.tolist())
    me.update()
    idx = {n: i for i, n in enumerate(names)}
    me.polygons.foreach_set("material_index", [idx[m] for m in mats])
    me.polygons.foreach_set("use_smooth", [False] * len(me.polygons))   # force flat shading
    a = me.attributes.new("tri_id", "FLOAT", "FACE")
    a.data.foreach_set("value", [float(i + 1) for i in range(len(me.polygons))])
    for _ in names:
        me.materials.append(None)
    ob = bpy.data.objects.new("__cat_render", me)
    if len(me.polygons) != len(T):
        raise RuntimeError(f"render mesh has {len(me.polygons)} faces for {len(T)} triangles")
    return ob, names


def mat_emission(name: str, rgb=None, attr: str | None = None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.0
    if attr:
        at = nt.nodes.new("ShaderNodeAttribute")
        at.attribute_type = "GEOMETRY"
        at.attribute_name = attr
        nt.links.new(at.outputs["Fac"], em.inputs["Color"])
    else:
        em.inputs["Color"].default_value = (*rgb, 1.0)
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return m


def mat_diffuse(name: str, rgb, rough: float, spec: float):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    if "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = spec
    nt.links.new(b.outputs["BSDF"], out.inputs["Surface"])
    return m


def make_world(name, light_rgb, light_strength, cam_rgb):
    w = bpy.data.worlds.new(name)
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    if cam_rgb is None:
        bg = nt.nodes.new("ShaderNodeBackground")
        bg.inputs["Color"].default_value = (*light_rgb, 1.0)
        bg.inputs["Strength"].default_value = light_strength
        nt.links.new(bg.outputs[0], out.inputs["Surface"])
        return w
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    bl = nt.nodes.new("ShaderNodeBackground")
    bl.inputs["Color"].default_value = (*light_rgb, 1.0)
    bl.inputs["Strength"].default_value = light_strength
    bc = nt.nodes.new("ShaderNodeBackground")
    bc.inputs["Color"].default_value = (*cam_rgb, 1.0)
    bc.inputs["Strength"].default_value = 1.0
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(bl.outputs[0], mix.inputs[1])
    nt.links.new(bc.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return w


def aim_matrix(loc, target) -> Matrix:
    f = Vector(target) - Vector(loc)
    rot = f.to_track_quat("-Z", "Y").to_matrix().to_4x4()
    rot.translation = Vector(loc)
    return rot


class Presenter:
    def __init__(self, args):
        self.cam_cfg = json.loads(Path(args.camera).read_text())
        self.scn_cfg = json.loads(Path(args.scene).read_text())
        self.args = args
        self.meta = {"passes_seconds": {}, "renders": 0}
        rc = self.scn_cfg["render"]
        self.rc = rc
        self.samples = int(args.samples or rc["samples"])
        self.ss = int(args.wire_ss or rc["wire"]["supersample"])
        self.line_px = float(args.wire_px or rc["wire"]["line_px"])
        self.tmp = Path(tempfile.mkdtemp(prefix="render_variant_", dir=str(_case_root() / ".cache")))

    # ---------------------------------------------------------------- scene construction
    def build(self):
        a = self.args
        bpy.ops.wm.open_mainfile(filepath=str(Path(a.blend).resolve()), load_ui=False, use_scripts=False)
        scene = bpy.context.scene
        self.scene = scene
        (V, T, mats), info = collect_cat()
        self.V, self.T, self.mats = V, T, mats
        self.meta["cat"] = info
        self.meta["triangles_rendered"] = int(len(T))
        log(f"cat: {len(T)} triangles from {list(info['objects'])}")
        # hide everything that came with the file (it is never saved)
        for o in list(scene.objects):
            o.hide_render = True
        for vl in scene.view_layers[1:]:
            vl.use = False
        scene.view_layers[0].material_override = None
        coll = bpy.data.collections.new("__presentation")
        scene.collection.children.link(coll)
        self.coll = coll
        self.cat, self.slot_names = build_render_mesh(V, T, mats)
        coll.objects.link(self.cat)

        rgb = self.scn_cfg["materials_linear_rgb"]
        sm = self.rc.get("shaded_material", {})
        rough, spec = float(sm.get("roughness", 0.6)), float(sm.get("specular_ior_level", 0.25))
        self.m_clay = mat_diffuse("__clay", rgb["clay"], rough, spec)
        self.m_beauty = {}
        unknown = []
        for n in self.slot_names:
            if n in rgb and n != "clay":
                self.m_beauty[n] = mat_diffuse("__beauty_" + n, rgb[n], rough, spec)
            else:
                unknown.append(n)
                self.m_beauty[n] = self.m_clay
        if unknown:
            log(f"WARNING material slots without a scene.json colour (rendered as clay): {unknown}")
        self.meta["material_slots"] = self.slot_names
        self.meta["material_slots_unknown"] = unknown
        self.m_white = mat_emission("__sil", rgb=(1.0, 1.0, 1.0))
        self.m_id = mat_emission("__id", attr="tri_id")

        # ground: shadow catcher at z = 0
        g = self.rc["ground"]
        half = float(g.get("size_m", 40.0)) / 2
        gm = bpy.data.meshes.new("__ground")
        gm.from_pydata([(-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0)], [], [(0, 1, 2, 3)])
        gm.materials.append(mat_diffuse("__ground_mat", self.scn_cfg["background_linear_rgb"], 1.0, 0.0))
        self.ground = bpy.data.objects.new("__ground", gm)
        self.ground.is_shadow_catcher = True
        coll.objects.link(self.ground)

        # lights (every one listed in scene.json; nothing hidden)
        self.lights = []
        created = []
        for L in self.rc["lights"]:
            ld = bpy.data.lights.new("__" + L["name"], L["type"])
            ld.energy = float(L["power_w"])
            ld.color = L.get("color", [1, 1, 1])
            if L["type"] == "AREA":
                ld.shape = L.get("shape", "DISK")
                ld.size = float(L["size"])
            elif L["type"] in ("POINT", "SPOT"):
                ld.shadow_soft_size = float(L.get("size", 0.1))
            elif L["type"] == "SUN":
                ld.angle = math.radians(float(L.get("angle_deg", 5)))
            lo = bpy.data.objects.new("__" + L["name"], ld)
            M0 = aim_matrix(L["location"], L["target"])
            coll.objects.link(lo)
            lo.matrix_world = M0
            self.lights.append((lo, M0))
            created.append({k: L[k] for k in L})
        self.meta["lights_created"] = created

        w = self.rc["world"]
        self.world_lit = make_world("__world", w["lighting_rgb"], float(w["lighting_strength"]),
                                    self.scn_cfg["background_linear_rgb"])
        self.world_black = make_world("__world_black", (0, 0, 0), 0.0, None)
        self.meta["world"] = {"lighting_rgb": w["lighting_rgb"], "lighting_strength": w["lighting_strength"],
                              "camera_ray_rgb": self.scn_cfg["background_linear_rgb"]}

        cd = bpy.data.cameras.new("__camera")
        cc = self.cam_cfg
        cd.lens = float(cc["focal_length_mm"])
        cd.sensor_width = float(cc["sensor_width_mm"])
        cd.sensor_fit = cc.get("sensor_fit", "HORIZONTAL")
        cd.clip_start = float(cc.get("clip_start", 0.05))
        cd.clip_end = float(cc.get("clip_end", 50.0))
        self.cam = bpy.data.objects.new("__camera", cd)
        coll.objects.link(self.cam)
        scene.camera = self.cam

        r = scene.render
        self.device = setup_device(scene)
        r.resolution_percentage = 100
        r.pixel_aspect_x = r.pixel_aspect_y = 1.0
        r.use_border = False
        r.use_compositing = False
        r.use_sequencer = False
        r.film_transparent = False
        r.dither_intensity = 0.0
        r.use_persistent_data = True
        r.use_motion_blur = False
        vs = scene.view_settings
        scene.display_settings.display_device = "sRGB"
        vs.view_transform = self.rc.get("view_transform", "Standard")
        vs.look = "None"
        vs.exposure = float(self.rc.get("exposure", 0.0))
        vs.gamma = 1.0
        vs.use_curve_mapping = False
        c = scene.cycles
        c.max_bounces = int(self.rc.get("max_bounces", 4))
        c.caustics_reflective = c.caustics_refractive = False
        c.sample_clamp_direct = 0.0
        c.sample_clamp_indirect = 10.0
        c.seed = 0
        c.use_animated_seed = False

        # camera check: derive from location/look_at/roll and compare with stored matrix_world
        M = camera_matrix(cc["location"], cc["look_at"], cc.get("roll_deg", 0.0))
        stored = np.array(cc["matrix_world"], float)
        err = float(np.abs(M - stored).max())
        self.meta["camera_matrix_max_abs_err"] = err
        if err > 1e-5:
            raise RuntimeError(f"camera.json matrix_world disagrees with location/look_at/roll by {err:.3g}")
        f_px = float(cc["focal_length_mm"]) / float(cc["sensor_width_mm"]) * cc["resolution"][0]
        if "f_px" in cc and abs(cc["f_px"] - f_px) > 1e-6:
            raise RuntimeError(f"camera.json f_px {cc['f_px']} != {f_px}")
        self.ref_M = M
        self.ref_res = tuple(int(x) for x in cc["resolution"])
        tgt = np.array(self.scn_cfg["orbit_target"], float)
        loc = np.array(cc["location"], float)
        self.orbit_target = tgt
        if "orbit_radius_m" in self.scn_cfg:     # explicit radius for novel views + turntable
            self.orbit_radius = float(self.scn_cfg["orbit_radius_m"])
        else:
            self.orbit_radius = float(np.linalg.norm(loc - np.array(cc["look_at"], float)))
        self.ref_az = math.degrees(math.atan2(loc[1] - tgt[1], loc[0] - tgt[0]))
        self.meta["orbit"] = {"target": tgt.tolist(), "radius_m": self.orbit_radius, "reference_az_deg": self.ref_az}

    # ---------------------------------------------------------------- per-render state
    def set_view(self, M: np.ndarray, view_az: float | None, res):
        self.cam.matrix_world = Matrix(M.tolist())
        rz = math.radians(0.0 if view_az is None else view_az - self.ref_az)
        t = Vector(self.orbit_target.tolist())
        R = Matrix.Translation(t) @ Matrix.Rotation(rz, 4, "Z") @ Matrix.Translation(-t)
        for lo, M0 in self.lights:
            lo.matrix_world = R @ M0
        self.scene.render.resolution_x, self.scene.render.resolution_y = int(res[0]), int(res[1])

    def set_mode(self, mode: str):
        s, c = self.scene, self.scene.cycles
        shaded = mode in ("clay", "beauty")
        self.ground.hide_render = not shaded
        for lo, _ in self.lights:
            lo.hide_render = not shaded
        s.world = self.world_lit if shaded else self.world_black
        mats = self.cat.data.materials
        for i, n in enumerate(self.slot_names):
            mats[i] = {"clay": self.m_clay, "beauty": self.m_beauty.get(n), "id": self.m_id, "silhouette": self.m_white}[mode]
        if shaded:
            c.samples = self.samples
            thr = self.rc.get("adaptive_threshold")
            c.use_adaptive_sampling = bool(thr)
            if thr:
                c.adaptive_threshold = float(thr)
            den = str(self.rc.get("denoiser", "NONE")).upper()
            c.use_denoising = den != "NONE"
            if c.use_denoising:
                c.denoiser = den
                c.denoising_input_passes = "RGB_ALBEDO_NORMAL"
            c.pixel_filter_type = "BLACKMAN_HARRIS"
            c.filter_width = 1.5
        else:
            c.samples = 1
            c.use_adaptive_sampling = False
            c.use_denoising = False
            c.pixel_filter_type = "BOX"
            c.filter_width = 0.01
        self.mode = mode

    def render_exr(self, tag: str) -> np.ndarray:
        """Render the current state; returns (H, W, 4) float32 scene-linear, top row first."""
        p = self.tmp / f"{tag}.exr"
        set_format(self.scene.render.image_settings, "OPEN_EXR", "32", "RGBA")
        self.scene.render.image_settings.color_management = "FOLLOW_SCENE"
        self.scene.render.filepath = str(p)
        self.scene.render.use_file_extension = False
        bpy.ops.render.render(write_still=True)
        self.meta["renders"] += 1
        img = bpy.data.images.load(str(p), check_existing=False)
        img.colorspace_settings.name = "Non-Color"
        W, H = img.size
        buf = np.empty(W * H * 4, np.float32)
        img.pixels.foreach_get(buf)
        bpy.data.images.remove(img)
        p.unlink()
        return buf.reshape(H, W, 4)[::-1]

    def render_ids(self, res, tag) -> np.ndarray:
        rgba = self.render_exr(tag)
        ids = np.rint(rgba[:, :, 0]).astype(np.int64)
        if np.abs(rgba[:, :, 0] - ids).max() > 1e-3 or ids.max() > len(self.T):
            raise RuntimeError("id pass is not integer-exact")
        return ids

    # ---------------------------------------------------------------- jobs
    def timed(self, key, fn, *a):
        t0 = time.time()
        r = fn(*a)
        self.meta["passes_seconds"][key] = round(self.meta["passes_seconds"].get(key, 0.0) + time.time() - t0, 3)
        return r

    def run_reference(self, passes, out: Path):
        W, H = self.ref_res
        self.set_view(self.ref_M, None, (W, H))
        if "silhouette" in passes:
            def sil():
                self.set_mode("silhouette")
                s = self.scene
                set_format(s.render.image_settings, "PNG", "8", "BW")
                s.render.image_settings.color_management = "FOLLOW_SCENE"
                s.render.image_settings.compression = 15
                s.render.filepath = str(out / "ref_silhouette.png")
                s.render.use_file_extension = False
                bpy.ops.render.render(write_still=True)
                self.meta["renders"] += 1
                # check it is binary, and identical to (id > 0) at pixel centres
                self.set_mode("id")
                ids = self.render_ids((W, H), "sil_ids")
                img = bpy.data.images.load(str(out / "ref_silhouette.png"), check_existing=False)
                img.colorspace_settings.name = "Non-Color"
                buf = np.empty(W * H * 4, np.float32)
                img.pixels.foreach_get(buf)
                bpy.data.images.remove(img)
                g = np.rint(buf.reshape(H, W, 4)[::-1, :, 0] * 255).astype(int)
                grey = int(((g > 0) & (g < 255)).sum())
                agree = float(((g >= 128) == (ids > 0)).mean())
                self.meta["silhouette"] = {"grey_pixels": grey, "grey_fraction_of_mask": grey / max(1, int((g >= 128).sum())),
                                           "white_pixels": int((g >= 128).sum()),
                                           "agreement_with_id_pass": agree,
                                           "visible_triangles_ref_view": int(len(np.unique(ids[ids > 0])))}
                log(f"silhouette: {self.meta['silhouette']}")
            self.timed("silhouette", sil)
        need_clay = "clay" in passes or "wire" in passes
        if need_clay:
            self.set_mode("clay")
            clay = self.timed("clay", self.render_exr, "ref_clay")
            if "clay" in passes:
                write_png(out / "ref_clay.png", srgb_encode(clay[:, :, :3]))
        if "wire" in passes:
            def wire():
                self.set_mode("id")
                self.scene.render.resolution_x, self.scene.render.resolution_y = W * self.ss, H * self.ss
                ids = self.render_ids(None, "ref_ids")
                write_png(out / "ref_wire.png", srgb_encode(self.composite(clay[:, :, :3], ids)))
                self.scene.render.resolution_x, self.scene.render.resolution_y = W, H
            self.timed("wire", wire)
        if "beauty" in passes:
            def beauty():
                self.set_mode("beauty")
                write_png(out / "ref_beauty.png", srgb_encode(self.render_exr("ref_beauty")[:, :, :3]))
            self.timed("beauty", beauty)

    def composite(self, clay_rgb, ids):
        a = wire_alpha(ids, self.ss, self.line_px)[:, :, None]
        wc = np.array(self.scn_cfg["wire_color_linear_rgb"], np.float32)
        return np.clip(clay_rgb, 0.0, 1.0) * (1 - a) + wc * a   # display-range clay, then lines

    def orbit_matrix(self, az, el):
        a, e = math.radians(az), math.radians(el)
        t = self.orbit_target
        loc = t + self.orbit_radius * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        return camera_matrix(loc, t, 0.0)

    def view_list(self):
        out = []
        for name, v in self.scn_cfg["views"].items():
            if v.get("straight_down"):
                out.append((name, top_camera_matrix(self.orbit_target, self.orbit_radius, v.get("image_up", [1, 0, 0])), 180.0))
            else:
                out.append((name, self.orbit_matrix(v["az_deg"], v["el_deg"]), float(v["az_deg"])))
        return out

    def render_set(self, cams, res, want_beauty: bool):
        """cams: [(name, M, az)]. Returns {name: (clay_wire_rgb, clay_rgb, beauty_rgb|None)} linear."""
        W, H = res
        clay, ids, beauty = {}, {}, {}
        self.set_mode("clay")
        for n, M, az in cams:
            self.set_view(M, az, res)
            clay[n] = self.render_exr("c_" + n)[:, :, :3]
        self.set_mode("id")
        for n, M, az in cams:
            self.set_view(M, az, (W * self.ss, H * self.ss))
            ids[n] = self.render_ids(None, "i_" + n)
        if want_beauty:
            self.set_mode("beauty")
            for n, M, az in cams:
                self.set_view(M, az, res)
                beauty[n] = self.render_exr("b_" + n)[:, :, :3]
        return {n: (self.composite(clay[n], ids[n]), clay[n], beauty.get(n)) for n, _, _ in cams}

    def run_views(self, out: Path):
        res = tuple(self.scn_cfg["view_resolution"])
        t0 = time.time()
        views = self.view_list()
        res_imgs = self.render_set(views, res, want_beauty=False)
        for n, (cw, c, _) in res_imgs.items():
            write_png(out / f"view_{n}_clay.png", srgb_encode(c))
            write_png(out / f"view_{n}_wire.png", srgb_encode(cw))
        self.meta["passes_seconds"]["views"] = round(time.time() - t0, 3)
        self.meta["views"] = {n: {"camera_location": M[:3, 3].round(5).tolist(), "light_rig_az_deg": az}
                              for n, M, az in views}

    def run_turntable(self, out: Path, keep_frames: bool):
        tt = self.scn_cfg["turntable"]
        res = tuple(tt["resolution"])
        n, step, el, fps = int(tt["frames"]), float(tt["step_deg"]), float(tt["el_deg"]), int(tt["fps"])
        t0 = time.time()
        cams = [(f"{k:03d}", self.orbit_matrix(self.ref_az + k * step, el), self.ref_az + k * step) for k in range(n)]
        fdir = out / "turntable_frames"
        fdir.mkdir(parents=True, exist_ok=True)
        # chunked so memory stays small at any frame count
        for i in range(0, n, 12):
            chunk = cams[i:i + 12]
            imgs = self.render_set(chunk, res, want_beauty=True)
            for name, (cw, _c, b) in imgs.items():
                write_png(fdir / f"clay_wire_{name}.png", srgb_encode(cw))
                write_png(fdir / f"beauty_{name}.png", srgb_encode(b))
            log(f"turntable frames {i}..{i + len(chunk) - 1} done ({time.time() - t0:.1f}s)")
        t1 = time.time()
        for kind in ("clay_wire", "beauty"):
            mp4 = out / f"turntable_{kind}.mp4"
            cmd = [FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(fdir / f"{kind}_%03d.png"),
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
                   "-movflags", "+faststart", str(mp4)]
            subprocess.run(cmd, check=True)
        self.meta["passes_seconds"]["turntable_render"] = round(t1 - t0, 3)
        self.meta["passes_seconds"]["turntable_encode"] = round(time.time() - t1, 3)
        self.meta["turntable"] = {"frames": n, "step_deg": step, "el_deg": el, "start_az_deg": self.ref_az,
                                  "resolution": list(res), "fps": fps, "codec": "libx264 yuv420p crf18",
                                  "frames_kept": keep_frames}
        if not keep_frames:
            shutil.rmtree(fdir)


def main() -> int:
    t_start = time.time()
    argv = script_args()
    ap = argparse.ArgumentParser(prog="render_variant.py")
    ap.add_argument("--blend", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", default="silhouette,clay,wire,beauty")
    ap.add_argument("--views", action="store_true")
    ap.add_argument("--turntable", action="store_true")
    ap.add_argument("--keep-frames", action="store_true", help="keep turntable frame PNGs")
    ap.add_argument("--camera", default=str(_case_root() / "contract" / "camera.json"))
    ap.add_argument("--scene", default=str(_case_root() / "contract" / "scene.json"))
    ap.add_argument("--samples", type=int, default=None, help="debug override (recorded in meta)")
    ap.add_argument("--wire-ss", type=int, default=None, help="debug override of wire supersample")
    ap.add_argument("--wire-px", type=float, default=None, help="debug override of wire line width")
    args = ap.parse_args(argv)

    # GPU lock first: held for the whole life of this rendering process (CONTRACT section 1)
    gl = _load_gpulock()
    waited = gl.acquire_gpu_lock(label=f"render_variant {Path(args.blend).name}")
    t_locked = time.time()

    passes = [p.strip() for p in args.passes.split(",") if p.strip()]
    bad = set(passes) - {"silhouette", "clay", "wire", "beauty"}
    if bad:
        raise SystemExit(f"unknown passes {bad}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pr = Presenter(args)
    try:
        pr.build()
        log(f"device {pr.device}; samples {pr.samples}; wire ss {pr.ss} line {pr.line_px}px")
        pr.run_reference(passes, out)
        if args.views:
            pr.run_views(out)
        if args.turntable:
            pr.run_turntable(out, args.keep_frames)
    finally:
        shutil.rmtree(pr.tmp, ignore_errors=True)
    m = pr.meta
    m.update({
        "engine": pr.scene.render.engine, "device": pr.device, "samples_shaded": pr.samples,
        "samples_silhouette_id": 1, "filter_silhouette_id": "BOX 0.01 px",
        "filter_shaded": "BLACKMAN_HARRIS 1.5 px", "denoiser": str(pr.rc.get("denoiser", "NONE")),
        "adaptive_threshold": pr.rc.get("adaptive_threshold"),
        "view_transform": pr.scene.view_settings.view_transform, "dither": pr.scene.render.dither_intensity,
        "wire": {"supersample": pr.ss, "line_px": pr.line_px, "color_linear_rgb": pr.scn_cfg["wire_color_linear_rgb"]},
        "samples_overridden": args.samples is not None,
        "blender_version": bpy.app.version_string,
        "blend": str(Path(args.blend).resolve()), "blend_sha256": sha256(Path(args.blend)),
        "camera_json": str(Path(args.camera).resolve()), "camera_sha256": sha256(Path(args.camera)),
        "camera_status": pr.cam_cfg.get("status"),
        "scene_json": str(Path(args.scene).resolve()), "scene_sha256": sha256(Path(args.scene)),
        "scene_status": pr.scn_cfg.get("status"),
        "passes": passes, "views": m.get("views"), "gpu_lock_wait_s": round(waited, 3),
        "seconds_total_after_lock": round(time.time() - t_locked, 3),
        "seconds_total": round(time.time() - t_start, 3),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    (out / "render_meta.json").write_text(json.dumps(m, indent=2))
    log(f"done in {m['seconds_total']:.1f}s; per pass {m['passes_seconds']}")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    sys.exit(rc)
