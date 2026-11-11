"""Build outputs/comparison.blend: the four final variants side by side + a separate presentation set.

    python3 scripts/make_comparison_blend.py [--variants-root variants] [--out outputs] \
        [--spacing 0.75]

(The same file is also the Blender-side script; the venv entry point launches
`Blender -b --factory-startup --python scripts/make_comparison_blend.py -- --stage build|verify`.)

Contents of comparison.blend
  CAT_100, CAT_200, CAT_500, CAT_1000   one collection per variant holding the parent empty
                                        `CAT_<N>_root` and the single mesh object `cat_<N>`,
                                        appended (not linked) from variants/<N>/final/cat_<N>.blend.
                                        The mesh keeps an identity LOCAL transform; only the parent
                                        empty is offset.  Custom properties on the empty: budget,
                                        validated_triangles, offset_m, spacing_m, offset_direction,
                                        source_blend, mesh_sha256.
  PRESENTATION                          presentation-only objects from contract/: the shared
                                        reference camera (exact camera.json matrix), an overview
                                        camera on the same line of sight, the ground plane and the
                                        lights of scene.json.  Never part of any cat count.

The cats are spaced along the reference camera's horizontal right vector so that the
overview camera sees them side by side.  To look at one variant through the exact reference
camera, set its `CAT_<N>_root` location to (0, 0, 0).

After saving, the verify stage reopens the file in a NEW Blender process and sums loop
triangles of the EVALUATED meshes of every object in each CAT_<N> collection (not
len(polygons)); it fails if any differs from that variant's validated count, if a cat object
has a non-identity local transform, or if any mesh lies outside CAT_* / PRESENTATION.
No rendering, so no GPU lock.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the folder that holds catlib/
from catlib import paths as _paths  # noqa: E402  (resolves the case: --case / LOWPOLY_CASE / cwd)
ROOT = _paths.ROOT
BUDGETS = (100, 200, 500, 1000)

try:
    import bpy  # noqa: F401
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


# =========================================================================== Blender side
def _script_args():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def _aim_quat(loc, target):
    from mathutils import Vector
    d = Vector(target) - Vector(loc)
    return d.to_track_quat("-Z", "Y")


def blender_build(spec: dict) -> None:
    import bpy
    from mathutils import Matrix, Vector
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    cam_cfg = json.loads(Path(spec["camera_json"]).read_text())
    scn_cfg = json.loads(Path(spec["scene_json"]).read_text())
    M = Matrix(cam_cfg["matrix_world"])
    right = Vector((M[0][0], M[1][0], 0.0)).normalized()
    spacing = float(spec["spacing"])
    n = len(spec["variants"])
    for i, v in enumerate(spec["variants"]):
        N = v["budget"]
        coll = bpy.data.collections.new(f"CAT_{N}")
        scene.collection.children.link(coll)
        with bpy.data.libraries.load(v["blend"], link=False) as (src, dst):
            names = list(src.objects)
            dst.objects = names
        objs = [o for o in dst.objects if o is not None]
        meshes = [o for o in objs if o.type == "MESH"]
        if len(objs) != 1 or len(meshes) != 1:
            raise RuntimeError(f"{v['blend']}: expected exactly one mesh object, got {[(o.name, o.type) for o in objs]}")
        ob = meshes[0]
        coll.objects.link(ob)
        off = right * ((i - (n - 1) / 2) * spacing)
        root = bpy.data.objects.new(f"CAT_{N}_root", None)
        root.empty_display_type = "PLAIN_AXES"
        root.empty_display_size = 0.1
        coll.objects.link(root)
        root.location = off
        root["budget"] = N
        root["validated_triangles"] = int(v["expected"])
        root["offset_m"] = [float(x) for x in off]
        root["spacing_m"] = spacing
        root["offset_direction"] = [float(x) for x in right]
        root["source_blend"] = v["blend"]
        root["mesh_sha256"] = v.get("mesh_sha256") or ""
        root["note"] = "parent empty only; the cat mesh keeps an identity local transform"
        ob.parent = root
        ob.matrix_parent_inverse = Matrix.Identity(4)
        ob.matrix_basis = Matrix.Identity(4)

    pres = bpy.data.collections.new("PRESENTATION")
    scene.collection.children.link(pres)
    pres["presentation_only"] = True
    pres["note"] = "camera / overview camera / ground / lights from contract/; not cat geometry, never counted"
    cd = bpy.data.cameras.new("reference_camera")
    cd.lens = float(cam_cfg["focal_length_mm"])
    cd.sensor_width = float(cam_cfg["sensor_width_mm"])
    cd.sensor_fit = cam_cfg.get("sensor_fit", "HORIZONTAL")
    cd.clip_start = float(cam_cfg.get("clip_start", 0.05))
    cd.clip_end = float(cam_cfg.get("clip_end", 50.0))
    cam = bpy.data.objects.new("reference_camera", cd)
    pres.objects.link(cam)
    cam.matrix_world = M
    cam["camera_status"] = cam_cfg.get("status", "")
    look_at = Vector(cam_cfg["look_at"])
    loc = Vector(cam_cfg["location"])
    od = cd.copy()
    od.name = "overview_camera"
    ov = bpy.data.objects.new("overview_camera", od)
    pres.objects.link(ov)
    k = float(spec["overview_factor"])
    ov.location = look_at + (loc - look_at) * k
    ov.rotation_mode = "QUATERNION"
    ov.rotation_quaternion = _aim_quat(ov.location, look_at)
    scene.camera = ov
    rc = scn_cfg["render"]
    half = float(rc["ground"].get("size_m", 4.0)) / 2
    gm = bpy.data.meshes.new("ground")
    gm.from_pydata([(-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0)], [], [(0, 1, 2, 3)])
    gmat = bpy.data.materials.new("ground_presentation")
    gmat.diffuse_color = (*scn_cfg["background_linear_rgb"], 1.0)
    gm.materials.append(gmat)
    g = bpy.data.objects.new("ground", gm)
    pres.objects.link(g)
    for L in rc["lights"]:
        ld = bpy.data.lights.new(L["name"], L["type"])
        ld.energy = float(L["power_w"])
        ld.color = L.get("color", [1, 1, 1])
        if L["type"] == "AREA":
            ld.shape = L.get("shape", "DISK")
            ld.size = float(L["size"])
        lo = bpy.data.objects.new(L["name"], ld)
        pres.objects.link(lo)
        lo.location = L["location"]
        lo.rotation_mode = "QUATERNION"
        lo.rotation_quaternion = _aim_quat(L["location"], L["target"])
    w = bpy.data.worlds.new("presentation_world")
    w.color = tuple(scn_cfg["background_linear_rgb"])
    scene.world = w
    scene.render.resolution_x, scene.render.resolution_y = cam_cfg["resolution"]
    scene.render.engine = "CYCLES"
    scene["note"] = ("Comparison of the four triangle-budget cats. Counts: loop triangles of evaluated meshes per "
                     "CAT_<N> collection. PRESENTATION holds presentation objects only.")
    bpy.ops.wm.save_as_mainfile(filepath=spec["out_blend"], compress=False)
    print("BUILD_OK", spec["out_blend"])


def blender_verify(spec: dict) -> int:
    import bpy
    from mathutils import Matrix
    bpy.ops.wm.open_mainfile(filepath=spec["out_blend"])
    dg = bpy.context.evaluated_depsgraph_get()
    expected = {f"CAT_{v['budget']}": int(v["expected"]) for v in spec["variants"]}
    bad = []
    counted = set()
    report = {}
    for cname, exp in expected.items():
        coll = bpy.data.collections.get(cname)
        if coll is None:
            bad.append(f"{cname}: collection missing")
            continue
        tris, objs = 0, []
        for ob in coll.all_objects:
            counted.add(ob.name)
            if ob.type in ("MESH", "CURVE", "SURFACE", "META", "FONT", "CURVES", "POINTCLOUD", "VOLUME", "GREASEPENCIL"):
                ev = ob.evaluated_get(dg)
                me = ev.to_mesh()
                if me is None:
                    continue
                me.calc_loop_triangles()
                n = len(me.loop_triangles)
                ev.to_mesh_clear()
                tris += n
                objs.append((ob.name, ob.type, n))
                if ob.matrix_basis != Matrix.Identity(4):
                    bad.append(f"{cname}: {ob.name} local transform is not identity")
                if ob.modifiers:
                    bad.append(f"{cname}: {ob.name} has modifiers {[m.type for m in ob.modifiers]}")
        report[cname] = {"evaluated_loop_triangles": tris, "expected": exp, "objects": objs}
        status = "OK" if tris == exp else "MISMATCH"
        print(f"{cname}: evaluated loop triangles {tris} (validated {exp}) {status}  objects {objs}")
        if tris != exp:
            bad.append(f"{cname}: {tris} evaluated loop triangles != validated {exp}")
    pres = bpy.data.collections.get("PRESENTATION")
    pres_objs = [(o.name, o.type) for o in pres.all_objects] if pres else []
    print(f"PRESENTATION (not counted): {pres_objs}")
    for ob in bpy.data.objects:
        if ob.name in counted:
            continue
        in_pres = pres is not None and ob.name in pres.all_objects
        if not in_pres:
            bad.append(f"object {ob.name} ({ob.type}) is in neither a CAT_ collection nor PRESENTATION")
        elif ob.type == "MESH" and ob.name != "ground":
            bad.append(f"unexpected mesh {ob.name} in PRESENTATION")
    Path(spec["report"]).write_text(json.dumps({"collections": report, "presentation": pres_objs,
                                                "problems": bad}, indent=1))
    for b in bad:
        print("VERIFY FAILURE:", b)
    print("VERIFY_OK" if not bad else "VERIFY_FAILED")
    return 0 if not bad else 1


# =========================================================================== venv side
def main_venv(argv=None) -> int:
    import argparse
    from catlib import paths
    ap = argparse.ArgumentParser(description="Build and verify outputs/comparison.blend.")
    ap.add_argument("--variants-root", type=Path, default=paths.VARIANTS)
    ap.add_argument("--out", type=Path, default=paths.OUTPUTS)
    ap.add_argument("--spacing", type=float, default=0.75, help="metres between cats along the camera's right vector")
    ap.add_argument("--overview-factor", type=float, default=3.4,
                    help="overview camera distance as a multiple of the reference camera distance")
    a = ap.parse_args(argv)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    variants = []
    for N in BUDGETS:
        fin = (a.variants_root / str(N) / "final").resolve()
        blend, vp, mp = fin / f"cat_{N}.blend", fin / "validation.json", fin / "metrics.json"
        if not blend.is_file() or not vp.is_file():
            print(f"make_comparison_blend: {fin} lacks cat_{N}.blend / validation.json", file=sys.stderr)
            return 2
        c = json.loads(vp.read_text())["counts"]
        if not c.get("all_agree") or c.get("final_triangles") is None or c["final_triangles"] > N:
            print(f"make_comparison_blend: budget {N} counts not validated: {c}", file=sys.stderr)
            return 2
        sha = json.loads(mp.read_text())["provenance"]["mesh_sha256"] if mp.is_file() else ""
        variants.append({"budget": N, "blend": str(blend), "expected": c["final_triangles"], "mesh_sha256": sha})
    spec = {"variants": variants, "camera_json": str(paths.CAMERA_JSON), "scene_json": str(paths.SCENE_JSON),
            "spacing": a.spacing, "overview_factor": a.overview_factor,
            "out_blend": str(out / "comparison.blend"), "report": str(out / "comparison_blend_counts.json")}
    spec_path = out / ".comparison_blend_spec.json"
    spec_path.write_text(json.dumps(spec, indent=1))
    if Path(spec["out_blend"]).exists():
        Path(spec["out_blend"]).unlink()
    rc_all = 0
    for stage in ("build", "verify"):
        cmd = [str(paths.BLENDER), "-b", "--factory-startup", "--python", str(Path(__file__).resolve()),
               "--", "--stage", stage, "--spec", str(spec_path)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        lines = [l for l in p.stdout.splitlines() if l.startswith(("CAT_", "PRESENTATION", "VERIFY", "BUILD"))]
        print(f"[{stage}] exit {p.returncode}")
        print("\n".join(lines))
        ok_marker = "BUILD_OK" if stage == "build" else "VERIFY_OK"
        if p.returncode != 0 or ok_marker not in p.stdout:
            print(p.stdout[-3000:], p.stderr[-3000:], file=sys.stderr)
            rc_all = 1
            break
    spec_path.unlink(missing_ok=True)
    if rc_all == 0:
        print(f"wrote {spec['out_blend']} and {spec['report']}")
    return rc_all


def main_blender() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "verify"], required=True)
    ap.add_argument("--spec", required=True)
    a = ap.parse_args(_script_args())
    spec = json.loads(Path(a.spec).read_text())
    if a.stage == "build":
        blender_build(spec)
        return 0
    return blender_verify(spec)


if __name__ == "__main__":
    if IN_BLENDER:
        try:
            rc = main_blender()
        except Exception:
            import traceback
            traceback.print_exc()
            rc = 1
        sys.exit(rc)
    sys.exit(main_venv())
