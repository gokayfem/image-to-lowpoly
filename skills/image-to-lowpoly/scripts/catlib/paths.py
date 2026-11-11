"""Standard paths: where the code lives and where the case lives.

numpy-free; importable from the analysis Python and from Blender's bundled Python.

The code (this package and the CLIs next to it) lives in the skill's `scripts/` folder.
A *case* is a separate directory holding one subject's reference, contract, proxy,
variants and outputs (layout in references/contract.md). The case directory is resolved,
in this order, from:

  1. `--case DIR` (or `--case=DIR`) anywhere on the command line. It is removed from
     sys.argv before the script's own argument parser runs, so every CLI accepts it;
  2. the environment variable `LOWPOLY_CASE`;
  3. the current working directory.

The resolved directory is written back to `LOWPOLY_CASE` so that child processes (Blender,
the refiner) work on the same case.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _pop_case_arg() -> str | None:
    argv = sys.argv
    for i, a in enumerate(list(argv)):
        if a == "--case" and i + 1 < len(argv):
            val = argv[i + 1]
            del argv[i:i + 2]
            return val
        if a.startswith("--case="):
            del argv[i]
            return a.split("=", 1)[1]
    return None


CODE = Path(__file__).resolve().parent.parent          # skills/image-to-lowpoly/scripts
_case = _pop_case_arg() or os.environ.get("LOWPOLY_CASE") or os.getcwd()
ROOT = Path(_case).expanduser().resolve()               # the case directory
os.environ["LOWPOLY_CASE"] = str(ROOT)

CONTRACT = ROOT / "contract"
CAMERA_JSON = CONTRACT / "camera.json"
SCENE_JSON = CONTRACT / "scene.json"
REFERENCE = ROOT / "reference"
REFERENCE_PNG = Path(os.environ.get("LOWPOLY_REFERENCE", str(REFERENCE / "reference.png")))
if not REFERENCE_PNG.is_file() and (REFERENCE / "cat.png").is_file():
    REFERENCE_PNG = REFERENCE / "cat.png"               # the worked example's file name
DERIVED = REFERENCE / "derived"
MASK_PNG = DERIVED / "mask.png"
LANDMARKS_JSON = DERIVED / "landmarks.json"
REGIONS_JSON = DERIVED / "regions.json"
REFERENCE_BBOX_JSON = DERIVED / "reference_bbox.json"
PROXY = ROOT / "proxy"
PROXY_NPZ = PROXY / "proxy_mesh.npz"
HEAD = ROOT / "head"
SCRIPTS = CODE
CATLIB = CODE / "catlib"
VARIANTS = ROOT / "variants"
BASELINES = ROOT / "baselines"
OUTPUTS = ROOT / "outputs"
CACHE = ROOT / ".cache"

# The analysis Python that child processes use (the refiner, evaluate.py).
VENV_PYTHON = Path(os.environ.get("LOWPOLY_PYTHON") or sys.executable)


def _find_blender() -> Path:
    for k in ("BLENDER", "CAT_BLENDER"):
        if os.environ.get(k):
            return Path(os.environ[k])
    w = shutil.which("blender")
    if w:
        return Path(w)
    return Path("/Applications/Blender.app/Contents/MacOS/Blender")


BLENDER = _find_blender()

# Shared scripts whose code affects official numbers (part of tooling_hash, contract §7).
SHARED_SCRIPTS = ("assemble_variant.py", "validate_variant.py", "evaluate.py", "render_variant.py")


def derived_paths(derived_dir: Path | str | None = None) -> dict:
    d = Path(derived_dir) if derived_dir else DERIVED
    return {
        "dir": d,
        "mask": d / "mask.png",
        "landmarks": d / "landmarks.json",
        "regions": d / "regions.json",
        "bbox": d / "reference_bbox.json",
    }
