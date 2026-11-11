"""Provenance hashes against stale outputs (CONTRACT §7).  stdlib only."""
from __future__ import annotations

import hashlib
from pathlib import Path

from . import paths


def sha256_file(path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Evaluation code only (CONTRACT §7).  Optimiser / construction modules (refine, topo,
# skeleton, proxy, ...) are deliberately excluded so adding or editing them does not make
# otherwise identical evaluations incomparable; their outputs are hashed via the mesh.
EVAL_CATLIB = ("__init__", "paths", "camera", "raster", "geom", "meshio", "metrics", "overlay",
               "provenance", "gpulock")


def tooling_files() -> list[Path]:
    """The explicit evaluation-code list: catlib/{EVAL_CATLIB}.py + the shared scripts that
    affect official numbers (paths.SHARED_SCRIPTS).  Exactly these files are hashed."""
    files = [paths.CATLIB / f"{n}.py" for n in EVAL_CATLIB if (paths.CATLIB / f"{n}.py").is_file()]
    files += [paths.SCRIPTS / n for n in paths.SHARED_SCRIPTS if (paths.SCRIPTS / n).is_file()]
    return files


def tooling_hash() -> tuple[str, list[str]]:
    h = hashlib.sha256()
    names = []
    for f in tooling_files():
        # Named as in the original layout (catlib/x.py, scripts/x.py) so the list reads the same
        # wherever the skill is installed; the hash covers the file contents.
        rel = ("catlib/" if f.parent == paths.CATLIB else "scripts/") + f.name
        names.append(rel)
        h.update(rel.encode() + b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest(), names


def _rel(p) -> str:
    """A path relative to the case directory when it is inside it (no machine paths in outputs)."""
    p = Path(p).resolve()
    try:
        return p.relative_to(paths.ROOT).as_posix()
    except ValueError:
        return str(p)


def provenance(mesh_path, derived_dir=None, camera_path=None, scene_path=None, proxy_path=None) -> dict:
    d = paths.derived_paths(derived_dir)
    th, files = tooling_hash()
    return {
        "reference_sha256": sha256_file(paths.REFERENCE_PNG),
        "mask_sha256": sha256_file(d["mask"]),
        "landmarks_sha256": sha256_file(d["landmarks"]),
        "regions_sha256": sha256_file(d["regions"]),
        "camera_sha256": sha256_file(camera_path or paths.CAMERA_JSON),
        "scene_sha256": sha256_file(scene_path or paths.SCENE_JSON),
        "mesh_sha256": sha256_file(mesh_path),
        "proxy_sha256": sha256_file(proxy_path or paths.PROXY_NPZ),   # secondary metrics depend on it
        "tooling_hash": th,
        "tooling_files": files,
        "derived_dir": _rel(d["dir"]),
    }
