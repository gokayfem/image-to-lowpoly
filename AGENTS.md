# AGENTS.md

Guidance for coding agents (Claude Code, Codex, Cursor, Copilot, Gemini) working on this repository. To use the
skill rather than develop it, see the README.

## Layout

```text
skills/image-to-lowpoly/        the skill (this is what gets installed)
  SKILL.md                      entry point: when to use it, workflow, hard rules, reference routing
  references/                   contract, budget-worker, counting, lessons
  scripts/catlib/               importable library (numpy only in the modules Blender imports)
    paths       where the code and the case live (--case / LOWPOLY_CASE / cwd); finds Blender
    camera      the one pinhole camera and pixel convention; raster: CPU silhouettes and ids
    meshio      the catmesh-1 format, landmark attachment; metrics: the official numbers
    topo        flip, split, collapse, weld, ring bridging, caps; refine: the shared optimiser
    skeleton, proxy, headkit    the parametric proxy, its SDF lookups, the symmetric head kit
    provenance, gpulock         hashes against stale outputs; the single render lock
  scripts/*.py                  CLIs (preflight, fit, proxy, baselines, refine, assemble, validate, evaluate,
                                render, finalize, aggregate, comparison, critic packet, head kit)
  scripts/timelapse/            build timelapse over the photo (from real stages and snapshots)
examples/cat/                   the worked case, laid out as a case directory
tests/                          smoke.py (Python only) and the original suites (some need Blender)
adapters/                       copy-in prompts for tools without skill support
gallery/                        README images (clay, wire, a timelapse frame)
.claude-plugin/                 Claude Code plugin + marketplace manifests
```

## Commands

```bash
python3 tests/smoke.py                                       # compile, import, four-way counts of the example
python3 tests/test_headkit.py                                # head kit welds (numpy)
python3 tests/test_refine.py --skip-blender                  # topology edits + refiner on a synthetic target
python3 tests/test_tooling.py                                # needs Blender
python3 skills/image-to-lowpoly/scripts/evaluate.py --case examples/cat --help
```

Blender scripts run as `blender -b --factory-startup --python scripts/<name>.py -- ...`; `evaluate.py`,
`validate_variant.py` and `finalize_variant.py` start Blender themselves.

## Rules for changes

- Subject-agnostic where it already is: counting, assembly, validation, metrics, rendering and the refiner
  must not learn anything about cats. The cat-specific parts are named in `references/contract.md` (region,
  material and landmark lists in `catlib/meshio.py`, the proxy template in `fit_camera_proxy.py` /
  `build_proxy.py`, the head kit) and in `examples/cat/`.
- No machine paths: the case comes from `--case` / `LOWPOLY_CASE`, Blender from `$BLENDER` or PATH, ffmpeg
  from PATH.
- The evaluation code is hashed into every `metrics.json` (`catlib/provenance.py`). Changing any of those
  files makes earlier numbers incomparable: re-score every budget before comparing.
- Honesty in code: a failed count, render or validation stops and writes nothing that could pass for a
  result. Nothing is padded to reach a count.
- No outputs in the repository except the worked example's four final meshes and the gallery images.
- No secrets in files or logs.
