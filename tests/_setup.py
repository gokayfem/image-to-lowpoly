"""Shared setup for every test (stdlib only, so Blender's Python can import it too).

Puts the skill's scripts folder (which holds `catlib/`) on sys.path and points
`LOWPOLY_CASE` at the worked example `examples/cat` unless the caller chose another case.
Tests that write files write them into a temp dir or into the case's `.cache/`
(git-ignored); none of them touches the shipped example data.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"
SCRIPTS = REPO / "skills" / "image-to-lowpoly" / "scripts"
os.environ.setdefault("LOWPOLY_CASE", str(REPO / "examples" / "cat"))
CASE = Path(os.environ["LOWPOLY_CASE"]).resolve()
for p in (str(SCRIPTS), str(TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)
