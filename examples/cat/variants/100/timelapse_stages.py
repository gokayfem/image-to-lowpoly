"""Write variants/100/timelapse/stages.json (CONTRACT §13) from the files reproduce_v3.sh made,
and verify the last refine snapshot reproduces the v3_welded checkpoint."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
TL = HERE / "timelapse"
stages = []
for f in sorted((TL / "construct").glob("*.json")):
    lab = json.loads(f.read_text()).get("label", f.stem)
    stages.append({"label": lab if f.stem != "09_built" else "built mesh (100 tris, closed)",
                   "kind": "construct", "mesh": f"construct/{f.name}"})
for r, u in (("r1", "u1"), ("r2", "u2"), ("r3", "u3"), ("r4", None)):
    stages.append({"label": f"refine {r}", "kind": "refine", "base_mesh": f"refine/{r}/base.json",
                   "snapshots": f"refine/{r}/snaps/"})
    if u:
        stages.append({"label": f"edge-flip fold repair {u}", "kind": "edit",
                       "before": f"edit/{u}_before.json", "mesh": f"edit/{u}_after.json"})
final = HERE / "checkpoints" / "v3_welded" / "mesh.json"
stages.append({"label": "final", "kind": "final", "mesh": "../checkpoints/v3_welded/mesh.json"})
snaps = sorted((TL / "refine" / "r4" / "snaps").glob("snap_*.json"), key=lambda p: int(p.stem.split("_")[1]))
last = np.array(json.loads(snaps[-1].read_text())["vertices"])
fin = np.array(json.loads(final.read_text())["vertices"])
diff = float(np.abs(last - fin).max())
(TL / "stages.json").write_text(json.dumps({"budget": 100, "final_checkpoint": "v3_welded",
                                            "last_snapshot_max_abs_diff_m": diff, "stages": stages}, indent=1))
print("stages:", len(stages), "last snapshot vs final max diff:", diff)
sys.exit(0 if diff < 1e-6 else 1)
