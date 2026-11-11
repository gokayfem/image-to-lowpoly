"""2x2 turntable mosaics of the four final variants, frame-synchronised.

    python3 scripts/make_turntable_mosaic.py [--variants-root variants] [--out outputs] \
        [--allow-missing]

Reads variants/<N>/final/renders/turntable_{clay_wire,beauty}.mp4 (72 frames, 768^2, 24 fps,
rendered by scripts/render_variant.py via finalize_variant.py).  Writes

  <out>/turntable_all_clay_wire.mp4, <out>/turntable_all_beauty.mp4   2x2, H.264 yuv420p
  <out>/turntables/turntable_<N>_{clay_wire,beauty}.mp4                 per-variant copies

Each tile is labelled (drawn with PIL into a static background, overlaid by ffmpeg; this
ffmpeg build has no drawtext) with the target budget and the ACTUAL validated triangle count
from final/validation.json.  Frame k of every tile is the same turntable azimuth (all
turntables start at the reference camera azimuth, 5 degree steps); the script checks the
per-variant frame counts and fps are identical.  CPU-only (libx264), no GPU lock.

--allow-missing (tooling tests only): a missing turntable becomes a labelled empty tile
instead of an error.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the folder that holds catlib/
from catlib import paths as _paths  # noqa: E402  (resolves the case: --case / LOWPOLY_CASE / cwd)
ROOT = _paths.ROOT
sys.path.insert(0, str(_paths.SCRIPTS))

from PIL import Image, ImageDraw  # noqa: E402

from catlib import paths  # noqa: E402
import integ_common as ic  # noqa: E402
from integ_common import BG, BUDGETS, INK, MUTED, BAD_RED, OK_GREEN, font  # noqa: E402

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
KINDS = {"clay_wire": "clay + wireframe", "beauty": "flat-colour beauty"}


def probe(p: Path) -> dict:
    q = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                        "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(p)],
                       capture_output=True, text=True, check=True)
    s = json.loads(q.stdout)["streams"][0]
    return {"w": int(s["width"]), "h": int(s["height"]), "fps": s["r_frame_rate"], "frames": int(s["nb_read_frames"])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="2x2 frame-synchronised turntable mosaics.")
    ap.add_argument("--variants-root", type=Path, default=paths.VARIANTS)
    ap.add_argument("--out", type=Path, default=paths.OUTPUTS)
    ap.add_argument("--allow-missing", action="store_true", help="tooling tests only: empty tile for a missing turntable")
    ap.add_argument("--crf", type=int, default=18)
    a = ap.parse_args(argv)
    out = a.out.resolve()
    entries = {N: ic.load_entry(a.variants_root, N, None) for N in BUDGETS}
    status = ic.camera_status(entries)
    (out / "turntables").mkdir(parents=True, exist_ok=True)
    rc = 0
    for kind, desc in KINDS.items():
        vids, infos = {}, {}
        for N, e in entries.items():
            p = e.render(f"turntable_{kind}.mp4")
            if p.is_file():
                vids[N] = p
                infos[N] = probe(p)
                shutil.copy2(p, out / "turntables" / f"turntable_{N}_{kind}.mp4")
            elif not a.allow_missing:
                print(f"make_turntable_mosaic: missing {p} (finalize_variant.py renders it)", file=sys.stderr)
                return 2
        if not vids:
            print(f"make_turntable_mosaic: no {kind} turntables at all", file=sys.stderr)
            return 2
        sig = {(i["w"], i["h"], i["fps"], i["frames"]) for i in infos.values()}
        if len(sig) != 1:
            print(f"make_turntable_mosaic: turntables are not frame-compatible: {infos}", file=sys.stderr)
            return 2
        tw, th, fps, frames = sig.pop()
        pad, gap, title_h, lab_h = 40, 24, 120, 84
        W = pad * 2 + 2 * tw + gap
        H = pad + title_h + 2 * (lab_h + th) + gap + pad
        W += W % 2; H += H % 2
        bg = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(bg)
        d.text((pad, pad - 6), f"Four triangle budgets: turntable, {desc}", font=font(38, "bold"), fill=INK)
        sub = f"{frames} frames, 5 degree steps, identical camera orbit and lights; frame-synchronised"
        if status != "locked":
            d.text((pad, pad + 48), sub + f"   |   camera status '{status}': NOT final", font=font(22, "bold"), fill=BAD_RED)
        else:
            d.text((pad, pad + 48), sub, font=font(22), fill=MUTED)
        pos = {}
        for i, (N, e) in enumerate(entries.items()):
            r, c = divmod(i, 2)
            x = pad + c * (tw + gap)
            y = pad + title_h + r * (lab_h + th + gap)
            t, colr = (f"{e.actual} triangles (validated, all four counts agree)", OK_GREEN) \
                if e.counts.get("all_agree") and e.actual is not None and e.actual <= N else \
                (f"{e.actual} triangles: COUNTS NOT VALIDATED", BAD_RED)
            d.text((x, y + 6), f"Budget {N}", font=font(32, "bold"), fill=INK)
            d.text((x, y + 48), t, font=font(22), fill=colr)
            pos[N] = (x, y + lab_h)
            if N not in vids:
                d.rectangle([x, y + lab_h, x + tw - 1, y + lab_h + th - 1], fill=(235, 235, 235), outline=(200, 200, 200))
                d.text((x + tw // 2, y + lab_h + th // 2), "turntable not rendered", font=font(28), fill=MUTED,
                       anchor="mm")
        bgp = out / f".mosaic_bg_{kind}.png"
        bg.save(bgp)
        cmd = [FFMPEG, "-y", "-loglevel", "error", "-loop", "1", "-framerate", fps, "-i", str(bgp)]
        order = [N for N in BUDGETS if N in vids]
        for N in order:
            cmd += ["-i", str(vids[N])]
        chain, last = [], "0:v"
        for k, N in enumerate(order):
            x, y = pos[N]
            lab = f"v{k}"
            chain.append(f"[{last}][{k + 1}:v]overlay=x={x}:y={y}:eof_action=pass[{lab}]")
            last = lab
        chain.append(f"[{last}]format=yuv420p[out]")
        mp4 = out / f"turntable_all_{kind}.mp4"
        cmd += ["-filter_complex", ";".join(chain), "-map", "[out]", "-frames:v", str(frames), "-r", fps,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(a.crf), "-preset", "medium",
                "-movflags", "+faststart", str(mp4)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        bgp.unlink(missing_ok=True)
        if p.returncode != 0:
            print(f"ffmpeg failed for {kind}: {p.stderr[-1500:]}", file=sys.stderr)
            rc = 1
            continue
        info = probe(mp4)
        if info["frames"] != frames:
            print(f"{mp4}: {info['frames']} frames, expected {frames}", file=sys.stderr)
            rc = 1
        print(f"wrote {mp4} ({info['w']}x{info['h']}, {info['frames']} frames, {fps} fps; tiles from "
              f"{order}{'; missing ' + str(sorted(set(BUDGETS) - set(order))) if len(order) < 4 else ''})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
