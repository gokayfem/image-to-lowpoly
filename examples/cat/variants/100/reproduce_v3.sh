#!/bin/sh
# Rebuild v3_welded (one closed manifold) deterministically and write the CONTRACT §13
# timelapse (construction stages, real refine snapshots, unfold edits).  CPU only except the
# final official evaluation ($S/evaluate.py takes the GPU lock itself).
set -e
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python3}
S=${LOWPOLY_SCRIPTS:-../../skills/image-to-lowpoly/scripts}
export LOWPOLY_CASE="$(pwd)"
V=variants/100
TL=$V/timelapse
rm -rf $TL; mkdir -p $TL/construct $TL/refine $TL/edit
$PY $V/build_welded.py --out $V/work/W0.json --stages-dir $TL/construct
cp $V/work/W0.json $TL/construct/09_built.json
run() {  # in out tag extra-args...
  in=$1; out=$2; tag=$3; shift 3
  mkdir -p $TL/refine/$tag/snaps; cp $in $TL/refine/$tag/base.json
  $PY $S/refine_vertices.py --mesh $in --out $out --quiet "$@" \
      --snapshot-dir $TL/refine/$tag/snaps --snapshot-every 250
}
unfold() { $PY $V/unfold.py $1 $2; cp $1 $TL/edit/$3_before.json; cp $2 $TL/edit/$3_after.json; }
run $V/work/W0.json  $V/work/W1.json r1 --iters 15000 --flips
unfold $V/work/W1.json $V/work/W1u.json u1
run $V/work/W1u.json $V/work/W2.json r2 --iters 8000 --seed 2
unfold $V/work/W2.json $V/work/W2u.json u2
run $V/work/W2u.json $V/work/W3.json r3 --iters 8000 --seed 3 --w-lm 10
unfold $V/work/W3.json $V/work/W3u.json u3
run $V/work/W3u.json $V/work/W6.json r4 --iters 8000 --seed 6 --w-lm 10
mkdir -p $V/checkpoints/v3_welded
cp $V/work/W6.json $V/checkpoints/v3_welded/mesh.json
$PY $V/timelapse_stages.py
[ "$1" = "--no-eval" ] || $PY $S/evaluate.py --mesh $V/checkpoints/v3_welded/mesh.json --views
