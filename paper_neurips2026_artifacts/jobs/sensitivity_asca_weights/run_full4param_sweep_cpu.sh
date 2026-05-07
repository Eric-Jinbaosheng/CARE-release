#!/usr/bin/env bash
set -euo pipefail

cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

PY=python
SCRIPT=paper_neurips2026_artifacts/scripts/run_asca_sensitivity.py
BENCH="textvqa,ocrvqa,chartqa,ocrbench,gqa"
N=1000
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="paper_neurips2026_artifacts/sensitivity_asca_weights/full4param_${STAMP}"
mkdir -p "$OUT"

run_one () {
  local wsup="$1"
  local wvalid="$2"
  local wbase="$3"
  local wrisk="$4"
  local tag="$5"
  echo "[RUN] $tag wsup=$wsup wvalid=$wvalid wbase=$wbase wrisk=$wrisk"
  $PY "$SCRIPT" \
    --sweep single \
    --benchmarks "$BENCH" \
    --n "$N" \
    --w_sup "$wsup" \
    --w_valid "$wvalid" \
    --w_base "$wbase" \
    --w_risk "$wrisk" \
    --varied_param "$tag" \
    --param_value "$tag" \
    --output_dir "$OUT/$tag"
}

# default
run_one 2.0 1.0 0.4 0.5 default

# w_sup sweep
run_one 1.0 1.0 0.4 0.5 wsup_1p0
run_one 1.5 1.0 0.4 0.5 wsup_1p5
run_one 2.5 1.0 0.4 0.5 wsup_2p5
run_one 3.0 1.0 0.4 0.5 wsup_3p0

# w_valid sweep
run_one 2.0 0.25 0.4 0.5 wvalid_0p25
run_one 2.0 0.5  0.4 0.5 wvalid_0p5
run_one 2.0 1.5  0.4 0.5 wvalid_1p5
run_one 2.0 2.0  0.4 0.5 wvalid_2p0

# w_base sweep
run_one 2.0 1.0 0.0 0.5 wbase_0p0
run_one 2.0 1.0 0.2 0.5 wbase_0p2
run_one 2.0 1.0 0.6 0.5 wbase_0p6
run_one 2.0 1.0 0.8 0.5 wbase_0p8

# w_risk sweep
run_one 2.0 1.0 0.4 0.0  wrisk_0p0
run_one 2.0 1.0 0.4 0.25 wrisk_0p25
run_one 2.0 1.0 0.4 0.75 wrisk_0p75
run_one 2.0 1.0 0.4 1.0  wrisk_1p0

echo "[DONE] all settings finished: $OUT"
