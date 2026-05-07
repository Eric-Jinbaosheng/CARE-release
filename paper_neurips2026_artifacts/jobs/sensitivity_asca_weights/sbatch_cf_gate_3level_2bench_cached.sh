#!/usr/bin/env bash
set -euo pipefail
ROOT=<ANON_ROOT>/peking/smolvlm2_paper/ets_clean
PY=<ANON_ROOT>/micromamba-root/envs/eccts310/bin/python
cd "$ROOT"

"$PY" paper_neurips2026_artifacts/scripts/run_cf_gate_sensitivity_cached.py \
  --n 1000 \
  --benchmarks chartqa,textvqa \
  --output_dir paper_neurips2026_artifacts/sensitivity_cf_gate/strict_default_loose_2bench \
  --official_eval
