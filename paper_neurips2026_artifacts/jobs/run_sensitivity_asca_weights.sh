#!/usr/bin/env bash
set -euo pipefail
cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

PY=<ANON_ROOT>/micromamba-root/envs/eccts310/bin/python
OUT=<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/sensitivity_asca_weights

$PY paper_neurips2026_artifacts/scripts/run_asca_sensitivity.py \
  --sweep quick \
  --n 1000 \
  --benchmarks textvqa,ocrvqa,chartqa,ocrbench,gqa \
  --output_dir "$OUT"
