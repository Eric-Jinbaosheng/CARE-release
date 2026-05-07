#!/usr/bin/env bash
set -euo pipefail
cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

PY=<ANON_ROOT>/micromamba-root/envs/eccts310/bin/python
OUT=<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/ablations/affected_subset_n1000

$PY paper_neurips2026_artifacts/scripts/ablation_affected_subset_analysis.py \
  --n 1000 \
  --benchmarks textvqa,ocrvqa,gqa,chartqa,ocrbench \
  --output_dir "$OUT"

