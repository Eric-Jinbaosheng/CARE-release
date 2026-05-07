#!/usr/bin/env bash
set -euo pipefail

cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

PY="<ANON_ROOT>/micromamba-root/envs/eccts310/bin/python"
OUT="${OUT:-<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/sensitivity_asca_weights_ovis2_n1000}"

"${PY}" paper_neurips2026_artifacts/scripts/run_asca_sensitivity_second_backbone.py \
  --backbone ovis2_2b \
  --n 1000 \
  --benchmarks textvqa,ocrvqa,chartqa,gqa \
  --output_dir "${OUT}"

echo "[DONE] ${OUT}"

