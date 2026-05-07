#!/usr/bin/env bash
set -euo pipefail
cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean
for f in paper_neurips2026_artifacts/jobs/ablation_n1000_4bench/sbatch_n1000_*.sh; do
  echo "[SUBMIT] $f"
  bash "$f"
done
