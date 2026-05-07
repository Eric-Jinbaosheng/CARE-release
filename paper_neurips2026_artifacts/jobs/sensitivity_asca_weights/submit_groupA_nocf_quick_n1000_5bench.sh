#!/usr/bin/env bash
set -euo pipefail

ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
cd "$ROOT"

SCRIPT="paper_neurips2026_artifacts/jobs/sensitivity_asca_weights/sbatch_groupA_nocf_quick_n1000_onebench.sh"
ACC="${ACC:-}"
PART="${PART:-}"

for b in textvqa ocrvqa chartqa ocrbench gqa; do
  echo "[SUBMIT] $b"
  sbatch_args=()
  if [[ -n "$ACC" ]]; then
    sbatch_args+=(--account="$ACC")
  fi
  if [[ -n "$PART" ]]; then
    sbatch_args+=(--partition="$PART")
  fi
  sbatch "${sbatch_args[@]}" \
    --export=ALL,BENCH="$b",OUT="paper_neurips2026_artifacts/sensitivity_asca_weights/groupA_nocf_quick_n1000_${b}" \
    "$SCRIPT"
done
