#!/usr/bin/env bash
set -euo pipefail
ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
cd "$ROOT"
BENCHES=(textvqa ocrvqa chartqa gqa)
ACCOUNTS=(<ANON_ACCOUNT> <ANON_ACCOUNT> <ANON_ACCOUNT> <ANON_ACCOUNT>)
for i in "${!BENCHES[@]}"; do
  b="${BENCHES[$i]}"; acc="${ACCOUNTS[$i]}"
  echo "[SUBMIT][ASCA_STRICT] $b -> $acc"
  ACC="$acc" bash paper_neurips2026_artifacts/jobs/second_backbone_qwen2_5_vl_3b_strict_n1000/sbatch_qwen2_5_vl_3b_asca_strict_n1000.sh "$b"
done
