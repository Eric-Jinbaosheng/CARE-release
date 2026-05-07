#!/usr/bin/env bash
set -euo pipefail
ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
cd "$ROOT"
BENCHES=(textvqa ocrvqa chartqa gqa ocrbench ai2d mme_rw coco amber)
ACCOUNTS=(
  <ANON_ACCOUNT>
  <ANON_ACCOUNT>
  <ANON_ACCOUNT>
  <ANON_ACCOUNT>
  <ANON_ACCOUNT>
)
for i in "${!BENCHES[@]}"; do
  b="${BENCHES[$i]}"
  acc="${ACCOUNTS[$(( i % ${#ACCOUNTS[@]} ))]}"
  echo "[SUBMIT][ASCA] $b -> $acc"
  ACC="$acc" PART="${PART:-}" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_internvl2_5_2b_n1000/sbatch_internvl2_5_2b_asca_n1000.sh "$b"
done
