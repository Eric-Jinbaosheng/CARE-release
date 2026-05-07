#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <benchmark>"
  echo "benchmark in: textvqa ocrvqa chartqa gqa ocrbench ai2d mme_rw coco amber"
  exit 1
fi
BENCH="$1"
ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
ACC="${ACC:-<ANON_ACCOUNT>}"
PART="${PART:-}"
CFG="$ROOT/paper_neurips2026_artifacts/configs/second_backbone_internvl2_5_2b_n1000/test_config_internvl2_5_2b_ttaug_${BENCH}.json"
CACHE="$ROOT/.runtime_cache/test_config_internvl2_5_2b_ttaug_${BENCH}_n1000"
[[ -f "$CFG" ]] || { echo "Missing config: $CFG"; exit 2; }
mkdir -p "$CACHE"
SBATCH_ARGS=(--account="$ACC" --job-name="ttaug-ivl25-${BENCH}-1k" --export="ALL,CONFIG_PATH=$CFG,SUBSET_LEN=1000,CACHE_PATH=$CACHE")
if [[ -n "$PART" ]]; then SBATCH_ARGS+=(--partition="$PART"); fi
sbatch "${SBATCH_ARGS[@]}" scripts/sbatch_clean.sh "$CFG"
