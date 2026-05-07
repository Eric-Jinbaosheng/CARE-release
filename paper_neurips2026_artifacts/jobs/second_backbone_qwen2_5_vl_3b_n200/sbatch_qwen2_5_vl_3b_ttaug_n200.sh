#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <benchmark>"
  echo "benchmark in: textvqa ocrvqa chartqa gqa"
  exit 1
fi
BENCH="$1"
ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
ACC="${ACC:-<ANON_ACCOUNT>}"
CFG="$ROOT/paper_neurips2026_artifacts/configs/second_backbone_qwen2_5_vl_3b_n200/test_config_qwen2_5_vl_3b_ttaug_${BENCH}.json"
CACHE="$ROOT/.runtime_cache/test_config_qwen2_5_vl_3b_ttaug_${BENCH}_n200"
[[ -f "$CFG" ]] || { echo "Missing config: $CFG"; exit 2; }
mkdir -p "$CACHE"
SBATCH_ARGS=(--account="$ACC" --job-name="ttaug-qw25-${BENCH}-200" --gres="gpu:1" --export="ALL,CONFIG_PATH=$CFG,SUBSET_LEN=200,CACHE_PATH=$CACHE")
sbatch "${SBATCH_ARGS[@]}" scripts/sbatch_clean.sh "$CFG"
