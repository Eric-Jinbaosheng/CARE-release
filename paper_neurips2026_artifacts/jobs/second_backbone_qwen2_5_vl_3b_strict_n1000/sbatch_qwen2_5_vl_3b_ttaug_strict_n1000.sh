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
CFG="$ROOT/paper_neurips2026_artifacts/configs/second_backbone_qwen2_5_vl_3b_ttaug_strict_n1000/test_config_qwen2_5_vl_3b_ttaug_strict_${BENCH}.json"
CACHE="$ROOT/.runtime_cache/test_config_qwen2_5_vl_3b_ttaug_strict_${BENCH}_n1000"
[[ -f "$CFG" ]] || { echo "Missing config: $CFG"; exit 2; }
mkdir -p "$CACHE"
sbatch --account="$ACC" --job-name="ttaug-qw25s-${BENCH}-1k" --gres="gpu:1" \
  --export="ALL,CONFIG_PATH=$CFG,SUBSET_LEN=1000,CACHE_PATH=$CACHE,FAIL_IF_NO_SCORE=1" \
  scripts/sbatch_clean.sh "$CFG"
