#!/usr/bin/env bash
set -euo pipefail

ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
ACC="${ACC:-<ANON_ACCOUNT>}"
CFG="$ROOT/benchmark_configs/test_config_smolvlm2_v91_cf3_routed_textvqa.json"
CACHE="$ROOT/.runtime_cache/test_config_smolvlm2_v91_cf3_routed_textvqa_n1000_r6"
JOB="p1-tvq-routed1k-r6"

mkdir -p "$CACHE"
cd "$ROOT"

sbatch \
  --account="$ACC" \
  --job-name="$JOB" \
  --export=ALL,CONFIG_PATH="$CFG",SUBSET_LEN=1000,CACHE_PATH="$CACHE" \
  scripts/sbatch_clean.sh "$CFG"
