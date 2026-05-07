#!/usr/bin/env bash
set -euo pipefail

cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

ACC="${ACC:-<ANON_ACCOUNT>}"
PART="${PART:-l40s_public}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

for b in ocrvqa ocrbench chartqa; do
  cfg="benchmark_configs/test_config_smolvlm2_v91_cf3_allcf_${b}.json"
  cache=".runtime_cache/test_config_smolvlm2_v91_cf3_allcf_${b}_n1000_${STAMP}"
  mkdir -p "$cache"
  echo "[SUBMIT] ${b} acc=${ACC} part=${PART}"
  sbatch \
    --account="${ACC}" \
    --partition="${PART}" \
    --job-name="allcf-${b}-1k" \
    --export=ALL,CONFIG_PATH="${cfg}",SUBSET_LEN=1000,CACHE_PATH="$(pwd)/${cache}" \
    scripts/sbatch_clean.sh "${cfg}"
done

