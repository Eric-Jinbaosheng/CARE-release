#!/bin/bash
# Submit v91_cf2 debug job: n=200 TextVQA only
# Goal: verify continuous log-likelihood CF scoring works
# Condition to proceed: rescue:harm >= 2:1 and v91_cf2 > v91_nocf

set -euo pipefail
ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
SUBSET_LEN="${SUBSET_LEN:-200}"
CONFIG="${ROOT}/benchmark_configs/test_config_smolvlm2_v91_cf2_textvqa.json"
CONFIG_STEM="$(basename "${CONFIG}" .json)"
CACHE="${ROOT}/.runtime_cache/${CONFIG_STEM}"

echo "Submitting v91_cf2 debug: n=${SUBSET_LEN} TextVQA"
rm -rf "${ROOT}/benchmark_results/n_samples_${SUBSET_LEN}/${CONFIG_STEM}" 2>/dev/null || true

sbatch --job-name="v91-cf2-debug-textvqa" \
  --export=ALL,CONFIG_PATH="${CONFIG}",SUBSET_LEN="${SUBSET_LEN}",CACHE_PATH="${CACHE}" \
  "${ROOT}/scripts/sbatch_clean.sh"
