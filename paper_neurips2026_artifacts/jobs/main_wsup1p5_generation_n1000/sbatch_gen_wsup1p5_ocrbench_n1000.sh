#!/usr/bin/env bash
set -euo pipefail

cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

ACC="${ACC:-<ANON_ACCOUNT>}"
PART="${PART:-}"  # keep empty to let scheduler choose
RUN_TAG="${RUN_TAG:-genA_$(date +%Y%m%d_%H%M%S)}"

CFG="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/configs/main_wsup1p5/test_config_smolvlm2_v91_nocf_wsup1p5_ocrbench.json"
CACHE="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_nocf_wsup1p5_ocrbench_n1000_${RUN_TAG}"

mkdir -p "${CACHE}"

if [[ -n "${PART}" ]]; then
  sbatch --account="${ACC}" --partition="${PART}" \
    --job-name="gen-wsup15-ocrbench-1k" \
    --export=ALL,CONFIG_PATH="${CFG}",SUBSET_LEN=1000,CACHE_PATH="${CACHE}" \
    scripts/sbatch_clean.sh "${CFG}"
else
  sbatch --account="${ACC}" \
    --job-name="gen-wsup15-ocrbench-1k" \
    --export=ALL,CONFIG_PATH="${CFG}",SUBSET_LEN=1000,CACHE_PATH="${CACHE}" \
    scripts/sbatch_clean.sh "${CFG}"
fi
