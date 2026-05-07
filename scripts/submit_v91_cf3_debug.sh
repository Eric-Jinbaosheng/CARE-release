#!/bin/bash
# Submit v91_cf3 debug/diagnostic job: n=200 TextVQA only.
# Usage:
#   CONFIG_NAME=test_config_smolvlm2_v91_cf3_textvqa ./scripts/submit_v91_cf3_debug.sh
#   CONFIG_NAME=test_config_smolvlm2_v91_cf3_force_grid_no_quality_gate_textvqa ./scripts/submit_v91_cf3_debug.sh

set -euo pipefail
ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
SUBSET_LEN="${SUBSET_LEN:-200}"
CONFIG_NAME="${CONFIG_NAME:-test_config_smolvlm2_v91_cf3_textvqa}"
CONFIG="${ROOT}/benchmark_configs/${CONFIG_NAME}.json"
CONFIG_STEM="$(basename "${CONFIG}" .json)"
CACHE="${ROOT}/.runtime_cache/${CONFIG_STEM}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

echo "Submitting ${CONFIG_STEM}: n=${SUBSET_LEN} TextVQA"
rm -rf "${ROOT}/benchmark_results/n_samples_${SUBSET_LEN}/${CONFIG_STEM}" 2>/dev/null || true

sbatch --job-name="v91-cf3-debug-textvqa" \
  --export=ALL,CONFIG_PATH="${CONFIG}",SUBSET_LEN="${SUBSET_LEN}",CACHE_PATH="${CACHE}" \
  "${ROOT}/scripts/sbatch_clean.sh"
