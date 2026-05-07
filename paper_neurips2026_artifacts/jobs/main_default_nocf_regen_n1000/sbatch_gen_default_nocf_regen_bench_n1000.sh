#!/usr/bin/env bash
set -euo pipefail

cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

BENCH="${BENCH:?BENCH required}"
ACC="${ACC:-<ANON_ACCOUNT>}"
PART="${PART:-}"  # keep empty to let scheduler choose
RUN_TAG="${RUN_TAG:-regenA_$(date +%Y%m%d_%H%M%S)}"

ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
CFG="${ROOT}/paper_neurips2026_artifacts/configs/main_default_nocf_regen_n1000/test_config_smolvlm2_v91_nocf_regen_${BENCH}.json"
CACHE="${ROOT}/.runtime_cache/test_config_smolvlm2_v91_nocf_regen_${BENCH}_n1000_${RUN_TAG}"

if [[ ! -f "${CFG}" ]]; then
  echo "[ERROR] config missing: ${CFG}" >&2
  exit 1
fi

mkdir -p "${CACHE}"

if [[ -n "${PART}" ]]; then
  sbatch \
    --account="${ACC}" \
    --partition="${PART}" \
    --job-name="regen-default-${BENCH}-1k" \
    --export=ALL,CONFIG_PATH="${CFG}",SUBSET_LEN=1000,CACHE_PATH="${CACHE}" \
    scripts/sbatch_clean.sh "${CFG}"
else
  sbatch \
    --account="${ACC}" \
    --job-name="regen-default-${BENCH}-1k" \
    --export=ALL,CONFIG_PATH="${CFG}",SUBSET_LEN=1000,CACHE_PATH="${CACHE}" \
    scripts/sbatch_clean.sh "${CFG}"
fi

