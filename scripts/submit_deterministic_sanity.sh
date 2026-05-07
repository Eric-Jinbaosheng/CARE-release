#!/bin/bash
# Step 4: deterministic-seeded sanity check.
# Submits paper_classical_deterministic + 3 eccts_disabled_deterministic
# variants on chosen benchmarks. With per-sample seeding ON, all four MUST
# match sample-by-sample.
#
# Usage:
#   SUBSET_LEN=50 BENCHMARKS="ocrbench textvqa" bash scripts/submit_deterministic_sanity.sh

set -euo pipefail
ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
SUBSET_LEN="${SUBSET_LEN:-50}"
BENCHMARKS="${BENCHMARKS:-ocrbench textvqa}"
VARIANTS="${VARIANTS:-v7c v7d v7e}"

echo "Deterministic-sanity submission: SUBSET_LEN=${SUBSET_LEN} BENCHMARKS=${BENCHMARKS}"

count=0
for bm in ${BENCHMARKS}; do
  CONFIG="${ROOT}/benchmark_configs/test_config_smolvlm2_paper_ttaug_classical_deterministic_${bm}.json"
  if [ -f "${CONFIG}" ]; then
    CONFIG_STEM="$(basename "${CONFIG}" .json)"
    CACHE="${ROOT}/.runtime_cache/${CONFIG_STEM}"
    rm -rf "${ROOT}/benchmark_results/n_samples_${SUBSET_LEN}/${CONFIG_STEM}" 2>/dev/null || true
    sbatch --job-name="detsan-baseline-${bm}" \
      --export=ALL,CONFIG_PATH="${CONFIG}",SUBSET_LEN="${SUBSET_LEN}",CACHE_PATH="${CACHE}" \
      "${ROOT}/scripts/sbatch_clean.sh"
    count=$((count + 1))
  fi

  for v in ${VARIANTS}; do
    CONFIG="${ROOT}/benchmark_configs/test_config_smolvlm2_eccts_disabled_deterministic_${v}_${bm}.json"
    if [ -f "${CONFIG}" ]; then
      CONFIG_STEM="$(basename "${CONFIG}" .json)"
      CACHE="${ROOT}/.runtime_cache/${CONFIG_STEM}"
      rm -rf "${ROOT}/benchmark_results/n_samples_${SUBSET_LEN}/${CONFIG_STEM}" 2>/dev/null || true
      sbatch --job-name="detsan-disabled-${v}-${bm}" \
        --export=ALL,CONFIG_PATH="${CONFIG}",SUBSET_LEN="${SUBSET_LEN}",CACHE_PATH="${CACHE}" \
        "${ROOT}/scripts/sbatch_clean.sh"
      count=$((count + 1))
    fi
  done
done

echo "Submitted ${count} jobs."
