#!/bin/bash
# Step 3: ECCTS-disabled equivalence sanity check.
#
# Submits paper-TTAug-classical re-run + ECCTS-disabled (v7c/v7d/v7e)
# on OCRBench and TextVQA at n=SUBSET_LEN (default 50).
#
# After all 8 jobs complete, compare per-sample predictions (xlsx files
# under benchmark_results/n_samples_<SUBSET_LEN>/<config>/<model>/).
# ECCTS-disabled should match the runB baseline sample-by-sample.
#
# Usage:
#   SUBSET_LEN=50 bash scripts/submit_equivalence_check.sh
#   SUBSET_LEN=200 BENCHMARKS="ocrbench" bash scripts/submit_equivalence_check.sh

set -euo pipefail

ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
SUBSET_LEN="${SUBSET_LEN:-50}"
BENCHMARKS="${BENCHMARKS:-ocrbench textvqa}"
VARIANTS="${VARIANTS:-v7c v7d v7e}"

echo "Equivalence-check submission"
echo "  ROOT       = ${ROOT}"
echo "  BENCHMARKS = ${BENCHMARKS}"
echo "  VARIANTS   = ${VARIANTS}"
echo "  SUBSET_LEN = ${SUBSET_LEN}"

for bm in ${BENCHMARKS}; do
  # Re-run baseline (noise floor)
  CONFIG="${ROOT}/benchmark_configs/test_config_smolvlm2_paper_ttaug_classical_runB_${bm}.json"
  if [ -f "${CONFIG}" ]; then
    CONFIG_STEM="$(basename "${CONFIG}" .json)"
    CACHE="${ROOT}/.runtime_cache/${CONFIG_STEM}"
    rm -rf "${ROOT}/benchmark_results/n_samples_${SUBSET_LEN}/${CONFIG_STEM}" 2>/dev/null || true
    sbatch --job-name="eqcheck-classical-runB-${bm}" \
      --export=ALL,CONFIG_PATH="${CONFIG}",SUBSET_LEN="${SUBSET_LEN}",CACHE_PATH="${CACHE}" \
      "${ROOT}/scripts/sbatch_clean.sh"
  else
    echo "WARN: missing ${CONFIG}"
  fi

  # ECCTS-disabled bypass per variant
  for v in ${VARIANTS}; do
    CONFIG="${ROOT}/benchmark_configs/test_config_smolvlm2_eccts_disabled_${v}_${bm}.json"
    if [ ! -f "${CONFIG}" ]; then
      echo "WARN: missing ${CONFIG}"
      continue
    fi
    CONFIG_STEM="$(basename "${CONFIG}" .json)"
    CACHE="${ROOT}/.runtime_cache/${CONFIG_STEM}"
    rm -rf "${ROOT}/benchmark_results/n_samples_${SUBSET_LEN}/${CONFIG_STEM}" 2>/dev/null || true
    sbatch --job-name="eqcheck-disabled-${v}-${bm}" \
      --export=ALL,CONFIG_PATH="${CONFIG}",SUBSET_LEN="${SUBSET_LEN}",CACHE_PATH="${CACHE}" \
      "${ROOT}/scripts/sbatch_clean.sh"
  done
done

echo "Equivalence-check submission complete (n=${SUBSET_LEN})."
