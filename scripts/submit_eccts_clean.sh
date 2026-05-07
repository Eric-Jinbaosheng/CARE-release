#!/bin/bash
# Submit ECCTS-on-paper-TTAug variants in the CLEAN repo.
#
# Defaults are Step 7 (n=50 debug):
#   3 variants × 4 gate settings × 3 benchmarks = 36 jobs at n=50
#
# For Step 8 (n=200 smoke), override:
#   STAGE=smoke SUBSET_LEN=200 GATES=normal BENCHMARKS="ocrbench gqa textvqa chartqa ocrvqa" \
#     bash scripts/submit_eccts_clean.sh
#
# Usage:
#   bash scripts/submit_eccts_clean.sh
#   STAGE=debug bash scripts/submit_eccts_clean.sh
#   VARIANTS=v7c GATES="normal disable_qual" BENCHMARKS=ocrbench bash ...

set -euo pipefail

ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
STAGE="${STAGE:-debug}"
SUBSET_LEN="${SUBSET_LEN:-}"
VARIANTS="${VARIANTS:-v7c v7d v7e}"
GATES="${GATES:-}"
BENCHMARKS="${BENCHMARKS:-}"

if [ "${STAGE}" = "debug" ]; then
    SUBSET_LEN="${SUBSET_LEN:-50}"
    GATES="${GATES:-normal disable_unc disable_qual disable_both}"
    BENCHMARKS="${BENCHMARKS:-ocrbench gqa textvqa}"
elif [ "${STAGE}" = "smoke" ]; then
    SUBSET_LEN="${SUBSET_LEN:-200}"
    GATES="${GATES:-normal}"
    BENCHMARKS="${BENCHMARKS:-ocrbench gqa textvqa chartqa ocrvqa}"
else
    echo "Unknown STAGE=${STAGE} (use 'debug' or 'smoke')" >&2
    exit 1
fi

echo "ECCTS-clean submission"
echo "  STAGE      = ${STAGE}"
echo "  ROOT       = ${ROOT}"
echo "  VARIANTS   = ${VARIANTS}"
echo "  GATES      = ${GATES}"
echo "  BENCHMARKS = ${BENCHMARKS}"
echo "  SUBSET_LEN = ${SUBSET_LEN}"

count=0
for v in ${VARIANTS}; do
  for g in ${GATES}; do
    for bm in ${BENCHMARKS}; do
      CONFIG="${ROOT}/benchmark_configs/test_config_smolvlm2_eccts_clean_${v}_${g}_${bm}.json"
      if [ ! -f "${CONFIG}" ]; then
        echo "WARN: missing ${CONFIG}"
        continue
      fi
      CONFIG_STEM="$(basename "${CONFIG}" .json)"
      CACHE="${ROOT}/.runtime_cache/${CONFIG_STEM}"
      rm -rf "${ROOT}/benchmark_results/n_samples_${SUBSET_LEN}/${CONFIG_STEM}" 2>/dev/null || true
      sbatch --job-name="eccts-${STAGE}-${v}-${g}-${bm}" \
        --export=ALL,CONFIG_PATH="${CONFIG}",SUBSET_LEN="${SUBSET_LEN}",CACHE_PATH="${CACHE}" \
        "${ROOT}/scripts/sbatch_clean.sh"
      count=$((count + 1))
    done
  done
done

echo "Submitted ${count} jobs (STAGE=${STAGE}, n=${SUBSET_LEN})."
