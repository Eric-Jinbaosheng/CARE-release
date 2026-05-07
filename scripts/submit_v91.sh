#!/bin/bash
# Submit v91 jobs.
# PHASE=p0    -> v91_disabled @ n=50, 2 bm (TextVQA + OCRBench) — bit-identical sanity
# PHASE=p1    -> v91_nocf    @ n=200, 8 bm (no AMBER)
# PHASE=p2    -> v91_cf      @ n=200, 8 bm
# PHASE=p3    -> v91_nocf+cf @ n=1000, TextVQA only
# PHASE=p4    -> v91_nocf+cf @ n=1000, 9 bm (all including AMBER)

set -euo pipefail
ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
PHASE="${PHASE:-p0}"

case "${PHASE}" in
  p0)
    SUBSET_LEN="${SUBSET_LEN:-50}"
    BENCHMARKS="${BENCHMARKS:-textvqa ocrbench}"
    MODES="${MODES:-disabled}"  # uses paper_ttaug_classical_deterministic + v91_disabled
    ;;
  p1)
    SUBSET_LEN="${SUBSET_LEN:-200}"
    BENCHMARKS="${BENCHMARKS:-ocrbench gqa textvqa chartqa ocrvqa ai2d mme_rw coco}"
    MODES="${MODES:-nocf}"
    ;;
  p2)
    SUBSET_LEN="${SUBSET_LEN:-200}"
    BENCHMARKS="${BENCHMARKS:-ocrbench gqa textvqa chartqa ocrvqa ai2d mme_rw coco}"
    MODES="${MODES:-cf}"
    ;;
  p3)
    SUBSET_LEN="${SUBSET_LEN:-1000}"
    BENCHMARKS="${BENCHMARKS:-textvqa}"
    MODES="${MODES:-nocf cf}"
    ;;
  p4)
    SUBSET_LEN="${SUBSET_LEN:-1000}"
    BENCHMARKS="${BENCHMARKS:-ocrbench gqa textvqa chartqa ocrvqa ai2d mme_rw coco amber}"
    MODES="${MODES:-nocf cf}"
    ;;
  *)
    echo "Unknown PHASE=${PHASE}" >&2; exit 1
    ;;
esac

echo "v91 submit PHASE=${PHASE} SUBSET_LEN=${SUBSET_LEN} BENCHMARKS=${BENCHMARKS} MODES=${MODES}"
count=0
for mode in ${MODES}; do
  for bm in ${BENCHMARKS}; do
    CONFIG="${ROOT}/benchmark_configs/test_config_smolvlm2_v91_${mode}_${bm}.json"
    if [ ! -f "${CONFIG}" ]; then echo "MISS ${CONFIG}"; continue; fi
    CONFIG_STEM="$(basename "${CONFIG}" .json)"
    CACHE="${ROOT}/.runtime_cache/${CONFIG_STEM}"
    rm -rf "${ROOT}/benchmark_results/n_samples_${SUBSET_LEN}/${CONFIG_STEM}" 2>/dev/null || true
    sbatch --job-name="v91-${PHASE}-${mode}-${bm}" \
      --export=ALL,CONFIG_PATH="${CONFIG}",SUBSET_LEN="${SUBSET_LEN}",CACHE_PATH="${CACHE}" \
      "${ROOT}/scripts/sbatch_clean.sh" >/dev/null
    count=$((count + 1))
  done
done

# For P0, also (re)submit the deterministic baseline as the comparison anchor
if [ "${PHASE}" = "p0" ]; then
  for bm in ${BENCHMARKS}; do
    CONFIG="${ROOT}/benchmark_configs/test_config_smolvlm2_paper_ttaug_classical_deterministic_${bm}.json"
    if [ ! -f "${CONFIG}" ]; then echo "MISS ${CONFIG}"; continue; fi
    CONFIG_STEM="$(basename "${CONFIG}" .json)"
    CACHE="${ROOT}/.runtime_cache/${CONFIG_STEM}"
    if [ -d "${ROOT}/benchmark_results/n_samples_${SUBSET_LEN}/${CONFIG_STEM}" ]; then
      echo "  baseline ${bm} @ n=${SUBSET_LEN} already exists, skipping"
      continue
    fi
    sbatch --job-name="v91-${PHASE}-baseline-${bm}" \
      --export=ALL,CONFIG_PATH="${CONFIG}",SUBSET_LEN="${SUBSET_LEN}",CACHE_PATH="${CACHE}" \
      "${ROOT}/scripts/sbatch_clean.sh" >/dev/null
    count=$((count + 1))
  done
fi

echo "Submitted ${count} v91-${PHASE} jobs"
