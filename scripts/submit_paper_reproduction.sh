#!/bin/bash
# Clean-repo paper TTAug reproduction on SmolVLM2-2.2B.
#
# Submits 9 benchmarks x 2 modes (base + paper_ttaug_classical) = 18 jobs.
#
# Usage:
#   bash scripts/submit_paper_reproduction.sh                       # SUBSET_LEN=1000 default, both modes, 9 bm
#   SUBSET_LEN=200 bash scripts/submit_paper_reproduction.sh        # smoke
#   MODE=base                  bash ...
#   MODE=paper_ttaug_classical bash ...
#   MODE=both (default)
#   BENCHMARKS="ocrbench gqa"  bash ...   # subset of bm

set -euo pipefail

ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
SUBSET_LEN="${SUBSET_LEN:-1000}"
MODE="${MODE:-both}"
BENCHMARKS="${BENCHMARKS:-ocrbench gqa textvqa chartqa amber ai2d ocrvqa mme_rw coco}"

case "${MODE}" in
  both)                  MODES=(base paper_ttaug_classical) ;;
  base)                  MODES=(base) ;;
  paper_ttaug_classical) MODES=(paper_ttaug_classical) ;;
  *) echo "ERROR: MODE must be base|paper_ttaug_classical|both" >&2; exit 1 ;;
esac

echo "Clean-repo paper TTAug reproduction"
echo "  ROOT      = ${ROOT}"
echo "  SUBSET_LEN= ${SUBSET_LEN}"
echo "  MODES     = ${MODES[*]}"
echo "  BENCHMARKS= ${BENCHMARKS}"

for mode in "${MODES[@]}"; do
  for bm in ${BENCHMARKS}; do
    CONFIG="${ROOT}/benchmark_configs/test_config_smolvlm2_${mode}_${bm}.json"
    [ -f "${CONFIG}" ] || { echo "WARNING: missing ${CONFIG}"; continue; }
    CONFIG_STEM="$(basename "${CONFIG}" .json)"
    CACHE="${ROOT}/.runtime_cache/${CONFIG_STEM}"
    rm -rf "${ROOT}/benchmark_results/n_samples_${SUBSET_LEN}/${CONFIG_STEM}" 2>/dev/null || true

    sbatch --job-name="cleanpaper-${mode}-${bm}" \
      --export=ALL,CONFIG_PATH="${CONFIG}",SUBSET_LEN="${SUBSET_LEN}",CACHE_PATH="${CACHE}",REUSE_RESULTS="0" \
      "${ROOT}/scripts/sbatch_clean.sh"
  done
done

echo "Submitted clean-repo reproduction jobs (mode=${MODE}, n=${SUBSET_LEN})."
