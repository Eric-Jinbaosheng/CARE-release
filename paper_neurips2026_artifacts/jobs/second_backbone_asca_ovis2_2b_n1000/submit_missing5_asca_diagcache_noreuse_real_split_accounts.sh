#!/usr/bin/env bash
set -euo pipefail

cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean
STAMP="$(date +%Y%m%d_%H%M%S)"
PART="${PART:-l40s_public}"

submit_one() {
  local bench="$1"
  local acc="$2"
  local cfg="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/configs/second_backbone_asca_ovis2_2b_n1000/test_config_ovis2_2b_asca_${bench}.json"
  local cache="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_ovis2_2b_asca_${bench}_n1000_diagregen_noreuse_REAL_${STAMP}"
  local work="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_ovis2_2b_asca_${bench}_diagregen_noreuse_REAL_${STAMP}"

  mkdir -p "$cache" "$work"
  echo "[SUBMIT] bench=${bench} acc=${acc} part=${PART}"
  sbatch \
    --account="${acc}" \
    --partition="${PART}" \
    --job-name="ov2dg-${bench}-1k" \
    --export=ALL,CONFIG_PATH="${cfg}",SUBSET_LEN=1000,CACHE_PATH="${cache}",WORKDIR_OVERRIDE="${work}" \
    scripts/sbatch_clean_noreuse.sh "${cfg}"
}

submit_one ocrbench <ANON_ACCOUNT>
submit_one ai2d     <ANON_ACCOUNT>
submit_one mme_rw   <ANON_ACCOUNT>
submit_one amber    <ANON_ACCOUNT>
submit_one coco     <ANON_ACCOUNT>

echo "[DONE] submitted 5 no-reuse diagregen jobs."
