#!/usr/bin/env bash
set -euo pipefail

cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

ABLS=(frequency_only no_format no_base_bias no_length_risk no_answer_space majority_vote base_only first_view)
BENCH=(textvqa ocrvqa gqa chartqa ocrbench)

for b in "${BENCH[@]}"; do
  for a in "${ABLS[@]}"; do
    CFG="paper_neurips2026_artifacts/configs/ablation_configs/test_config_smolvlm2_v91_nocf_ablation_${a}_${b}.json"
    if [[ ! -f "$CFG" ]]; then
      echo "Missing config: $CFG"
      continue
    fi
    export CONFIG_PATH="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/${CFG}"
    export SUBSET_LEN=200
    export CACHE_PATH="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/$(basename "$CFG" .json)_n200"
    sbatch --job-name="abl200-${a}-${b}" scripts/sbatch_clean.sh "$CONFIG_PATH"
  done
done
