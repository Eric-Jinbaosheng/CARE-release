#!/usr/bin/env bash
set -euo pipefail

cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

for b in textvqa ocrvqa gqa ocrbench; do
  CFG="paper_neurips2026_artifacts/configs/second_backbone_configs/test_config_ovis2_1b_ttaug_det_${b}.json"
  if [[ ! -f "$CFG" ]]; then
    echo "Missing config: $CFG"
    continue
  fi
  export CONFIG_PATH="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/${CFG}"
  export SUBSET_LEN=200
  export CACHE_PATH="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/$(basename "$CFG" .json)_n200"
  sbatch --job-name="ovis1b200-${b}" scripts/sbatch_clean.sh "$CONFIG_PATH"
done
