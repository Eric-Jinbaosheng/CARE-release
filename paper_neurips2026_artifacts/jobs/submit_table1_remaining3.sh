#!/usr/bin/env bash
set -euo pipefail
cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean
for cfg in paper_neurips2026_artifacts/configs/table1_remaining3/test_config_smolvlm2_table1_*.json; do
  stem=$(basename "$cfg" .json)
  export CONFIG_PATH="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/$cfg"
  export SUBSET_LEN=1000
  export CACHE_PATH="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/${stem}"
  job="t1-${stem#test_config_smolvlm2_table1_}"
  sbatch --job-name="$job" scripts/sbatch_clean.sh "$CONFIG_PATH"
done
