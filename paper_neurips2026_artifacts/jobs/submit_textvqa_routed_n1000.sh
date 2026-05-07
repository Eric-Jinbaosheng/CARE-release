#!/usr/bin/env bash
set -euo pipefail

cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

CFG="benchmark_configs/test_config_smolvlm2_v91_cf3_routed_textvqa.json"
export CONFIG_PATH="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/${CFG}"
export SUBSET_LEN=1000
export CACHE_PATH="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf3_routed_textvqa"

sbatch --job-name="v91-routed-n1000-textvqa" scripts/sbatch_clean.sh "$CONFIG_PATH"
