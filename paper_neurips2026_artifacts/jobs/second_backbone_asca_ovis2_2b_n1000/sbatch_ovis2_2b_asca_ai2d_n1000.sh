#!/usr/bin/env bash
set -euo pipefail

cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

ACC="${ACC:-<ANON_ACCOUNT>}"
PART="${PART:-l40s_public}"
CFG="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/configs/second_backbone_asca_ovis2_2b_n1000/test_config_ovis2_2b_asca_ai2d.json"
CACHE="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_ovis2_2b_asca_ai2d_n1000"

mkdir -p "${CACHE}"

sbatch       --account="${ACC}"       --partition="${PART}"       --job-name="asca-ov2-ai2d-1k"       --export=ALL,CONFIG_PATH="${CFG}",SUBSET_LEN=1000,CACHE_PATH="${CACHE}"       scripts/sbatch_clean.sh "${CFG}"
