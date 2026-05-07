#!/usr/bin/env bash
set -euo pipefail

cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

ACC="${ACC:-<ANON_ACCOUNT>}"
PART="${PART:-l40s_public,a100_tandon}"
CFG="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/configs/ablation_configs/test_config_smolvlm2_v91_nocf_ablation_no_length_risk_ocrbench.json"
CACHE="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_nocf_ablation_no_length_risk_ocrbench_n1000"

mkdir -p "${CACHE}"

sbatch   --account="${ACC}"   --partition="${PART}"   --job-name="abl1k-no_length_risk-ocrbench"   --export=ALL,CONFIG_PATH="${CFG}",SUBSET_LEN=1000,CACHE_PATH="${CACHE}"   scripts/sbatch_clean.sh "${CFG}"
