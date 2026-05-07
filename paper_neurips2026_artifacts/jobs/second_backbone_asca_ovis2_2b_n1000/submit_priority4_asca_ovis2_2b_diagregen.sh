#!/usr/bin/env bash
set -euo pipefail

cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

ACC="${ACC:-<ANON_ACCOUNT>}"
PART="${PART:-l40s_public}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

BENCHES=(textvqa ocrvqa chartqa gqa)

for b in "${BENCHES[@]}"; do
  cfg="paper_neurips2026_artifacts/configs/second_backbone_asca_ovis2_2b_n1000/test_config_ovis2_2b_asca_${b}.json"
  cache="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_ovis2_2b_asca_${b}_n1000_diagregen_${STAMP}"
  mkdir -p "${cache}"
  echo "[SUBMIT] ${b} ACC=${ACC} PART=${PART}"
  sbatch \
    --account="${ACC}" \
    --partition="${PART}" \
    --job-name="ov2asca-${b}-1k-dg" \
    --export=ALL,CONFIG_PATH="${cfg}",SUBSET_LEN=1000,CACHE_PATH="${cache}" \
    scripts/sbatch_clean.sh "${cfg}"
done

