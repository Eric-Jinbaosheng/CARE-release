#!/bin/bash
set -euo pipefail

ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
cd "$ROOT"

ACC="${ACC:-<ANON_ACCOUNT>}"
PART="${PART:-l40s_public}"
SUBSET_LEN="${SUBSET_LEN:-1000}"

STEMS=(
  test_config_smolvlm2_v91_nocf_sens_default_textvqa
  test_config_smolvlm2_v91_nocf_sens_wsup1p0_textvqa
  test_config_smolvlm2_v91_nocf_sens_wsup1p5_textvqa
  test_config_smolvlm2_v91_nocf_sens_wsup2p5_textvqa
  test_config_smolvlm2_v91_nocf_sens_wsup3p0_textvqa
  test_config_smolvlm2_v91_nocf_sens_wvalid0p25_textvqa
  test_config_smolvlm2_v91_nocf_sens_wvalid0p5_textvqa
  test_config_smolvlm2_v91_nocf_sens_wvalid1p5_textvqa
  test_config_smolvlm2_v91_nocf_sens_wvalid2p0_textvqa
  test_config_smolvlm2_v91_nocf_sens_default_chartqa
  test_config_smolvlm2_v91_nocf_sens_wsup1p0_chartqa
  test_config_smolvlm2_v91_nocf_sens_wsup1p5_chartqa
  test_config_smolvlm2_v91_nocf_sens_wsup2p5_chartqa
  test_config_smolvlm2_v91_nocf_sens_wsup3p0_chartqa
  test_config_smolvlm2_v91_nocf_sens_wvalid0p25_chartqa
  test_config_smolvlm2_v91_nocf_sens_wvalid0p5_chartqa
  test_config_smolvlm2_v91_nocf_sens_wvalid1p5_chartqa
  test_config_smolvlm2_v91_nocf_sens_wvalid2p0_chartqa
)

for stem in "${STEMS[@]}"; do
  cfg="paper_neurips2026_artifacts/configs/sensitivity_groupA_truepipeline/${stem}.json"
  cache="${ROOT}/.runtime_cache/${stem}_n${SUBSET_LEN}"
  short="${stem#test_config_smolvlm2_v91_nocf_sens_}"
  short="${short//_/-}"
  job="sensA-${short}-n${SUBSET_LEN}"
  echo "[SUBMIT] ${job}"
  sbatch \
    --account="${ACC}" \
    --partition="${PART}" \
    --job-name="${job}" \
    --export=ALL,CONFIG_PATH="${cfg}",SUBSET_LEN="${SUBSET_LEN}",CACHE_PATH="${cache}" \
    scripts/sbatch_clean.sh
done

