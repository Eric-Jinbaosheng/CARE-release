#!/bin/bash
set -euo pipefail

ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
cd "$ROOT"

SUBSET_LEN="${SUBSET_LEN:-1000}"

# Comma-separated accounts can be overridden at runtime:
#   ACCOUNTS="<ANON_ACCOUNT>,<ANON_ACCOUNT>,<ANON_ACCOUNT>,<ANON_ACCOUNT>"
ACCOUNTS_CSV="${ACCOUNTS:-<ANON_ACCOUNT>,<ANON_ACCOUNT>,<ANON_ACCOUNT>,<ANON_ACCOUNT>}"
IFS=',' read -r -a ACC_ARR <<< "$ACCOUNTS_CSV"
if [ "${#ACC_ARR[@]}" -eq 0 ]; then
  echo "[ERROR] no accounts provided in ACCOUNTS" >&2
  exit 1
fi

# Optional partition. Keep empty to let scheduler choose.
PART="${PART:-}"

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

idx=0
for stem in "${STEMS[@]}"; do
  cfg="paper_neurips2026_artifacts/configs/sensitivity_groupA_truepipeline/${stem}.json"
  cache="${ROOT}/.runtime_cache/${stem}_n${SUBSET_LEN}"
  short="${stem#test_config_smolvlm2_v91_nocf_sens_}"
  short="${short//_/-}"
  job="sensA-${short}-n${SUBSET_LEN}"

  acc="${ACC_ARR[$((idx % ${#ACC_ARR[@]}))]}"
  idx=$((idx + 1))

  echo "[SUBMIT] ${job} -> ${acc}"
  if [ -n "$PART" ]; then
    sbatch \
      --account="${acc}" \
      --partition="${PART}" \
      --job-name="${job}" \
      --export=ALL,CONFIG_PATH="${cfg}",SUBSET_LEN="${SUBSET_LEN}",CACHE_PATH="${cache}" \
      scripts/sbatch_clean.sh
  else
    sbatch \
      --account="${acc}" \
      --job-name="${job}" \
      --export=ALL,CONFIG_PATH="${cfg}",SUBSET_LEN="${SUBSET_LEN}",CACHE_PATH="${cache}" \
      scripts/sbatch_clean.sh
  fi
done

