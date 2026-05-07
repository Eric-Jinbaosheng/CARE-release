#!/usr/bin/env bash
set -euo pipefail

cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

ACC="${ACC:-<ANON_ACCOUNT>}"
PART="${PART:-l40s_public}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
PY="<ANON_ROOT>/micromamba-root/envs/eccts310/bin/python"

BENCHES=(textvqa ocrvqa chartqa gqa)

for b in "${BENCHES[@]}"; do
  cfg="paper_neurips2026_artifacts/configs/second_backbone_asca_ovis2_2b_n1000/test_config_ovis2_2b_asca_${b}.json"
  cfg_stem="$(basename "${cfg}" .json)"
  cache="${ROOT}/.runtime_cache/${cfg_stem}_n1000_diagregen_noreuse_${STAMP}"
  work="${ROOT}/benchmark_results/n_samples_1000/${cfg_stem}_diagregen_noreuse_${STAMP}"
  mkdir -p "${cache}" "${cache}/tmp" "${work}" "${ROOT}/logs"
  cp "${cfg}" "${work}/"

  echo "[SUBMIT] ${b} ACC=${ACC} PART=${PART}"
  sbatch \
    --account="${ACC}" \
    --partition="${PART}" \
    --job-name="ov2asca-${b}-1k-nr" \
    --nodes=1 --ntasks=1 --gres=gpu:1 \
    --cpus-per-task=16 --mem=64G --time=24:00:00 \
    --output="${ROOT}/logs/ov2asca-${b}-1k-nr-%j.out" \
    --error="${ROOT}/logs/ov2asca-${b}-1k-nr-%j.err" \
    --export=ALL,CONFIG_PATH="${cfg}",SUBSET_LEN=1000,CACHE_PATH="${cache}" \
    --wrap="cd ${ROOT} && export PYTHONPATH=${ROOT}:\$PYTHONPATH && export TOKENIZERS_PARALLELISM=false && export AUTO_SPLIT=0 && export DIST_TIMEOUT=99999999999 && export UNSLOTH_DISABLE_FAST_GENERATION=1 && export SHARED_HF_CACHE=<ANON_ROOT>/hf_cache && export HF_HOME=\$SHARED_HF_CACHE && export TRANSFORMERS_CACHE=\$SHARED_HF_CACHE && export DATASETS_CACHE=\$SHARED_HF_CACHE && export TORCH_HOME=\$SHARED_HF_CACHE && export TMPDIR=${cache}/tmp && export NLTK_DATA=<ANON_ROOT>/nltk_data && ${PY} run.py --work-dir \"${work}\" --verbose --config \"${cfg}\""
done

