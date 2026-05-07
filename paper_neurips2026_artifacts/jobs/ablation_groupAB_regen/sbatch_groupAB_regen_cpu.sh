#!/usr/bin/env bash
set -euo pipefail

ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
PY="${PY:-<ANON_ROOT>/micromamba-root/envs/eccts310/bin/python}"
ACC="${ACC:-<ANON_ACCOUNT>}"
TIME="${TIME:-01:00:00}"
CPUS="${CPUS:-8}"
MEM="${MEM:-32G}"

# skip coco for now
BENCHMARKS="${BENCHMARKS:-textvqa,ocrvqa,chartqa,ocrbench,gqa,ai2d,mme_rw,amber}"
REGEN_TAG="${REGEN_TAG:-regenA_20260503_161454}"
OUT_DIR="${OUT_DIR:-paper_neurips2026_artifacts/ablations/groupA_groupB_from_regen}"

mkdir -p "$ROOT/logs"

sbatch \
  --account="$ACC" \
  --job-name="abl-GA-GB-regen" \
  --nodes=1 --ntasks=1 \
  --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" \
  --output="$ROOT/logs/abl-GA-GB-regen-%j.out" \
  --error="$ROOT/logs/abl-GA-GB-regen-%j.err" \
  --wrap="cd $ROOT && \
$PY paper_neurips2026_artifacts/scripts/run_ablation_groupA_rerank_only.py --repo_root $ROOT --n 1000 --regen_tag $REGEN_TAG --benchmarks $BENCHMARKS --output_dir $OUT_DIR && \
$PY paper_neurips2026_artifacts/scripts/run_ablation_groupB_allcf_ungated.py --repo_root $ROOT --n 1000 --benchmark textvqa --dataset_tag TextVQA_VAL --full_cfg_stem test_config_smolvlm2_v91_nocf_regen_textvqa --full_model V91NoCF_SmolVLM2_2B --cf_diag .runtime_cache/test_config_smolvlm2_v91_cf3_routed_textvqa_n1000/diagnostics/v91cf3_samples.jsonl --output_dir $OUT_DIR"
