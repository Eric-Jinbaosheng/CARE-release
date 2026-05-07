#!/usr/bin/env bash
set -euo pipefail
cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

BENCH="${BENCH:?BENCH required}"
ACC="${ACC:-<ANON_ACCOUNT>}"
PART="${PART:-l40s_public}"
PY="<ANON_ROOT>/micromamba-root/envs/eccts310/bin/python"
OUT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/ablations/no_base_consistency_n1000"

sbatch \
  --account="$ACC" \
  --partition="$PART" \
  --job-name="abl-nobase-${BENCH}-1k" \
  --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=24G --time=01:30:00 \
  --gres=gpu:1 \
  --output="logs/abl-nobase-${BENCH}-1k-%j.out" \
  --error="logs/abl-nobase-${BENCH}-1k-%j.err" \
  --wrap="cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean && $PY paper_neurips2026_artifacts/scripts/run_ablation_no_base_consistency_n1000.py --n 1000 --benchmarks ${BENCH} --output_dir ${OUT}"
