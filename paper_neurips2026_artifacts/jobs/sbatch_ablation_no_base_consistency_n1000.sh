#!/usr/bin/env bash
set -euo pipefail
cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

ACC="${ACC:-<ANON_ACCOUNT>}"
PART="${PART:-l40s_public}"

sbatch \
  --account="$ACC" \
  --partition="$PART" \
  --job-name="abl-nobase-1k" \
  --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=24G --time=02:00:00 \
  --gres=gpu:1 \
  --output="logs/abl-nobase-1k-%j.out" \
  --error="logs/abl-nobase-1k-%j.err" \
  --wrap="cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean && bash paper_neurips2026_artifacts/jobs/run_ablation_no_base_consistency_n1000.sh"
