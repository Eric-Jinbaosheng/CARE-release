#!/usr/bin/env bash
set -euo pipefail

BENCH="${BENCH:?BENCH required, e.g. textvqa}"
ACC="${ACC:-<ANON_ACCOUNT>}"
PART="${PART:-l40s_public}"
ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
PY="<ANON_ROOT>/micromamba-root/envs/eccts310/bin/python"
OUT_DIR="${OUT_DIR:-$ROOT/paper_neurips2026_artifacts/sensitivity_asca_weights_jobsplit/n1000/${BENCH}}"

mkdir -p "$ROOT/logs" "$OUT_DIR"

sbatch \
  --account="$ACC" \
  --partition="$PART" \
  --job-name="sens-${BENCH}-1k" \
  --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=24G --time=04:00:00 \
  --gres=gpu:1 \
  --output="$ROOT/logs/sens-${BENCH}-1k-%j.out" \
  --error="$ROOT/logs/sens-${BENCH}-1k-%j.err" \
  --wrap="cd $ROOT && $PY paper_neurips2026_artifacts/scripts/run_asca_sensitivity.py --sweep quick --n 1000 --benchmark ${BENCH} --output_dir ${OUT_DIR}"
