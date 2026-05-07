#!/usr/bin/env bash
#SBATCH --job-name=asca-gA-nocf-1b
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
PY="<ANON_ROOT>/micromamba-root/envs/eccts310/bin/python"

BENCH="${BENCH:-textvqa}"
OUT="${OUT:-paper_neurips2026_artifacts/sensitivity_asca_weights/groupA_nocf_quick_n1000_${BENCH}}"

cd "$ROOT"
mkdir -p logs "$OUT"

echo "[INFO] benchmark=$BENCH"
echo "[INFO] output=$OUT"

"$PY" paper_neurips2026_artifacts/scripts/run_asca_sensitivity.py \
  --sweep quick \
  --n 1000 \
  --benchmarks "$BENCH" \
  --default-method nocf \
  --official_eval \
  --require_official_eval \
  --full-support-only \
  --output_dir "$OUT"

echo "[DONE] $BENCH -> $OUT"
