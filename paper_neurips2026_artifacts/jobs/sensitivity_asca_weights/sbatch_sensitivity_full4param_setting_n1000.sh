#!/usr/bin/env bash
set -euo pipefail

ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
PY="${PY:-<ANON_ROOT>/micromamba-root/envs/eccts310/bin/python}"
SCRIPT="$ROOT/paper_neurips2026_artifacts/scripts/run_asca_sensitivity.py"

SETTAG="${SETTAG:?SETTAG required}"
W_SUP="${W_SUP:?W_SUP required}"
W_VALID="${W_VALID:?W_VALID required}"
W_BASE="${W_BASE:?W_BASE required}"
W_RISK="${W_RISK:?W_RISK required}"

ACC="${ACC:-<ANON_ACCOUNT>}"
TIME="${TIME:-08:00:00}"
CPUS="${CPUS:-8}"
MEM="${MEM:-32G}"
BENCHMARKS="${BENCHMARKS:-textvqa,ocrvqa,chartqa,ocrbench,gqa,ai2d,mme_rw,coco,amber}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

OUT_BASE="${OUT_BASE:-$ROOT/paper_neurips2026_artifacts/sensitivity_asca_weights/full4param_9bench_${RUN_TAG}}"
OUT_DIR="${OUT_BASE}/${SETTAG}"

mkdir -p "$ROOT/logs" "$OUT_DIR"

sbatch \
  --account="$ACC" \
  --job-name="sens4p-${SETTAG}" \
  --nodes=1 --ntasks=1 \
  --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" \
  --gres=gpu:1 \
  --output="$ROOT/logs/sens4p-${SETTAG}-%j.out" \
  --error="$ROOT/logs/sens4p-${SETTAG}-%j.err" \
  --wrap="cd $ROOT && $PY $SCRIPT --sweep single --n 1000 --benchmarks $BENCHMARKS --w_sup $W_SUP --w_valid $W_VALID --w_base $W_BASE --w_risk $W_RISK --varied_param $SETTAG --param_value $SETTAG --output_dir $OUT_DIR"

