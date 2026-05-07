#!/usr/bin/env bash
set -euo pipefail

ROOT=<ANON_ROOT>/peking/smolvlm2_paper/ets_clean
PY=<ANON_ROOT>/micromamba-root/envs/eccts310/bin/python
SCRIPT=$ROOT/paper_neurips2026_artifacts/scripts/run_asca_sensitivity.py

ACC=${ACC:-<ANON_ACCOUNT>}
PART=${PART:-}

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$ROOT/paper_neurips2026_artifacts/sensitivity_asca_weights/groupA_nocf_wsup_wvalid_4bench_n1000_${STAMP}
mkdir -p "$OUT"

cat > "$OUT/run.sh" << 'RUNEOF'
#!/usr/bin/env bash
set -euo pipefail
cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean
export MPLCONFIGDIR=/tmp/mpl-<ANON_USER>
mkdir -p "$MPLCONFIGDIR"
PY=<ANON_ROOT>/micromamba-root/envs/eccts310/bin/python
OUT_DIR=__OUT_DIR__

# w_sup sweep
for wsup in 1.0 1.5 2.0 2.5 3.0; do
  echo "[RUN] w_sup=${wsup}"
  $PY paper_neurips2026_artifacts/scripts/run_asca_sensitivity.py \
    --sweep single \
    --n 1000 \
    --benchmarks chartqa,textvqa,ocrvqa,ocrbench \
    --w_sup "$wsup" --w_valid 1.0 --w_base 0.4 --w_risk 0.5 \
    --varied_param w_sup --param_value "$wsup" \
    --default-method nocf \
    --official_eval --require_official_eval \
    --full-support-only \
    --output_dir "$OUT_DIR/wsup_${wsup}"
done

# w_valid sweep
for wvalid in 0.25 0.5 1.0 1.5 2.0; do
  echo "[RUN] w_valid=${wvalid}"
  $PY paper_neurips2026_artifacts/scripts/run_asca_sensitivity.py \
    --sweep single \
    --n 1000 \
    --benchmarks chartqa,textvqa,ocrvqa,ocrbench \
    --w_sup 2.0 --w_valid "$wvalid" --w_base 0.4 --w_risk 0.5 \
    --varied_param w_valid --param_value "$wvalid" \
    --default-method nocf \
    --official_eval --require_official_eval \
    --full-support-only \
    --output_dir "$OUT_DIR/wvalid_${wvalid}"
done

echo "[DONE] outputs in $OUT_DIR"
RUNEOF

sed -i "s#__OUT_DIR__#$OUT#g" "$OUT/run.sh"
chmod +x "$OUT/run.sh"

SBATCH_ARGS=(
  --account="$ACC"
  --job-name="sensA-nocf-4b"
  --cpus-per-task=8
  --mem=96G
  --time=12:00:00
  --output="$ROOT/logs/sensA-nocf-4b-%j.out"
  --error="$ROOT/logs/sensA-nocf-4b-%j.err"
)
if [[ -n "$PART" ]]; then
  SBATCH_ARGS+=(--partition="$PART")
fi

jid=$(sbatch "${SBATCH_ARGS[@]}" --wrap="cd $ROOT && bash $OUT/run.sh" | awk '{print $4}')
echo "SUBMITTED_JOB=$jid"
echo "OUTPUT_DIR=$OUT"
echo "RUN_SCRIPT=$OUT/run.sh"
