#!/usr/bin/env bash
set -euo pipefail
cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean
export MPLCONFIGDIR=/tmp/mpl-<ANON_USER>
mkdir -p "$MPLCONFIGDIR"
PY=<ANON_ROOT>/micromamba-root/envs/eccts310/bin/python
OUT_DIR=<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/sensitivity_asca_weights/groupA_nocf_wsup_wvalid_4bench_n1000_20260506_081136

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
    --output_dir "$OUT_DIR/wvalid_${wvalid}"
done

echo "[DONE] outputs in $OUT_DIR"
