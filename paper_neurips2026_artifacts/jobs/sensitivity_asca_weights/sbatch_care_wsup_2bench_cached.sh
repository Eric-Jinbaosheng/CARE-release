#!/usr/bin/env bash
set -euo pipefail
ROOT=<ANON_ROOT>/peking/smolvlm2_paper/ets_clean
PY=<ANON_ROOT>/micromamba-root/envs/eccts310/bin/python
OUT=$ROOT/paper_neurips2026_artifacts/sensitivity_asca_weights/wsup_only_2bench_chartqa_textvqa
mkdir -p "$OUT"
cd "$ROOT"

for wsup in 1.0 1.5 2.0 2.5 3.0; do
  "$PY" paper_neurips2026_artifacts/scripts/run_asca_sensitivity.py \
    --sweep single \
    --n 1000 \
    --benchmarks chartqa,textvqa \
    --w_sup "$wsup" \
    --w_valid 1.0 \
    --w_base 0.4 \
    --w_risk 0.5 \
    --varied_param w_sup \
    --param_value "$wsup" \
    --output_dir "$OUT/wsup_${wsup}"
done

# merge all per-setting raw into one csv
"$PY" - << 'PY'
import csv
from pathlib import Path
root=Path('<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/sensitivity_asca_weights/wsup_only_2bench')
out=root/'care_wsup_2bench_summary.csv'
rows=[]
for p in sorted(root.glob('wsup_*/sensitivity_raw.csv')):
    with p.open() as f:
        r=csv.DictReader(f)
        rows.extend(list(r))
fields=list(rows[0].keys()) if rows else []
with out.open('w',newline='') as f:
    if fields:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
print('[DONE]',out)
PY
