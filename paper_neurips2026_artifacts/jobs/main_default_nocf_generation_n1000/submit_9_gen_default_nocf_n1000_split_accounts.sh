#!/usr/bin/env bash
set -euo pipefail

cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

RUN_TAG="${RUN_TAG:-genDefault_$(date +%Y%m%d_%H%M%S)}"
PART="${PART:-}"  # optional; keep empty unless you want to force partition

accounts=(
  "<ANON_ACCOUNT>"
  "<ANON_ACCOUNT>"
  "<ANON_ACCOUNT>"
  "<ANON_ACCOUNT>"
  "<ANON_ACCOUNT>"
)

benches=(textvqa ocrvqa chartqa ocrbench gqa ai2d mme_rw coco amber)

for i in "${!benches[@]}"; do
  b="${benches[$i]}"
  acc="${accounts[$((i % ${#accounts[@]}))]}"
  echo "[SUBMIT] ${b} -> ${acc} (RUN_TAG=${RUN_TAG} PART=${PART:-auto})"
  ACC="${acc}" PART="${PART}" RUN_TAG="${RUN_TAG}" BENCH="${b}" \
    bash paper_neurips2026_artifacts/jobs/main_default_nocf_generation_n1000/sbatch_gen_default_nocf_bench_n1000.sh
done

echo "[DONE] submitted 9 default-nocf generation jobs (n=1000)"

