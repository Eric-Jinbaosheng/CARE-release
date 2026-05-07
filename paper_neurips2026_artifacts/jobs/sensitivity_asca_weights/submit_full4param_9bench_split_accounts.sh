#!/usr/bin/env bash
set -euo pipefail

cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BENCHMARKS="${BENCHMARKS:-textvqa,ocrvqa,chartqa,ocrbench,gqa,ai2d,mme_rw,coco,amber}"

accounts=(
  "<ANON_ACCOUNT>"
  "<ANON_ACCOUNT>"
  "<ANON_ACCOUNT>"
  "<ANON_ACCOUNT>"
  "<ANON_ACCOUNT>"
)

settings=(
  "default 2.0 1.0 0.4 0.5"
  "wsup_1p0 1.0 1.0 0.4 0.5"
  "wsup_1p5 1.5 1.0 0.4 0.5"
  "wsup_2p5 2.5 1.0 0.4 0.5"
  "wsup_3p0 3.0 1.0 0.4 0.5"
  "wvalid_0p25 2.0 0.25 0.4 0.5"
  "wvalid_0p5 2.0 0.5 0.4 0.5"
  "wvalid_1p5 2.0 1.5 0.4 0.5"
  "wvalid_2p0 2.0 2.0 0.4 0.5"
  "wbase_0p0 2.0 1.0 0.0 0.5"
  "wbase_0p2 2.0 1.0 0.2 0.5"
  "wbase_0p6 2.0 1.0 0.6 0.5"
  "wbase_0p8 2.0 1.0 0.8 0.5"
  "wrisk_0p0 2.0 1.0 0.4 0.0"
  "wrisk_0p25 2.0 1.0 0.4 0.25"
  "wrisk_0p75 2.0 1.0 0.4 0.75"
  "wrisk_1p0 2.0 1.0 0.4 1.0"
)

i=0
for row in "${settings[@]}"; do
  read -r tag wsup wvalid wbase wrisk <<< "$row"
  acc="${accounts[$((i % ${#accounts[@]}))]}"
  echo "[SUBMIT] $tag -> $acc (RUN_TAG=$RUN_TAG)"
  ACC="$acc" RUN_TAG="$RUN_TAG" BENCHMARKS="$BENCHMARKS" \
  SETTAG="$tag" W_SUP="$wsup" W_VALID="$wvalid" W_BASE="$wbase" W_RISK="$wrisk" \
  bash paper_neurips2026_artifacts/jobs/sensitivity_asca_weights/sbatch_sensitivity_full4param_setting_n1000.sh
  i=$((i + 1))
done

echo "[DONE] submitted ${#settings[@]} jobs"
echo "Output root: <ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/sensitivity_asca_weights/full4param_9bench_${RUN_TAG}"

