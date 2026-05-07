#!/usr/bin/env bash
set -euo pipefail

ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
cd "$ROOT"

SUBSET="${SUBSET:-1000}"

# Round-robin accounts you already use.
accounts=(
  "<ANON_ACCOUNT>"
  "<ANON_ACCOUNT>"
  "<ANON_ACCOUNT>"
  "<ANON_ACCOUNT>"
  "<ANON_ACCOUNT>"
)

benches=(
  "textvqa"
  "ocrvqa"
  "chartqa"
  "ocrbench"
  "gqa"
  "ai2d"
  "mme_rw"
  "coco"
  "amber"
)

for i in "${!benches[@]}"; do
  b="${benches[$i]}"
  acc="${accounts[$((i % ${#accounts[@]}))]}"
  cfg="$ROOT/paper_neurips2026_artifacts/configs/main_wsup1p5/test_config_smolvlm2_v91_nocf_wsup1p5_${b}.json"
  cache="$ROOT/.runtime_cache/test_config_smolvlm2_v91_nocf_wsup1p5_${b}_n${SUBSET}"
  job="wsup15-${b}-n${SUBSET}"

  echo "[SUBMIT] $job -> $acc"
  sbatch \
    --account="$acc" \
    --job-name="$job" \
    --export=ALL,CONFIG_PATH="$cfg",SUBSET_LEN="$SUBSET",CACHE_PATH="$cache" \
    scripts/sbatch_clean.sh "$cfg"
done

