#!/usr/bin/env bash
set -euo pipefail

ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
ACC="${ACC:-<ANON_ACCOUNT>}"
SUBSET=200

# Core ASCA ablations requested by paper gap:
# full (already exists), frequency_only, majority_vote,
# no_format(no contract validity), no_base_bias(no base consistency), no_length_risk
ABLS=(frequency_only majority_vote no_format no_base_bias no_length_risk)
BENCH=(textvqa ocrvqa gqa chartqa ocrbench)

cd "$ROOT"

has_result() {
  local stem="$1"
  local d="$ROOT/benchmark_results/n_samples_${SUBSET}/${stem}"
  [[ -d "$d" ]] || return 1
  find "$d" -type f \( -name '*_acc.csv' -o -name '*_score.json' -o -name '*_score.csv' \) | grep -q .
}

for b in "${BENCH[@]}"; do
  for a in "${ABLS[@]}"; do
    stem="test_config_smolvlm2_v91_nocf_ablation_${a}_${b}"
    cfg="$ROOT/paper_neurips2026_artifacts/configs/ablation_configs/${stem}.json"
    cache="$ROOT/.runtime_cache/${stem}_n${SUBSET}_r6"
    job="abl200-${a}-${b}-r6"

    if [[ ! -f "$cfg" ]]; then
      echo "[MISS_CFG] $cfg"
      continue
    fi

    if has_result "$stem"; then
      echo "[SKIP_DONE] $stem"
      continue
    fi

    mkdir -p "$cache"
    echo "[SUBMIT] $job"
    sbatch \
      --account="$ACC" \
      --job-name="$job" \
      --export=ALL,CONFIG_PATH="$cfg",SUBSET_LEN=${SUBSET},CACHE_PATH="$cache" \
      scripts/sbatch_clean.sh "$cfg"
  done
done
