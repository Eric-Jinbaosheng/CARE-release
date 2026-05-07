#!/usr/bin/env bash
set -euo pipefail

ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
cd "$ROOT"

ACCOUNT="${ACCOUNT:-<ANON_ACCOUNT>}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/.runtime_cache}"
LOG_TSV="paper_neurips2026_artifacts/reports/submitted_jobs_20260430.tsv"

mkdir -p "$(dirname "$LOG_TSV")"
printf "priority\tjob_name\tsubset\tconfig\tjobid\n" > "$LOG_TSV"

has_result() {
  local subset="$1"
  local cfg="$2"
  local stem
  stem="$(basename "$cfg" .json)"
  local d="$ROOT/benchmark_results/n_samples_${subset}/${stem}"
  [[ -d "$d" ]] || return 1
  find "$d" -type f \( -name "*_acc.csv" -o -name "*_score.json" -o -name "*_score.csv" \) | grep -q .
}

submit_one() {
  local priority="$1"
  local subset="$2"
  local cfg_rel="$3"
  local job_name="$4"
  local cfg_abs="$ROOT/$cfg_rel"

  if [[ ! -f "$cfg_abs" ]]; then
    echo "[MISS_CONFIG] $cfg_rel"
    return 0
  fi

  if has_result "$subset" "$cfg_abs"; then
    echo "[SKIP_DONE] n=${subset} $(basename "$cfg_abs")"
    return 0
  fi

  local cache_path="$CACHE_ROOT/$(basename "$cfg_abs" .json)_n${subset}"
  mkdir -p "$cache_path"

  local out
  out=$(sbatch \
    --account="$ACCOUNT" \
    --job-name="$job_name" \
    --export=ALL,CONFIG_PATH="$cfg_abs",SUBSET_LEN="$subset",CACHE_PATH="$cache_path" \
    scripts/sbatch_clean.sh "$cfg_abs")

  local jobid
  jobid=$(echo "$out" | awk '{print $4}')
  echo "[SUBMIT] $priority $job_name jobid=$jobid"
  printf "%s\t%s\t%s\t%s\t%s\n" "$priority" "$job_name" "$subset" "$cfg_rel" "$jobid" >> "$LOG_TSV"
}

# P1: TextVQA routed CF n=1000 final
submit_one "P1" 1000 "benchmark_configs/test_config_smolvlm2_v91_cf3_routed_textvqa.json" "p1-tvq-routed1k"

# P2: CF sensitivity sweep (appendix): TextVQA/OCRVQA n=200
for d in textvqa ocrvqa; do
  for g in g01 g02 g03 g04 g05 g06 g07 g08; do
    cfg="benchmark_configs/test_config_smolvlm2_v91_cf3_routed_${d}_${g}.json"
    submit_one "P2" 200 "$cfg" "p2-${d}-${g}"
  done
done

# P3: Second backbone sanity (Ovis2-1B) n=200
for b in textvqa ocrvqa gqa ocrbench; do
  cfg="paper_neurips2026_artifacts/configs/second_backbone_configs/test_config_ovis2_1b_ttaug_det_${b}.json"
  submit_one "P3" 200 "$cfg" "p3-ovis1b-${b}"
done

# P4: ASCA/NoCF component ablations n=200 (full) on 5 benchmarks
abls_200=(frequency_only no_format no_base_bias no_length_risk no_answer_space majority_vote base_only first_view)
bench=(textvqa ocrvqa gqa chartqa ocrbench)
for b in "${bench[@]}"; do
  for a in "${abls_200[@]}"; do
    cfg="paper_neurips2026_artifacts/configs/ablation_configs/test_config_smolvlm2_v91_nocf_ablation_${a}_${b}.json"
    submit_one "P4" 200 "$cfg" "p4-abl200-${a}-${b}"
  done
done

# P5: ASCA/NoCF component ablations n=1000 (priority subset)
abls_1000=(frequency_only no_format no_base_bias no_length_risk majority_vote)
for b in "${bench[@]}"; do
  for a in "${abls_1000[@]}"; do
    cfg="paper_neurips2026_artifacts/configs/ablation_configs/test_config_smolvlm2_v91_nocf_ablation_${a}_${b}.json"
    submit_one "P5" 1000 "$cfg" "p5-abl1k-${a}-${b}"
  done
done

echo "Saved submission log: $LOG_TSV"
