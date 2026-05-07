#!/usr/bin/env bash
set -euo pipefail
cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

ACC="<ANON_ACCOUNT>" PART="l40s_public" BENCH="textvqa" \
  bash paper_neurips2026_artifacts/jobs/sbatch_ablation_no_base_consistency_bench_n1000.sh

ACC="<ANON_ACCOUNT>" PART="l40s_public" BENCH="ocrvqa" \
  bash paper_neurips2026_artifacts/jobs/sbatch_ablation_no_base_consistency_bench_n1000.sh

ACC="<ANON_ACCOUNT>" PART="l40s_public" BENCH="chartqa" \
  bash paper_neurips2026_artifacts/jobs/sbatch_ablation_no_base_consistency_bench_n1000.sh

ACC="<ANON_ACCOUNT>" PART="l40s_public" BENCH="ocrbench" \
  bash paper_neurips2026_artifacts/jobs/sbatch_ablation_no_base_consistency_bench_n1000.sh

ACC="<ANON_ACCOUNT>" PART="l40s_public" BENCH="gqa" \
  bash paper_neurips2026_artifacts/jobs/sbatch_ablation_no_base_consistency_bench_n1000.sh
