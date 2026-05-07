#!/usr/bin/env bash
set -euo pipefail
cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

# Prioritized 5 benchmarks (GQA may skip automatically if diagnostics missing).
ACC="<ANON_ACCOUNT>" PART="l40s_public" BENCH="textvqa" \
  bash paper_neurips2026_artifacts/jobs/sensitivity_asca_weights/sbatch_sensitivity_asca_bench_n1000.sh

ACC="<ANON_ACCOUNT>" PART="l40s_public" BENCH="ocrvqa" \
  bash paper_neurips2026_artifacts/jobs/sensitivity_asca_weights/sbatch_sensitivity_asca_bench_n1000.sh

ACC="<ANON_ACCOUNT>" PART="l40s_public" BENCH="chartqa" \
  bash paper_neurips2026_artifacts/jobs/sensitivity_asca_weights/sbatch_sensitivity_asca_bench_n1000.sh

ACC="<ANON_ACCOUNT>" PART="l40s_public" BENCH="ocrbench" \
  bash paper_neurips2026_artifacts/jobs/sensitivity_asca_weights/sbatch_sensitivity_asca_bench_n1000.sh

ACC="<ANON_ACCOUNT>" PART="l40s_public" BENCH="gqa" \
  bash paper_neurips2026_artifacts/jobs/sensitivity_asca_weights/sbatch_sensitivity_asca_bench_n1000.sh
