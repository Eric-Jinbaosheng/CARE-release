#!/usr/bin/env bash
set -euo pipefail
cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean
ACC="${ACC1:-<ANON_ACCOUNT>}" PART="${PART1:-}" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_internvl2_5_2b_n1000/sbatch_internvl2_5_2b_asca_n1000.sh textvqa
ACC="${ACC2:-<ANON_ACCOUNT>}" PART="${PART2:-}" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_internvl2_5_2b_n1000/sbatch_internvl2_5_2b_asca_n1000.sh ocrvqa
ACC="${ACC3:-<ANON_ACCOUNT>}" PART="${PART3:-}" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_internvl2_5_2b_n1000/sbatch_internvl2_5_2b_asca_n1000.sh chartqa
ACC="${ACC4:-<ANON_ACCOUNT>}" PART="${PART4:-}" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_internvl2_5_2b_n1000/sbatch_internvl2_5_2b_asca_n1000.sh gqa
