#!/usr/bin/env bash
set -euo pipefail
cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

ACC="<ANON_ACCOUNT>" PART="l40s_public,a100_tandon,h100_t" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_ovis2_2b_n1000/sbatch_ovis2_2b_asca_textvqa_n1000.sh
ACC="<ANON_ACCOUNT>" PART="l40s_public,a100_tandon,h100_t" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_ovis2_2b_n1000/sbatch_ovis2_2b_asca_ocrvqa_n1000.sh
ACC="<ANON_ACCOUNT>" PART="l40s_public,a100_tandon,h100_t" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_ovis2_2b_n1000/sbatch_ovis2_2b_asca_chartqa_n1000.sh
ACC="<ANON_ACCOUNT>" PART="l40s_public,a100_tandon,h100_t" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_ovis2_2b_n1000/sbatch_ovis2_2b_asca_gqa_n1000.sh
