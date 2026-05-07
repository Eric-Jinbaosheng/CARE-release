#!/usr/bin/env bash
set -euo pipefail
cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

ACC="<ANON_ACCOUNT>" PART="l40s_public" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_ovis2_2b_wvalid025_n1000/sbatch_ovis2_2b_asca_wvalid025_textvqa_n1000.sh
ACC="<ANON_ACCOUNT>" PART="l40s_public" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_ovis2_2b_wvalid025_n1000/sbatch_ovis2_2b_asca_wvalid025_ocrvqa_n1000.sh
ACC="<ANON_ACCOUNT>" PART="l40s_public" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_ovis2_2b_wvalid025_n1000/sbatch_ovis2_2b_asca_wvalid025_chartqa_n1000.sh
ACC="<ANON_ACCOUNT>" PART="l40s_public" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_ovis2_2b_wvalid025_n1000/sbatch_ovis2_2b_asca_wvalid025_gqa_n1000.sh
