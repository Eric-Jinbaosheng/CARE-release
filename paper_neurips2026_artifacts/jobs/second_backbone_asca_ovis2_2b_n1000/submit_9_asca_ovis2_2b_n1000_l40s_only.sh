#!/usr/bin/env bash
set -euo pipefail
cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

# L40S-only submission to avoid partition incompatibility on some accounts.
ACC="<ANON_ACCOUNT>" PART="l40s_public" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_ovis2_2b_n1000/sbatch_ovis2_2b_asca_textvqa_n1000.sh
ACC="<ANON_ACCOUNT>" PART="l40s_public" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_ovis2_2b_n1000/sbatch_ovis2_2b_asca_ocrvqa_n1000.sh
ACC="<ANON_ACCOUNT>" PART="l40s_public" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_ovis2_2b_n1000/sbatch_ovis2_2b_asca_chartqa_n1000.sh
ACC="<ANON_ACCOUNT>" PART="l40s_public" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_ovis2_2b_n1000/sbatch_ovis2_2b_asca_gqa_n1000.sh
ACC="<ANON_ACCOUNT>" PART="l40s_public" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_ovis2_2b_n1000/sbatch_ovis2_2b_asca_ocrbench_n1000.sh
ACC="<ANON_ACCOUNT>" PART="l40s_public" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_ovis2_2b_n1000/sbatch_ovis2_2b_asca_ai2d_n1000.sh
ACC="<ANON_ACCOUNT>" PART="l40s_public" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_ovis2_2b_n1000/sbatch_ovis2_2b_asca_mme_rw_n1000.sh
ACC="<ANON_ACCOUNT>" PART="l40s_public" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_ovis2_2b_n1000/sbatch_ovis2_2b_asca_coco_n1000.sh
ACC="<ANON_ACCOUNT>" PART="l40s_public" bash paper_neurips2026_artifacts/jobs/second_backbone_asca_ovis2_2b_n1000/sbatch_ovis2_2b_asca_amber_n1000.sh
