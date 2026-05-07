# Priority Execution Status (2026-04-30)

## 1) Queue cancellation request
- Checked queue first (`squeue -u <ANON_USER>`): no pending jobs at submission start.
- So there was nothing to cancel before this batch.

## 2) Account used for submissions
- Requested account string `<ANON_ACCOUNT>` is invalid on Slurm (account/partition mismatch).
- Validated and used: `<ANON_ACCOUNT>`.

## 3) Non-GPU artifacts (already consolidated)
Directory:
- `paper_neurips2026_artifacts/final_priority_bundle/non_gpu/`

Key files inside:
- `ALL_NON_GPU_ARTIFACTS.md`
- `answer_space_rules.md`
- `answer_space_rules.tex`
- `answer_space_rules.json`
- `qualitative_examples.md`
- `qualitative_examples_compact.tex`
- `bootstrap_ci_main.tex`
- `bootstrap_ci_report.md`
- `compute_cost_table.tex`
- `compute_cost_report.md`
- `cf_coverage_risk_report.md`
- `final_neurips_gap_report.md`

## 4) GPU jobs submitted by priority
Submission log:
- `paper_neurips2026_artifacts/reports/submitted_jobs_20260430.tsv`

Counts:
- `P1` TextVQA routed CF n=1000 final: **1**
- `P2` CF sensitivity sweep (tau/gate grid): **13** (done-runs auto-skipped)
- `P3` Second backbone sanity (Ovis2-1B, n=200): **4**
- `P4` ASCA component ablations n=200 (5 benchmarks x 8 variants): **40**
- `P5` ASCA priority ablations n=1000 (5 benchmarks x 5 variants): **25**

Total submitted this batch: **83 jobs**.

## 5) Running / pending snapshot
Current queue snapshot command:
- `squeue -u <ANON_USER>`

Observed immediately after submission:
- Running: `p1-tvq-routed1k` (job `7605046`)
- Remaining jobs pending with reason `QOSGrpGRES`.

## 6) Important paths
- Root: `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean`
- Logs: `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/logs`
- Results: `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results`
- Job list TSV: `paper_neurips2026_artifacts/reports/submitted_jobs_20260430.tsv`

