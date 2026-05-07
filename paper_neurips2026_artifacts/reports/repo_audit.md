# Repo Audit (NeurIPS 2026 artifacts)

## 1. Core paths

- Repo root: `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean`
- Configs: `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_configs`
- Results: `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results`
- Logs: `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/logs`
- Benchmark scripts: `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/scripts`

## 2. Where things live

- Config JSONs: `benchmark_configs/*.json`
- Result folders: `benchmark_results/n_samples_{N}/test_config_*/{ModelName}/...`
- Aggregate scores: `*_acc.csv`, `*_score.json`, `*_score.csv`, `*_rating.json`
- Per-sample outputs often in: `*.xlsx`, sometimes `*.csv`/`*.jsonl` next to aggregates
- Slurm wrappers: `scripts/sbatch_clean.sh`, `scripts/sbatch_clean_fast.sh`

## 3. n=1000 availability matrix (base / TTAug / noCF / routed / force-grid)

| Benchmark | TTAug baseline | V91-NoCF | V91-CF3-routed | TextVQA force-grid check |
|---|---|---|---|---|
| ocrbench | YES | YES | YES | - |
| gqa | YES | YES | YES | - |
| textvqa | YES | YES | NO | YES |
| chartqa | YES | YES | YES | - |
| ocrvqa | YES | YES | YES | - |
| ai2d | YES | YES | YES | - |
| mme_rw | YES | YES | YES | - |
| coco | YES | YES | YES | - |
| amber | YES | YES | YES | - |

## 4. Missing results

- Missing for base/noCF main table (n=1000): `none`
- Missing routed-CF n=1000 rows: `['textvqa']`
- Known gap from this cycle: `textvqa` routed n=1000 lacks finalized aggregate file; only partial artifact exists.

## 5. What can be computed immediately (no reruns)

- Main n=1000 baseline vs V91-NoCF table from `experiment_metric_index_20260429.*`.
- CF negative ablations on TextVQA n=200 (no_quality_gate / force_switch / force_grid variants).
- Compute-cost call-count estimates from logs (`[V91-RERANK]`, progress bars, wall-clock lines).
- Answer-space rules export from `vlmeval/vlm/tta/tta_v91_aggregator.py`.

## 6. What needs GPU reruns

- TextVQA routed CF n=1000 (if final table requires routed row).
- NoCF component ablations n=200 / n=1000 priority set.
- Optional second-backbone experiments if supported and time allows.

## 7. Existing run scripts

- sbatch wrappers: `sbatch_clean.sh, sbatch_clean_fast.sh`
- submit scripts: `submit_deterministic_sanity.sh, submit_eccts_clean.sh, submit_equivalence_check.sh, submit_paper_reproduction.sh, submit_v91.sh, submit_v91_cf2_debug.sh, submit_v91_cf3_debug.sh`

## 8. Next commands (exact)

```bash
cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean
python paper_neurips2026_artifacts/scripts/make_paper_tables.py --textvqa-cf-override 72.28
python paper_neurips2026_artifacts/scripts/bootstrap_ci.py --n-boot 10000
python paper_neurips2026_artifacts/scripts/compute_cost_report.py
python paper_neurips2026_artifacts/scripts/cf_coverage_risk_analysis.py --textvqa-cf-override 72.28
python paper_neurips2026_artifacts/scripts/export_answer_space_rules.py
```
