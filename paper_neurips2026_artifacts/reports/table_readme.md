# Table Readme

## Sources
- metric index: `logs/experiment_metric_index_20260429.json`
- diagnostics: `.runtime_cache/test_config_*/diagnostics/v91*samples.jsonl`

## Important caveats
- Main tables are n=1000 only.
- `cf_textvqa_focused` uses focused force-grid CF check.
- TextVQA routed CF may be missing; this script can apply `--textvqa-cf-override` (default 72.08) to avoid silent NA in draft tables.
- Do not present override as routed-CF universal result.

## Generated files
- `paper_neurips2026_artifacts/tables/main_results_n1000.csv`
- `paper_neurips2026_artifacts/tables/main_results_n1000.tex`
- `paper_neurips2026_artifacts/tables/cf_results_n1000.csv`
- `paper_neurips2026_artifacts/tables/cf_results_n1000.tex`
- `paper_neurips2026_artifacts/tables/cf_textvqa_focused.csv`
- `paper_neurips2026_artifacts/tables/cf_textvqa_focused.tex`
- `paper_neurips2026_artifacts/tables/cf_bad_ablation.csv`
- `paper_neurips2026_artifacts/tables/cf_bad_ablation.tex`
- `paper_neurips2026_artifacts/tables/cf_usage_table.csv`
