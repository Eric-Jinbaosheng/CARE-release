# Compute Cost Report

This report combines measured wall-clock runtime from start/finish log lines with deterministic call-count estimates.

- AS-TTA (V91-NoCF) uses the same 8 deterministic views as TTAug; overhead is in aggregation/rerank logic.
- Optional CF verifier adds candidate log-likelihood evaluations only on routed subsets (see CF usage and logprob calls/sample).

## Output
- `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/compute_cost_table.csv`
- `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/compute_cost_table.tex`
