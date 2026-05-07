# CF Coverage-Risk Report

Key message supported by this analysis:
- Ungated / force-switch CF variants can cause high-risk switching and severe drops.
- Strictly gated variants have much smaller coverage and lower risk.
- CF is best framed as a sparse verifier, not a broad score booster.

Notes:
- Rescue/harm counts are proxy estimates when official per-sample correctness is unavailable.
- TextVQA focused force-grid row can be overridden via --textvqa-cf-override (default 72.28).

Outputs:
- `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/cf_coverage_risk.csv`
- `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/cf_coverage_risk.tex`
- figure generation skipped: `matplotlib` not available in current environment.
