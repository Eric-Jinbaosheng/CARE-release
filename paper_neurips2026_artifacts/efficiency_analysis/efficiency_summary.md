# Efficiency Summary

Representative setting: **TextVQA, n=1000** (existing logs + diagnostics).

| Method | Gen./sample | Verifier evals/sample | Route rate | Switch rate | Latency/sample | Rel. cost vs TTAug |
|---|---:|---:|---:|---:|---:|---:|
| Base | 1.00 | 0.00 | 0.00% | 0.00% | TODO | TODO |
| TTAug | 8.00 | 0.00 | 0.00% | 0.00% | 8.22s | 1.00x |
| CARE w/o switch | 8.00 | 0.00 | 0.00% | 0.00% | TODO | TODO |
| CARE full | 8.00 | 5.16 | 4.55% | 3.45% | TODO | TODO |

## Notes
- Latency is parsed from `Starting ...` / `Finished ...` log timestamps.
- `CARE full` route/switch/logprob statistics are computed from available diagnostic cache rows.
- Missing fields are marked `TODO` (no fabrication).

## Draft Subsection (LaTeX)

\subsection{Efficiency Analysis}
Efficiency matters because test-time gains are only useful when they do not rely on brute-force candidate expansion. CARE is designed to improve answer selection under a fixed candidate-generation budget: it keeps the same deterministic eight-view pool as TTAug and reallocates computation to constrained ranking and sparse verification.

\input{paper_neurips2026_artifacts/efficiency_analysis/efficiency_table_latex.tex}

\textbf{Obs. ❶ (same generation budget).} Base uses one generation per sample, while both TTAug and CARE use eight generations per sample. CARE w/o switch does not add candidate generation beyond TTAug; it only reranks the same candidate pool.

\textbf{Obs. ❷ (sparse verifier overhead).} CARE full introduces extra likelihood evaluation only through the routed verifier. In our representative run, routing and final switching are sparse (route rate and switch rate in Table~\ref{tab:efficiency_care}), indicating concentrated verification rather than global reranking. The efficiency gain should therefore be interpreted as better use of an existing candidate budget, not additional candidate generation.
