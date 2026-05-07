# CARE: Constrained Candidate Ranking with Gated Evidence Switching

Anonymous code release for our NeurIPS 2026 submission.

## What this repo contains
- Deterministic multi-view test-time augmentation (TTAug) evaluation pipeline.
- **CARE** selector for candidate reranking.
- Optional gated counterfactual verifier (sparse switch route).
- Benchmark/evaluation scripts and paper artifact scripts.

## Method (short)
Given candidate answers from K augmented views, CARE scores each candidate with:

\[
\text{score}(a) = w_{sup}\,\text{support}(a) + w_{valid}\,\text{valid}(a) + w_{base}\,\text{base}(a) - w_{risk}\,\text{risk}(a)
\]

Default weights used in our main setting:
- `w_sup = 2.0`
- `w_valid = 1.0`
- `w_base = 0.4`
- `w_risk = 0.5`

Optional CF verifier is **strictly gated** and only applied to a sparse routed subset.

## Repository layout
- `vlmeval/` core model/dataset/inference code
- `benchmark_configs/` runnable experiment configs
- `scripts/` launch and utility scripts
- `paper_neurips2026_artifacts/` tables/figures/analysis scripts used for paper artifacts

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run one config:
```bash
python run.py --config benchmark_configs/<your_config>.json
```

Typical cluster run helper:
```bash
bash scripts/sbatch_clean.sh benchmark_configs/<your_config>.json SUBSET_LEN=1000
```

## Reproducibility notes
- Main comparisons use the same deterministic augmentation policy and evaluator across methods.
- CARE (no-switch) uses the same generated candidate pool as TTAug and changes only selection.
- Full CARE adds routed verification overhead only on a small subset.

## License
Code is provided for academic research and anonymous review.
