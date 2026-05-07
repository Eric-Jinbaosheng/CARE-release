# CARE: Constrained Candidate Ranking with Gated Evidence Switching

Anonymous code release for NeurIPS 2026 submission.

**Paper title**: *Constrained Candidate Ranking with Gated Evidence Switching for Small Vision-Language Models*

Anonymous project page:
- https://anonymous.4open.science/r/CARE-release-E4B1/

---

## Overview

CARE is a **training-free test-time scaling** method for small VLMs.
It does **not** change model weights and does **not** increase the candidate-generation budget beyond deterministic multi-view augmentation.

Instead, CARE improves the **answer selection stage**:
- build candidate answers from deterministic augmented views;
- rerank candidates with answer-space-aware constraints;
- optionally apply sparse gated counterfactual switching for uncertain cases.

---

## Method

For each sample, deterministic augmentation produces a candidate pool:

`A_i = { y_i^k }_{k=1..K}`

CARE score for candidate `a`:

```text
s_CARE(a) =
    w_fre  * s_freq(a)
  + w_val  * s_valid(a)
  + w_con  * s_base(a)
  - w_len  * s_risk(a)
```

Where:
- `s_freq(a)`: support frequency across augmented views;
- `s_valid(a)`: answer-space format validity;
- `s_base(a)`: consistency with original-view answer;
- `s_risk(a)`: verbosity/length risk penalty.

Default weights:

```text
w_fre  = 2.0
w_val  = 1.0
w_con  = 0.4
w_len  = 0.5
```

### Optional gated evidence switching

Raw CF score:

```text
CF_raw(a) = logP_ctrl(a) - logP_rel(a)
```

Switch from top-1 `a0` to alternative `a1` only when both hold:

```text
CF(a1) - CF(a0) > tau_cf
s_CARE(a0) - s_CARE(a1) <= tau_gap
```

This CF module is **sparse and gated**, not a global reranker.

---

## Repository Layout

- `benchmark_configs/` — experiment config files
- `scripts/` — launch and utility scripts
- `vlmeval/` — core evaluation and model code
- `paper_neurips2026_artifacts/` — scripts/tables/figures for paper artifacts
- `run.py` — main evaluation entry
- `requirements.txt` — dependencies

---

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run one config:

```bash
python run.py \
  --config benchmark_configs/<config_name>.json \
  --work-dir benchmark_results/<run_name>
```

Typical cluster launch:

```bash
bash scripts/sbatch_clean.sh \
  benchmark_configs/<config_name>.json \
  SUBSET_LEN=1000
```

---

## Main Experiment Notes

- Deterministic multi-view augmentation uses `K = 8` views.
- CARE uses the **same candidate pool** as TTAug.
- CARE (no-switch) changes only selection logic.
- CARE full adds routed/gated CF verification on a sparse subset.

Example:

```bash
python run.py \
  --config benchmark_configs/test_config_smolvlm2_v91_cf3_routed_textvqa.json \
  --work-dir benchmark_results/n_samples_1000/test_config_smolvlm2_v91_cf3_routed_textvqa \
  --verbose
```

---

## Benchmarks

The paper evaluates on nine benchmarks:
- OCRBench
- GQA
- TextVQA
- ChartQA
- OCRVQA
- AI2D
- MME-RealWorld
- COCO Captions
- AMBER

Each benchmark is reported with its native metric.

---

## Reproducibility

- Training-free evaluation only.
- Fixed model weights.
- Deterministic augmentation and consistent evaluator protocol.

---

## Citation

Repository is anonymous for review. Citation will be added after review.

## License

For academic research and anonymous review.
