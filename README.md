# CARE: Constrained Candidate Ranking with Gated Evidence Switching

Anonymous code release for our NeurIPS 2026 submission:

**Constrained Candidate Ranking with Gated Evidence Switching for Small Vision-language Models**

Anonymous project page:

https://anonymous.4open.science/r/CARE-release-E4B1/

## Overview

CARE is a training-free test-time scaling method for small vision-language models. Instead of increasing the generation budget or modifying model weights, CARE improves the final prediction by selecting a more reliable answer from the candidate pool produced by deterministic visual augmentations.

The main idea is that the correct answer often already appears among augmented candidates, but generic aggregation methods may select a distractor. CARE therefore focuses on the answer-selection bottleneck.

CARE consists of two components:

1. **Constrained Candidate Ranking**  
   CARE ranks candidate answers within a question-specific answer space using format-aware validity and quality-aware consistency.

2. **Gated Evidence Switching**  
   CARE optionally applies a sparse counterfactual verifier to switch from the top-ranked answer to an alternative only when the evidence strongly supports the switch and the ranking margin is small.

## Method

Given an image-question pair, CARE first constructs multiple augmented views and queries the same small VLM on each view. This produces a candidate answer pool:

\[
A_i = \{y_i^k\}_{k=1}^{K}.
\]

CARE then scores each candidate answer using a constrained ranking function. The implementation follows the same principle as the paper method:

\[
s_{\text{CARE}}(a)
=
w_{\text{valid}} s_{\text{valid}}(a)
+
w_{\text{freq}} s_{\text{freq}}(a)
+
w_{\text{base}} s_{\text{base}}(a)
-
w_{\text{risk}} s_{\text{risk}}(a).
\]

The score includes:

- `s_valid(a)`: whether the candidate matches the expected answer format.
- `s_freq(a)`: how frequently the candidate appears across augmented views.
- `s_base(a)`: whether the candidate agrees with the original-view response.
- `s_risk(a)`: length or verbosity risk, used to penalize overly long or over-specific answers.

The default ranking weights are:

```text
w_freq  = 2.0
w_valid = 1.0
w_base  = 0.4
w_risk  = 0.5

After ranking, CARE can optionally apply gated evidence switching. For a candidate answer, CARE compares its likelihood under two counterfactual visual perturbations:

dropping relevant regions;
dropping control regions.

The raw counterfactual score is:

CF
raw
	​
(a)=logP
ctrl
	​
(a)−logP
rel
	​
(a).

The counterfactual score is normalized within the candidate pool. CARE only switches from the top-ranked answer to an alternative when both conditions hold:

CF(a
1
	​
)−CF(a
0
	​
)>τ
CF
	​
,

and

s
CARE
	​
(a
0
	​
)−s
CARE
	​
(a
1
	​
)≤τ
gap
	​
.

Thus, counterfactual evidence is not used as a global reranker. It is used only as a gated calibration step for uncertain cases.

Repository Layout
benchmark_configs/              Experiment configuration files
scripts/                        Launch and utility scripts
vlmeval/                        Core evaluation and model code
paper_neurips2026_artifacts/    Scripts for paper tables, figures, and analysis
run.py                          Main evaluation entry point
requirements.txt                Python dependencies
Quick Start

Create an environment and install dependencies:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Run one benchmark configuration:

python run.py \
  --config benchmark_configs/<config_name>.json \
  --work-dir benchmark_results/<run_name>

Typical cluster launch:

bash scripts/sbatch_clean.sh \
  benchmark_configs/<config_name>.json \
  SUBSET_LEN=1000
Reproducing Main Experiments

The main experiments use deterministic multi-view augmentation with the same candidate pool for TTAug and CARE. CARE changes only the final candidate selection unless the optional gated evidence-switching route is enabled.

Example command:

python run.py \
  --config benchmark_configs/test_config_smolvlm2_v91_cf3_routed_textvqa.json \
  --work-dir benchmark_results/n_samples_1000/test_config_smolvlm2_v91_cf3_routed_textvqa \
  --verbose

For paper artifact generation, see:

paper_neurips2026_artifacts/

This directory contains scripts for producing tables, ablations, sensitivity analysis, and diagnostic outputs used in the paper.

Benchmarks

The paper evaluates CARE on nine benchmarks:

OCRBench
GQA
TextVQA
ChartQA
OCRVQA
AI2D
MME-RealWorld
COCO Captions
AMBER

Each benchmark is reported with its native evaluation metric. COCO Captions is evaluated with ROUGE-L.

Reproducibility Notes
All experiments are training-free.
Model weights are fixed.
CARE uses the same deterministic augmented candidate pool as TTAug.
The default number of augmented views is K = 8.
The optional counterfactual verifier is routed and strictly gated.
Full CARE adds verification overhead only on a sparse subset of uncertain cases.
Citation

This repository is currently anonymous for review. Citation information will be added after the review process.

License

Code is provided for academic research and anonymous review.
