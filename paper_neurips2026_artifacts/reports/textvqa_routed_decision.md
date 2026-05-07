# TextVQA Routed CF n=1000 Decision

## Current status
- Config exists:
  - `benchmark_configs/test_config_smolvlm2_v91_cf3_routed_textvqa.json`
- A dedicated submit script is provided:
  - `paper_neurips2026_artifacts/jobs/submit_textvqa_routed_n1000.sh`
- In `experiment_metric_index_20260429.json`, TextVQA routed CF n=1000 aggregate row is currently missing.

## What is valid now
- Focused TextVQA CF check is available and should be reported explicitly as focused evidence:
  - TTAug baseline: **72.28**
  - V91-NoCF: **71.96**
  - V91-CF3 force-grid: **72.28**
  - Recorded rescue/harm/net: **9 / 4 / +5**
  - CF used: **23/1000**

## Paper-safe phrasing if routed n=1000 is still unavailable
- Keep V91-NoCF as main method.
- Report TextVQA CF as a focused verifier check (force-grid strict-gate), not a universal routed-CF conclusion.
- Keep ungated/force-switch CF as negative ablations.

## Command
```bash
cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean
bash paper_neurips2026_artifacts/jobs/submit_textvqa_routed_n1000.sh
```
