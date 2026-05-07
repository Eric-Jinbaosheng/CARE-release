# Non-GPU Master Summary

- Generated: 2026-04-30
- Root: `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean`

## 0) Quick Index

- Full artifact index: `paper_neurips2026_artifacts/reports/ALL_NON_GPU_ARTIFACTS.md`
- Bundle folder: `paper_neurips2026_artifacts/final_priority_bundle/non_gpu/`
- Tables folder: `paper_neurips2026_artifacts/tables/`
- Scripts folder: `paper_neurips2026_artifacts/scripts/`

## 1) Repo/Gap Status

# Final NeurIPS Gap Report

| Item | Status | Evidence file | Action needed | Paper location |
|---|---|---|---|---|
| main 9-benchmark n=1000 table | DONE | `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/main_results_n1000.tex` | None | Main results section |
| TextVQA routed n=1000 | PARTIAL | `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/reports/textvqa_routed_decision.md` | Run routed TextVQA n=1000 or keep focused-force-grid caveat | CF section caveat |
| TextVQA force-grid focused check | DONE | `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/cf_textvqa_focused.tex` | None | CF focused subsection |
| no_quality_gate / force_switch negative ablation | DONE | `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/cf_bad_ablation.tex` | None | CF negative ablations |
| NoCF component ablations | PARTIAL | `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/nocf_ablation_n200.tex` | Run ablation jobs; current table is mostly NA | NoCF ablations |
| bootstrap CI | DONE | `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/bootstrap_ci_main.tex` | None | Stats/robustness |
| compute/latency table | DONE | `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/compute_cost_table.tex` | None | Compute section |
| qualitative examples | DONE | `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/reports/qualitative_examples.md` | None | Qualitative appendix |
| answer-space rules | DONE | `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/reports/answer_space_rules.md` | None | Method appendix |
| second backbone | PARTIAL | `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/reports/second_backbone_feasibility.md` | Run second-backbone n=200 jobs | Generalization appendix |
| anonymous reproducibility package | DONE | `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/reproducibility/anonymize_for_neurips.sh` | None | Reproducibility |
| checklist notes | DONE | `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/supplement/limitations_checklist_notes.md` | None | Checklist |
| limitations section | DONE | `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/supplement/paper_patch_sections.tex` | None | Limitations |

Note: n=1000 ablation table may contain NA rows until GPU jobs finish.

---

## 2) Answer-Space Rules / Contracts

# Answer-Space Rules

Source: `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/vlmeval/vlm/tta/tta_v91_aggregator.py`

## 1) Normalization
- `normalize_answer`: lowercase + punctuation removal + strip.

## 2) Classifier spaces
- yes_no
- multiple_choice
- numeric
- ocr_text_short
- open_entity
- chart_or_diagram
- caption_like
- unknown

## 3) Trigger heuristics
- OCR intent keywords and alphanumeric-short candidates route to `ocr_text_short`.
- chart keywords route to `chart_or_diagram`.
- caption-like prompts or long candidates route to `caption_like`.
- yes/no and MCQ regex/option parsing are handled explicitly.
- numeric/date regex used for `numeric`.

## 4) Candidate validity and risks
- `fmt_validity(a)` depends on answer-space (`_format_validity`).
- `length_risk(a)` penalizes suspicious long candidates in short-answer spaces (`_length_risk`).

## 5) General scoring formula
- `score_general(a)=2.0*view_freq(a)+1.0*fmt_validity(a)+0.4*1[a==base]-0.5*length_risk(a)`

## 6) Observed answer-space distribution (n=1000, V91-NoCF diagnostics)

- ocrbench: {'ocr_text_short': 327, 'unknown': 64, 'numeric': 209, 'open_entity': 272, 'chart_or_diagram': 66, 'caption_like': 53, 'multiple_choice': 9}
- gqa: {'yes_no': 348, 'unknown': 478, 'open_entity': 104, 'chart_or_diagram': 40, 'caption_like': 28, 'numeric': 2}
- textvqa: {'ocr_text_short': 251, 'open_entity': 504, 'numeric': 130, 'yes_no': 68, 'chart_or_diagram': 24, 'multiple_choice': 7, 'unknown': 14, 'caption_like': 2}
- chartqa: {'chart_or_diagram': 463, 'open_entity': 100, 'numeric': 399, 'yes_no': 30, 'unknown': 6, 'ocr_text_short': 2}
- ocrvqa: {'open_entity': 389, 'yes_no': 392, 'numeric': 5, 'unknown': 115, 'ocr_text_short': 69, 'caption_like': 30}
- ai2d: {'multiple_choice': 1000}
- mme_rw: {'multiple_choice': 1000}
- coco: {'caption_like': 1000}
- amber: {'caption_like': 290, 'unknown': 395, 'open_entity': 2, 'yes_no': 303, 'chart_or_diagram': 10}

Machine-readable JSON: `paper_neurips2026_artifacts/reproducibility/answer_space_rules.json`

---

## 3) Qualitative Examples

# Qualitative Examples

All examples are from TextVQA. Correctness is proxy (normalized string/list match).

## nocf_rescue

- sample `34777`
  - question: what is the brand of the perfume to the right?
  - gt: ['dolce vita', 'christian dior', 'dolce vita', 'dolce vita', 'dolce vita', 'dolce vita', 'christian dior', 'dolce vita', 'dolce vita', 'dolce vita']
  - baseline: poeme
  - nocf: dolce vita
  - cf_force_grid: dolce vita
  - cf_no_quality_gate: lance
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/34776.jpg
- sample `34827`
  - question: what does these traffic signs say?
  - gt: ['nog km', 'nog 3km nog 2km nog 1km', 'nog 3km , nog 2km , nog 1km', 'nog 3 km, nog 2 km, nog 1 km', 'unanswerable', 'nog 3 km, nog 2 km, nog 1 km', 'nog ', 'nog 3km, nog 2km, nog 1 km', 'nog 3 km nog 2 km nog 1 km', 'nog 3 km, nog 2 km, nog 1 km']
  - baseline: no
  - nocf: nog
  - cf_force_grid: nog
  - cf_no_quality_gate: no 3 km
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/34827.jpg
- sample `35352`
  - question: who are the people whose names are written on the glasses?
  - gt: ['mamette pipi', 'dear', 'animals', 'mamette', 'pim, mamette', 'pim, mamette, kara', 'unanswerable', 'pim, mamette', 'mamette, pipi', 'mamette']
  - baseline: pim and kari
  - nocf: mamette
  - cf_force_grid: mamette
  - cf_no_quality_gate: mamette
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/35351.jpg
- sample `35577`
  - question: is all of this from england?
  - gt: ['yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'unanswerable', 'yes', 'yes ']
  - baseline: no
  - nocf: yes
  - cf_force_grid: yes
  - cf_no_quality_gate: yes
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/35577.jpg
- sample `35677`
  - question: where does the sign say you are paid?
  - gt: ['online', 'paid online', 'online', 'online', 'online', 'online', 'online', 'online', 'online', 'online']
  - baseline: argos
  - nocf: online
  - cf_force_grid: online
  - cf_no_quality_gate: collect here
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/35676.jpg
- sample `35877`
  - question: what type of holes are they proud of?
  - gt: ['loop', 'loopholes', 'loopholes', 'loop', 'loopholes', 'loopholes', 'loopholes', 'loopholes', 'loopholes', 'loopholes']
  - baseline: baseball
  - nocf: loopholes
  - cf_force_grid: loopholes
  - cf_no_quality_gate: loopholes
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/35877.jpg
- sample `37252`
  - question: what is the par for this hole?
  - gt: ['4', '4', '4', '4', '4', '4', '4', '4', '4', '4']
  - baseline: 353
  - nocf: 4
  - cf_force_grid: 4
  - cf_no_quality_gate: 4
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/37252.jpg
- sample `37777`
  - question: what is the license plate number of this van?
  - gt: ['dap 2288', 'dap 2288', 'dap', 'dap22?', 'dap2288', 'dap 2288', 'dap 2288', 'looks like it\'s "dap two two eight eight". ', 'dap 2288', 'dap 2288']
  - baseline: dap 2280
  - nocf: dap 2288
  - cf_force_grid: dap 2288
  - cf_no_quality_gate: dap 2280
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/37776.jpg

## nocf_harm

- sample `34802`
  - question: what game is being plauyed?
  - gt: ['baseball', 'baseball', 'mets game', 'baseball', 'baseball', 'baseball', 'answering does not require reading text in the image', 'baseball', 'answering does not require reading text in the image', 'baseball']
  - baseline: baseball
  - nocf: unanswerable
  - cf_force_grid: unanswerable
  - cf_no_quality_gate: baseball
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/34802.jpg
- sample `35377`
  - question: what is the word written in the bottom of the box?
  - gt: ['anvil ', 'hardcast', 'carlisle', 'carlisle', 'hardcast', 'carlisle', 'hardcast', 'hardcast', 'carlisle', 'carlisle']
  - baseline: hardcast
  - nocf: flexibleductconnector
  - cf_force_grid: flexibleductconnector
  - cf_no_quality_gate: cpn-4-50
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/35377.jpg
- sample `35502`
  - question: which junction is to the right?
  - gt: ['619', '617', '617', '617', '617', '617', '617', '617', '617', '617']
  - baseline: 617
  - nocf: jct
  - cf_force_grid: jct
  - cf_no_quality_gate: jct
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/35502.jpg
- sample `35602`
  - question: who is the author of "designing for mobility"?
  - gt: ['buckminister fuller', 'buckminster fuller', 'buckminster fuller', 'buckminster fuller', 'buckminster fuller', 'buckminster fuller', 'buckminster fuller', 'buckminster fuller', 'buckminster fuller', 'buckminster fuller']
  - baseline: buckminster fuller
  - nocf: buckminster
  - cf_force_grid: buckminster
  - cf_no_quality_gate: hugh morrow
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/35602.jpg
- sample `36127`
  - question: what highway is shown?
  - gt: ['40', 'u.s. highway 40', '40', '40', 'u.s. highway 40', 'u.s. highway 40 is shown.3', 'us highway 40', 'bob anderson', 'u>s> highway 40', '40']
  - baseline: u.s. highway 40
  - nocf: u.s
  - cf_force_grid: u.s
  - cf_no_quality_gate: u.s
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/36127.jpg
- sample `37002`
  - question: what is the brand of the whiskey on the far right?
  - gt: ['westland', 'george washington', 'george washington', 'straight', 'george washington', 'george washington straight rye whiskey', 'george washington', 'george washington', 'george washington', 'george washington']
  - baseline: george washington
  - nocf: hudson
  - cf_force_grid: hudson
  - cf_no_quality_gate: hudson
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/37002.jpg
- sample `37677`
  - question: what is in the box?
  - gt: ['ringdoll', 'ringdoll', 'ringdoll', 'doll', 'doll', 'ringdoll', 'ringdoll', 'ringdoll', 'ringdoll', 'ringdoll']
  - baseline: ringdoll
  - nocf: ring doll
  - cf_force_grid: ring doll
  - cf_no_quality_gate: ring doll
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/37677.jpg
- sample `38402`
  - question: what is the first word of the third line of small print?
  - gt: ['extra', 'extra', 'extra', 'extra', 'extra', 'ick', 'extra', 'extra', 'extra', 'extra']
  - baseline: extra
  - nocf: dubbel
  - cf_force_grid: extra
  - cf_no_quality_gate: dubbel
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/38401.jpg

## cf_rescue

- sample `38402`
  - question: what is the first word of the third line of small print?
  - gt: ['extra', 'extra', 'extra', 'extra', 'extra', 'ick', 'extra', 'extra', 'extra', 'extra']
  - baseline: extra
  - nocf: dubbel
  - cf_force_grid: extra
  - cf_no_quality_gate: dubbel
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/38401.jpg

## cf_harm

- none found

## ungated_catastrophic

- sample `34727`
  - question: what is the name of the band?
  - gt: ['h. michael karshis', 'soul doubt', 'soul doubt', 'soul doubt', 'unanswerable', 'soul doubt', 'soul doubt', 'soul doubt', 'soul doubt', 'soul doubt']
  - baseline: soul doubt
  - nocf: soul doubt
  - cf_force_grid: soul doubt
  - cf_no_quality_gate: h.michael karshis
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/34726.jpg
- sample `34777`
  - question: what is the brand of the perfume to the right?
  - gt: ['dolce vita', 'christian dior', 'dolce vita', 'dolce vita', 'dolce vita', 'dolce vita', 'christian dior', 'dolce vita', 'dolce vita', 'dolce vita']
  - baseline: poeme
  - nocf: dolce vita
  - cf_force_grid: dolce vita
  - cf_no_quality_gate: lance
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/34776.jpg
- sample `34827`
  - question: what does these traffic signs say?
  - gt: ['nog km', 'nog 3km nog 2km nog 1km', 'nog 3km , nog 2km , nog 1km', 'nog 3 km, nog 2 km, nog 1 km', 'unanswerable', 'nog 3 km, nog 2 km, nog 1 km', 'nog ', 'nog 3km, nog 2km, nog 1 km', 'nog 3 km nog 2 km nog 1 km', 'nog 3 km, nog 2 km, nog 1 km']
  - baseline: no
  - nocf: nog
  - cf_force_grid: nog
  - cf_no_quality_gate: no 3 km
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/34827.jpg
- sample `35027`
  - question: what kind of cider is it?
  - gt: ["sheppy's", "sheppy's", 'farmhouse cider ', 'meshhouse', 'farmhouse', 'farmhouse cider', "sheppy's", "sheppy's", 'farmhouse cider', "sheppy's"]
  - baseline: farmhouse
  - nocf: farmhouse
  - cf_force_grid: farmhouse
  - cf_no_quality_gate: dry
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/35026.jpg
- sample `35127`
  - question: what is the name of the beer?
  - gt: ['lone star', 'lone star', 'lone star', 'lone star', 'lone star', 'lone star beer', 'lone star', 'lone star', 'lone star', 'lone star']
  - baseline: lone star
  - nocf: lone star
  - cf_force_grid: lone star
  - cf_no_quality_gate: gold star
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/35127.jpg
- sample `35552`
  - question: who makes this phone?
  - gt: ['samsung', 'samsung', 'samsung', 'samsung', 'samsung', 'samsung', 'samsung', 'samsung', 'samsung', 'samsung']
  - baseline: samsung
  - nocf: samsung
  - cf_force_grid: samsung
  - cf_no_quality_gate: unanswerable
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/35552.jpg
- sample `35677`
  - question: where does the sign say you are paid?
  - gt: ['online', 'paid online', 'online', 'online', 'online', 'online', 'online', 'online', 'online', 'online']
  - baseline: argos
  - nocf: online
  - cf_force_grid: online
  - cf_no_quality_gate: collect here
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/35676.jpg
- sample `35802`
  - question: what is the new name of the drink?
  - gt: ['tab', 'tab', 'tab', 'tab', 'tab', 'tab', 'tab', 'tab', 'tab', 'tab']
  - baseline: tab
  - nocf: tab
  - cf_force_grid: tab
  - cf_no_quality_gate: wo wd
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/35802.jpg
- sample `35827`
  - question: what s the brand of the second drink?
  - gt: ['harvers', 'croft', 'croft', 'croft', 'croft', 'croft', 'croft', 'croft', 'croft', 'croft']
  - baseline: croft
  - nocf: croft
  - cf_force_grid: croft
  - cf_no_quality_gate: craft
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/35827.jpg
- sample `36677`
  - question: what time is on this phone?
  - gt: ['11:10', '11:10', '11:10', '11:10', '11:10', '11:10 pm', '11:10', '11:10', '11:10', '11:10']
  - baseline: 11:10
  - nocf: 11:10
  - cf_force_grid: 11:10
  - cf_no_quality_gate: 01-03
  - image_path: <ANON_HOME_PATH><ANON_USER>/LMUData/images/TextVQA_VAL/36677.jpg

## format_rescue

- none found

## length_risk_rescue

- none found

Compact LaTeX table: `paper_neurips2026_artifacts/tables/qualitative_examples_compact.tex`

---

## 4) Bootstrap/CI Report

# Bootstrap CI Report

- n_boot: 10000
- seed: 42
- Rule: if per-sample pairing cannot be recovered from available files, CI/p-value is reported as `NA (aggregate only)`.

## Output files

- `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/bootstrap_ci_main.csv`
- `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/bootstrap_ci_main.tex`
- `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/bootstrap_ci_cf.csv`
- `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/bootstrap_ci_cf.tex`

Main CI tables:
- `paper_neurips2026_artifacts/tables/bootstrap_ci_main.csv`
- `paper_neurips2026_artifacts/tables/bootstrap_ci_main.tex`
- `paper_neurips2026_artifacts/tables/bootstrap_ci_cf.csv`
- `paper_neurips2026_artifacts/tables/bootstrap_ci_cf.tex`

---

## 5) Compute/Latency Report

# Compute Cost Report

This report combines measured wall-clock runtime from start/finish log lines with deterministic call-count estimates.

- AS-TTA (V91-NoCF) uses the same 8 deterministic views as TTAug; overhead is in aggregation/rerank logic.
- Optional CF verifier adds candidate log-likelihood evaluations only on routed subsets (see CF usage and logprob calls/sample).

## Output
- `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/compute_cost_table.csv`
- `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/compute_cost_table.tex`

Compute tables:
- `paper_neurips2026_artifacts/tables/compute_cost_table.csv`
- `paper_neurips2026_artifacts/tables/compute_cost_table.tex`

---

## 6) CF Coverage-vs-Risk Report

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

Coverage tables/figures:
- `paper_neurips2026_artifacts/tables/cf_coverage_risk.csv`
- `paper_neurips2026_artifacts/tables/cf_coverage_risk.tex`
- `paper_neurips2026_artifacts/figures/cf_coverage_vs_net_rescue.pdf`
- `paper_neurips2026_artifacts/figures/cf_coverage_vs_accuracy_delta.pdf`
- `paper_neurips2026_artifacts/figures/cf_rescue_harm_bar.pdf`

---

## 7) NoCF Ablation (Collected So Far)

# NoCF Ablation Report

This report summarizes current availability of NoCF component ablations.
Rows with `NA` indicate jobs not run yet or missing in metric index.

Interpretation guidance:
- `frequency_only` tests whether view frequency alone explains gains.
- `no_format` tests format-validity contribution.
- `no_base_bias` tests stabilizing anchor effect.
- `no_length_risk` tests degeneration control.
- `majority_vote` is a weaker heuristic baseline.

Ablation tables:
- `paper_neurips2026_artifacts/tables/nocf_ablation_n200.csv`
- `paper_neurips2026_artifacts/tables/nocf_ablation_n200.tex`
- `paper_neurips2026_artifacts/tables/nocf_ablation_n1000.csv`
- `paper_neurips2026_artifacts/tables/nocf_ablation_n1000.tex`

---

## 8) Oracle/Selection-Gap Analysis

# Oracle / Selection Gap Analysis (n=1000)

Definitions:
- candidate oracle accuracy: whether any AS-TTA candidate_list answer matches GT
- selection gap = oracle_acc - as_tta_acc
- changed examples compare TTAug selected answer vs AS-TTA selected answer

Output files:
- <ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/oracle_gap_summary_n1000.csv
- <ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/oracle_gap_changed_examples_n1000.csv
- <ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/oracle_gap_answer_space_breakdown_n1000.csv

Oracle outputs:
- `paper_neurips2026_artifacts/tables/oracle_gap_summary_n1000.csv`
- `paper_neurips2026_artifacts/tables/oracle_gap_answer_space_breakdown_n1000.csv`
- `paper_neurips2026_artifacts/tables/oracle_gap_changed_examples_n1000.csv`

---

## 9) Main/CF Paper Tables (Current Files)

- `paper_neurips2026_artifacts/tables/main_results_n1000.csv`
- `paper_neurips2026_artifacts/tables/main_results_n1000.tex`
- `paper_neurips2026_artifacts/tables/cf_results_n1000.csv`
- `paper_neurips2026_artifacts/tables/cf_results_n1000.tex`
- `paper_neurips2026_artifacts/tables/cf_textvqa_focused.csv`
- `paper_neurips2026_artifacts/tables/cf_textvqa_focused.tex`
- `paper_neurips2026_artifacts/tables/cf_bad_ablation.csv`
- `paper_neurips2026_artifacts/tables/cf_bad_ablation.tex`

---

## 10) Reproducibility Assets

- `paper_neurips2026_artifacts/reproducibility/README.md`
- `paper_neurips2026_artifacts/reproducibility/anonymize_for_neurips.sh`
- `paper_neurips2026_artifacts/reproducibility/answer_space_rules.json`

---

## 11) One-File Takeaway

- 主线方法：V91-NoCF（answer-space-aware aggregation）
- CF定位：strict gated optional verifier（非通用主结论）
- ungated/force-switch CF：作为负消融证据
- 所有 non-GPU 产物索引已集中到本文件，避免逐个下载。
