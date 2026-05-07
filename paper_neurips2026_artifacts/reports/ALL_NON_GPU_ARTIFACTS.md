# Non-GPU Artifacts Bundle

- Generated at: 2026-04-30T02:28:19.299139
- Repo root: `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean`

## File Index

- [oracle_gap_summary_n1000.csv](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/oracle_gap_summary_n1000.csv) - OK
- [oracle_gap_answer_space_breakdown_n1000.csv](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/oracle_gap_answer_space_breakdown_n1000.csv) - OK
- [oracle_gap_changed_examples_n1000.csv](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/oracle_gap_changed_examples_n1000.csv) - OK
- [cf_coverage_risk.csv](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/cf_coverage_risk.csv) - OK
- [cf_bad_ablation.csv](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/cf_bad_ablation.csv) - OK
- [bootstrap_ci_main.csv](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/bootstrap_ci_main.csv) - OK
- [bootstrap_ci_cf.csv](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/bootstrap_ci_cf.csv) - OK
- [compute_cost_table.csv](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/compute_cost_table.csv) - OK
- [answer_space_rules.tex](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/answer_space_rules.tex) - OK
- [qualitative_examples_compact.tex](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/qualitative_examples_compact.tex) - OK
- [main_results_n1000.tex](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/main_results_n1000.tex) - OK
- [cf_results_n1000.tex](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/cf_results_n1000.tex) - OK
- [cf_textvqa_focused.tex](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/cf_textvqa_focused.tex) - OK
- [nocf_ablation_n200.csv](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/nocf_ablation_n200.csv) - OK
- [nocf_ablation_n1000.csv](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/tables/nocf_ablation_n1000.csv) - OK
- [oracle_gap_report.md](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/reports/oracle_gap_report.md) - OK
- [bootstrap_ci_report.md](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/reports/bootstrap_ci_report.md) - OK
- [compute_cost_report.md](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/reports/compute_cost_report.md) - OK
- [cf_coverage_risk_report.md](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/reports/cf_coverage_risk_report.md) - OK
- [answer_space_rules.md](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/reports/answer_space_rules.md) - OK
- [qualitative_examples.md](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/reports/qualitative_examples.md) - OK
- [final_neurips_gap_report.md](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/reports/final_neurips_gap_report.md) - OK
- [answer_space_rules.json](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/reproducibility/answer_space_rules.json) - OK
- [experiment_metric_index_20260429.csv](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/logs/experiment_metric_index_20260429.csv) - OK
- [experiment_metric_index_20260429.json](<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/logs/experiment_metric_index_20260429.json) - OK

## Key Snapshot

### oracle_gap_summary_n1000.csv
```
benchmark,dataset,n,candidate_oracle_acc,ttaug_acc,as_tta_acc,selection_gap,changed_tta_right_as_wrong,changed_tta_wrong_as_right,note
ocrbench,OCRBench,1000,0.805000,0.711000,0.706000,0.099000,27,22,ok
gqa,GQA_TestDev_Balanced,1000,0.656000,0.395000,0.407000,0.249000,61,73,ok
textvqa,TextVQA_VAL,1000,0.885000,0.767000,0.770000,0.115000,34,37,ok
chartqa,ChartQA_TEST,1000,0.825000,0.696000,0.713000,0.112000,24,41,ok
ocrvqa,OCRVQA_TEST,1000,0.737000,0.606000,0.607000,0.130000,36,37,ok
ai2d,AI2D_TEST,1000,1.000000,0.684000,0.688000,0.312000,26,30,ok
mme_rw,MME-RealWorld-Lite,1000,1.000000,0.309000,0.315000,0.685000,54,60,ok
coco,COCO_VAL,1000,0.001000,0.000000,0.000000,0.001000,0,0,ok
amber,AMBER,1000,0.882000,0.767000,0.778000,0.104000,47,58,ok
```

### cf_bad_ablation.csv
```
Variant,Dataset,n,Score,Delta_vs_NoCF,Finding
cf3_no_quality_gate,TextVQA,200,61.25,-7.80,ungated switching harms
cf3_force_switch_analysis,TextVQA,200,60.75,-8.30,CF winner selector collapses
```

### cf_coverage_risk.csv
```
config,dataset,n,metric,score,nocf_score,delta_vs_nocf,cf_used_rate,prediction_changed_rate,rescue,harm,neutral_change,net_rescue,main_block_reason,diag_path
test_config_smolvlm2_v91_cf3_routed_ai2d,AI2D_TEST,1000,Overall,0.688,0.688,0.0,0.0,0.0,0,0,0,0,no_cf_scores,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf3_routed_ai2d/diagnostics/v91cf3_samples.jsonl
test_config_smolvlm2_v91_cf_ai2d,AI2D_TEST,200,Overall,0.67,0.67,0.0,0.0,0.0,0,0,0,0,unknown,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf_ai2d/diagnostics/v91cf_samples.jsonl
test_config_smolvlm2_v91_cf3_routed_amber,AMBER,1000,Avg ACC,74.704628452513,74.704628452513,0.0,0.0,0.0,0,0,0,0,no_cf_scores,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf3_routed_amber/diagnostics/v91cf3_samples.jsonl
test_config_smolvlm2_v91_cf3_routed_coco,COCO_VAL,1000,ROUGE_L,18.768026820554358,18.768026820554358,0.0,0.0,0.0,0,0,0,0,no_cf_scores,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf3_routed_coco/diagnostics/v91cf3_samples.jsonl
test_config_smolvlm2_v91_cf_coco,COCO_VAL,200,ROUGE_L,18.642228516865508,18.642228516865508,0.0,0.0,0.125,0,0,0,0,unknown,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf_coco/diagnostics/v91cf_samples.jsonl
test_config_smolvlm2_v91_cf3_routed_chartqa,ChartQA_TEST,1000,Overall,76.3,76.3,0.0,0.002,0.005,0,0,2,0,no_cf_scores,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf3_routed_chartqa/diagnostics/v91cf3_samples.jsonl
test_config_smolvlm2_v91_cf_chartqa,ChartQA_TEST,200,Overall,83.5,83.5,0.0,0.08,0.01,0,0,0,0,unknown,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf_chartqa/diagnostics/v91cf_samples.jsonl
test_config_smolvlm2_v91_cf3_routed_gqa,GQA_TestDev_Balanced,1000,Overall,10.2,9.9,0.29999999999999893,0.037,0.037,10,3,24,7,no_cf_scores,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf3_routed_gqa/diagnostics/v91cf3_samples.jsonl
test_config_smolvlm2_v91_cf_gqa,GQA_TestDev_Balanced,200,Overall,10.0,10.0,0.0,0.2,0.085,0,0,1,0,unknown,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf_gqa/diagnostics/v91cf_samples.jsonl
test_config_smolvlm2_v91_cf3_routed_mme_rw,MME-RealWorld-Lite,1000,Overall,0.315,,,0.0,0.0,,,,,no_cf_scores,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf3_routed_mme_rw/diagnostics/v91cf3_samples.jsonl
test_config_smolvlm2_v91_cf_mme_rw,MME-RealWorld-Lite,200,Overall,0.385,,,0.0,0.035,,,,,unknown,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf_mme_rw/diagnostics/v91cf_samples.jsonl
test_config_smolvlm2_v91_cf3_routed_ocrbench,OCRBench,1000,Final Score Norm,72.5,72.5,0.0,0.010447761194029851,0.014925373134328358,3,3,6,0,no_cf_scores,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf3_routed_ocrbench/diagnostics/v91cf3_samples.jsonl
test_config_smolvlm2_v91_cf_ocrbench,OCRBench,200,Final Score Norm,14.7,14.7,0.0,0.155,0.045,0,0,0,0,unknown,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf_ocrbench/diagnostics/v91cf_samples.jsonl
test_config_smolvlm2_v91_cf3_routed_ocrvqa,OCRVQA_TEST,1000,Overall,16.1,16.2,-0.09999999999999787,0.007,0.007,0,1,6,-1,no_cf_scores,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf3_routed_ocrvqa/diagnostics/v91cf3_samples.jsonl
test_config_smolvlm2_v91_cf_ocrvqa,OCRVQA_TEST,200,Overall,17.5,17.5,0.0,0.155,0.03,0,1,0,-1,unknown,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf_ocrvqa/diagnostics/v91cf_samples.jsonl
test_config_smolvlm2_v91_cf2_textvqa,TextVQA_VAL,200,Overall,69.05000000000001,69.05000000000001,0.0,0.03,0.0,0,0,0,0,beta_zero,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf2_textvqa/diagnostics/v91cf_samples.jsonl
test_config_smolvlm2_v91_cf3_force_grid_no_quality_gate_textvqa,TextVQA_VAL,200,Overall,61.75000000000001,69.05000000000001,-7.300000000000004,0.315,0.275,7,22,26,-15,low_cf_margin,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf3_force_grid_no_quality_gate_textvqa/diagnostics/v91cf3_samples.jsonl
test_config_smolvlm2_v91_cf3_force_grid_textvqa,TextVQA_VAL,1000,Overall,72.28,71.96000000000001,0.3199999999999932,0.023,0.025,9,6,6,3,low_cf_margin,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf3_force_grid_textvqa/diagnostics/v91cf3_samples.jsonl
test_config_smolvlm2_v91_cf3_force_switch_analysis_textvqa,TextVQA_VAL,200,Overall,60.75000000000001,69.05000000000001,-8.300000000000004,0.265,0.265,6,23,24,-17,no_cf_scores,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf3_force_switch_analysis_textvqa/diagnostics/v91cf3_samples.jsonl
test_config_smolvlm2_v91_cf3_no_quality_gate_textvqa,TextVQA_VAL,200,Overall,61.25000000000001,69.05000000000001,-7.800000000000004,0.26,0.26,6,22,24,-16,no_cf_scores,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf3_no_quality_gate_textvqa/diagnostics/v91cf3_samples.jsonl
test_config_smolvlm2_v91_cf3_routed_textvqa,TextVQA_VAL,200,Overall,69.05000000000001,69.05000000000001,0.0,0.005,0.005,0,0,1,0,no_cf_scores,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf3_routed_textvqa/diagnostics/v91cf3_samples.jsonl
test_config_smolvlm2_v91_cf3_routed_textvqa_g01,TextVQA_VAL,200,Overall,69.05000000000001,,,0.005,0.005,,,,,no_cf_scores,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf3_routed_textvqa_g01/diagnostics/v91cf3_samples.jsonl
test_config_smolvlm2_v91_cf3_routed_textvqa_g02,TextVQA_VAL,200,Overall,69.05000000000001,,,0.005,0.005,,,,,no_cf_scores,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf3_routed_textvqa_g02/diagnostics/v91cf3_samples.jsonl
test_config_smolvlm2_v91_cf3_routed_textvqa_g04,TextVQA_VAL,200,Overall,69.05000000000001,,,0.005,0.005,,,,,no_cf_scores,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf3_routed_textvqa_g04/diagnostics/v91cf3_samples.jsonl
test_config_smolvlm2_v91_cf3_score_only_textvqa,TextVQA_VAL,200,Overall,69.05000000000001,69.05000000000001,0.0,0.0,0.0,0,0,0,0,cf3_score_only,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf3_score_only_textvqa/diagnostics/v91cf3_samples.jsonl
test_config_smolvlm2_v91_cf3_textvqa,TextVQA_VAL,200,Overall,69.05000000000001,69.05000000000001,0.0,0.0,0.0,0,0,0,0,cf3_verifier_block,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf3_textvqa/diagnostics/v91cf3_samples.jsonl
test_config_smolvlm2_v91_cf_textvqa,TextVQA_VAL,200,Overall,69.05000000000001,69.05000000000001,0.0,0.374,0.036,0,0,3,0,unknown,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/test_config_smolvlm2_v91_cf_textvqa/diagnostics/v91cf_samples.jsonl
```

### bootstrap_ci_main.csv
```
Benchmark,Metric,Baseline,Method,Delta,95% CI,p-value,Pairing status,Notes
ocrbench,Final Score Norm,73.70,72.50,-1.200,NA (aggregate only),NA,aggregate_only/aggregate_only,base:<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_paper_ttaug_classical_ocrbench/TTAugClassical_SmolVLM2_2B/TTAugClassical_SmolVLM2_2B_OCRBench.xlsx; nocf:<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_ocrbench/V91NoCF_SmolVLM2_2B/V91NoCF_SmolVLM2_2B_OCRBench.xlsx
gqa,Overall,4.300,9.900,5.600,NA (aggregate only),NA,aggregate_only/aggregate_only,base:<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_paper_ttaug_classical_gqa/TTAugClassical_SmolVLM2_2B/TTAugClassical_SmolVLM2_2B_GQA_TestDev_Balanced.xlsx; nocf:<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_gqa/V91NoCF_SmolVLM2_2B/V91NoCF_SmolVLM2_2B_GQA_TestDev_Balanced.xlsx
textvqa,Overall,72.28,71.96,-0.3200,NA (aggregate only),NA,aggregate_only/aggregate_only,base:<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_paper_ttaug_classical_textvqa/TTAugClassical_SmolVLM2_2B/TTAugClassical_SmolVLM2_2B_TextVQA_VAL.xlsx; nocf:<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_textvqa/V91NoCF_SmolVLM2_2B/V91NoCF_SmolVLM2_2B_TextVQA_VAL.xlsx
chartqa,Overall,75.30,76.30,1.000,NA (aggregate only),NA,aggregate_only/aggregate_only,base:<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_paper_ttaug_classical_chartqa/TTAugClassical_SmolVLM2_2B/TTAugClassical_SmolVLM2_2B_ChartQA_TEST.xlsx; nocf:<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_chartqa/V91NoCF_SmolVLM2_2B/V91NoCF_SmolVLM2_2B_ChartQA_TEST.xlsx
ocrvqa,Overall,12.10,16.20,4.100,NA (aggregate only),NA,aggregate_only/aggregate_only,base:<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_paper_ttaug_classical_ocrvqa/TTAugClassical_SmolVLM2_2B/TTAugClassical_SmolVLM2_2B_OCRVQA_TEST.xlsx; nocf:<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_ocrvqa/V91NoCF_SmolVLM2_2B/V91NoCF_SmolVLM2_2B_OCRVQA_TEST.xlsx
ai2d,Overall,0.6840,0.6880,0.0040,"[-0.0100, 0.0190]",0.6294,paired:1000,base:TTAugClassical_SmolVLM2_2B_AI2D_TEST_openai_result.xlsx; nocf:V91NoCF_SmolVLM2_2B_AI2D_TEST_openai_result.xlsx
mme_rw,Overall,0.3090,0.3150,0.0060,NA (aggregate only),NA,aggregate_only/aggregate_only,base:<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_paper_ttaug_classical_mme_rw/TTAugClassical_SmolVLM2_2B/TTAugClassical_SmolVLM2_2B_MME-RealWorld-Lite.xlsx; nocf:<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_mme_rw/V91NoCF_SmolVLM2_2B/V91NoCF_SmolVLM2_2B_MME-RealWorld-Lite.xlsx
coco,ROUGE_L,15.78,18.77,2.990,NA (aggregate only),NA,aggregate_only/aggregate_only,base:<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_paper_ttaug_classical_coco/TTAugClassical_SmolVLM2_2B/TTAugClassical_SmolVLM2_2B_COCO_VAL.xlsx; nocf:<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_coco/V91NoCF_SmolVLM2_2B/V91NoCF_SmolVLM2_2B_COCO_VAL.xlsx
amber,Avg ACC,74.29,74.70,0.4161,"[-0.0090, 0.0310]",0.307,paired:1000,base:TTAugClassical_SmolVLM2_2B_AMBER_auxmatch.xlsx; nocf:V91NoCF_SmolVLM2_2B_AMBER_auxmatch.xlsx
```

### compute_cost_table.csv
```
Method,Generations/sample,Logprob calls/sample,CF usage,Prediction change rate,Runtime_sec,Runtime_status,Relative cost vs base,Log source,Notes
Base_SmolVLM2,1.00,NA,NA,NA,NA,missing_log,NA,NA,single-view base inference
TTAug_deterministic,8.00,NA,NA,NA,NA,missing_log,NA,NA,8 deterministic views
V91_NoCF,8.00,0.00,0.0000,0.0040,NA,missing_log,NA,NA,"same 8-view generation, rerank-only"
V91_CF3_Routed,8.00,4.59,0.0050,0.0050,NA,missing_log,NA,NA,optional sparse CF verifier
V91_CF3_ForceGrid,8.00,7.25,0.0230,0.0250,NA,missing_log,NA,NA,focused CF stress-test
CF3_NoQualityGate,8.00,4.84,0.2600,0.2600,NA,missing_log,NA,NA,negative ablation
CF3_ForceSwitch,8.00,4.84,0.2650,0.2650,NA,missing_log,NA,NA,negative ablation
```

### main_results_n1000.tex
```
\begin{table}[t]
\centering
\begin{tabular}{lcccccc}
\toprule
Benchmark & Metric & Base & TTAug & AS-TTA & $\Delta$ vs TTAug & n \\
\midrule
ocrbench & Final Score Norm & 72.90 & 73.70 & 72.50 & -1.200 & 1000 \\
gqa & Overall & 0.0000 & 4.300 & 9.900 & 5.600 & 1000 \\
textvqa & Overall & 73.16 & 72.28 & 71.96 & -0.3200 & 1000 \\
chartqa & Overall & 74.20 & 75.30 & 76.30 & 1.000 & 1000 \\
ocrvqa & Overall & 0.0000 & 12.10 & 16.20 & 4.100 & 1000 \\
ai2d & Overall & 0.6850 & 0.6840 & 0.6880 & 0.0040 & 1000 \\
mme_rw & Overall & 0.2780 & 0.3090 & 0.3150 & 0.0060 & 1000 \\
coco & ROUGE_L & 9.062 & 15.78 & 18.77 & 2.990 & 1000 \\
amber & Avg ACC & 68.69 & 74.29 & 74.70 & 0.4161 & 1000 \\
\bottomrule
\end{tabular}
\caption{Main benchmark results on n=1000 subsets.}
\label{tab:main_n1000}
\end{table}
```

### cf_results_n1000.tex
```
\begin{table}[t]
\centering
\begin{tabular}{lcccccc}
\toprule
Benchmark & Metric & NoCF & Routed CF & $\Delta$ & CF usage & n \\
\midrule
ocrbench & Final Score Norm & 72.50 & 72.50 & 0.0000 & 0.0104 & 1000 \\
gqa & Overall & 9.900 & 10.20 & 0.3000 & 0.0370 & 1000 \\
textvqa & Overall & 71.96 & 72.28 & 0.3200 & 0.0050 & 1000 \\
chartqa & Overall & 76.30 & 76.30 & 0.0000 & 0.0020 & 1000 \\
ocrvqa & Overall & 16.20 & 16.10 & -0.1000 & 0.0070 & 1000 \\
ai2d & Overall & 0.6880 & 0.6880 & 0.0000 & 0.0000 & 1000 \\
mme_rw & Overall & 0.3150 & 0.3150 & 0.0000 & 0.0000 & 1000 \\
coco & ROUGE_L & 18.77 & 18.77 & 0.0000 & 0.0000 & 1000 \\
amber & Avg ACC & 74.70 & 74.70 & 0.0000 & 0.0000 & 1000 \\
\bottomrule
\end{tabular}
\caption{Optional routed CF verifier vs V91-NoCF on n=1000 subsets.}
\label{tab:cf_routed_n1000}
\end{table}
```

## Full Content Pointers

Large files are linked above and kept on disk as canonical sources.