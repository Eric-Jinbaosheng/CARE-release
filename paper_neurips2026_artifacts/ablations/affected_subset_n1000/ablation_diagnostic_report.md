# Ablation Affected-Subset Analysis (n=1000)

## 保留的 overall ablation（原表）

```csv
benchmark,full,frequency_only,majority_vote,no_format,no_base_consistency,no_length_risk
textvqa,0.77,0.771,0.77,0.771,0.771,0.77
ocrvqa,0.607,0.596,0.607,0.607,0.605,0.607
gqa,0.407,,,,,
chartqa,0.713,0.709,,0.713,0.711,0.713
ocrbench,0.706,0.706,0.706,0.705,0.706,0.706
```

## Changed-case 总结

```csv
benchmark,ablation,n,source_path,changed,full_wins,ablation_wins,full_net_gain,changed_subset_full_acc,changed_subset_ablation_acc,changed_subset_delta
textvqa,frequency_only,1000,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_ablation_frequency_only_textvqa/V91NoCFAbl_frequency_only_SmolVLM2_2B/V91NoCFAbl_frequency_only_SmolVLM2_2B_TextVQA_VAL.xlsx,44,12,13,-1,0.431818,0.454545,0.022727
textvqa,no_format,1000,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_ablation_no_format_textvqa/V91NoCFAbl_no_format_SmolVLM2_2B/V91NoCFAbl_no_format_SmolVLM2_2B_TextVQA_VAL.xlsx,2,0,1,-1,0.000000,0.500000,0.500000
textvqa,no_base_consistency,1000,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/ablations/no_base_consistency_n1000/textvqa/predictions.csv,29,6,7,-1,0.344828,0.379310,0.034483
textvqa,no_length_risk,1000,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_ablation_no_length_risk_textvqa/V91NoCFAbl_no_length_risk_SmolVLM2_2B/V91NoCFAbl_no_length_risk_SmolVLM2_2B_TextVQA_VAL.xlsx,0,0,0,0,NA,NA,NA
textvqa,majority_vote,1000,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_ablation_majority_vote_textvqa/V91NoCFAbl_majority_vote_SmolVLM2_2B/V91NoCFAbl_majority_vote_SmolVLM2_2B_TextVQA_VAL.xlsx,0,0,0,0,NA,NA,NA
ocrvqa,frequency_only,1000,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_ablation_frequency_only_ocrvqa/V91NoCFAbl_frequency_only_SmolVLM2_2B/V91NoCFAbl_frequency_only_SmolVLM2_2B_OCRVQA_TEST.xlsx,38,14,3,11,0.368421,0.078947,-0.289474
ocrvqa,no_format,1000,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_ablation_no_format_ocrvqa/V91NoCFAbl_no_format_SmolVLM2_2B/V91NoCFAbl_no_format_SmolVLM2_2B_OCRVQA_TEST.xlsx,1,0,0,0,0.000000,0.000000,0.000000
ocrvqa,no_base_consistency,1000,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/ablations/no_base_consistency_n1000/ocrvqa/predictions.csv,16,2,0,2,0.125000,0.000000,-0.125000
ocrvqa,no_length_risk,1000,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_ablation_no_length_risk_ocrvqa/V91NoCFAbl_no_length_risk_SmolVLM2_2B/V91NoCFAbl_no_length_risk_SmolVLM2_2B_OCRVQA_TEST.xlsx,0,0,0,0,NA,NA,NA
ocrvqa,majority_vote,1000,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_ablation_majority_vote_ocrvqa/V91NoCFAbl_majority_vote_SmolVLM2_2B/V91NoCFAbl_majority_vote_SmolVLM2_2B_OCRVQA_TEST.xlsx,0,0,0,0,NA,NA,NA
chartqa,frequency_only,1000,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_ablation_frequency_only_chartqa/V91NoCFAbl_frequency_only_SmolVLM2_2B/V91NoCFAbl_frequency_only_SmolVLM2_2B_ChartQA_TEST.xlsx,37,10,6,4,0.270270,0.162162,-0.108108
chartqa,no_format,1000,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_ablation_no_format_chartqa/V91NoCFAbl_no_format_SmolVLM2_2B/V91NoCFAbl_no_format_SmolVLM2_2B_ChartQA_TEST.xlsx,0,0,0,0,NA,NA,NA
chartqa,no_base_consistency,1000,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/ablations/no_base_consistency_n1000/chartqa/predictions.csv,38,8,6,2,0.210526,0.157895,-0.052632
chartqa,no_length_risk,1000,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_ablation_no_length_risk_chartqa/V91NoCFAbl_no_length_risk_SmolVLM2_2B/V91NoCFAbl_no_length_risk_SmolVLM2_2B_ChartQA_TEST.xlsx,0,0,0,0,NA,NA,NA
ocrbench,frequency_only,1000,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_ablation_frequency_only_ocrbench/V91NoCFAbl_frequency_only_SmolVLM2_2B/V91NoCFAbl_frequency_only_SmolVLM2_2B_OCRBench.xlsx,43,6,6,0,0.162791,0.162791,0.000000
ocrbench,no_format,1000,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_ablation_no_format_ocrbench/V91NoCFAbl_no_format_SmolVLM2_2B/V91NoCFAbl_no_format_SmolVLM2_2B_OCRBench.xlsx,1,1,0,1,1.000000,0.000000,-1.000000
ocrbench,no_base_consistency,1000,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/ablations/no_base_consistency_n1000/ocrbench/predictions.csv,35,5,5,0,0.171429,0.171429,0.000000
ocrbench,no_length_risk,1000,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_ablation_no_length_risk_ocrbench/V91NoCFAbl_no_length_risk_SmolVLM2_2B/V91NoCFAbl_no_length_risk_SmolVLM2_2B_OCRBench.xlsx,0,0,0,0,NA,NA,NA
ocrbench,majority_vote,1000,<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_ablation_majority_vote_ocrbench/V91NoCFAbl_majority_vote_SmolVLM2_2B/V91NoCFAbl_majority_vote_SmolVLM2_2B_OCRBench.xlsx,0,0,0,0,NA,NA,NA
```

## 说明
- changed/full_wins/ablation_wins/full_net_gain 均按 sample-level correctness 计算。
- low-margin 子集定义：top1 support - top2 support <= 1/8 或 <= 2/8。
- format-affected 子集：candidate pool 里 contract validity 至少两档。
- length-risk-affected 子集：candidate pool 同时存在 risk=0 与 risk>0 候选。

## 缺失/跳过
- gqa: missing_diag
- gqa: missing_frequency_only
- gqa: missing_no_format
- gqa: missing_no_base_consistency
- gqa: missing_no_length_risk
- gqa: missing_majority_vote
- chartqa: missing_majority_vote
