# ASCA Sensitivity Report

## 默认参数
- w_sup=2.0, w_valid=1.0, w_base=0.4, w_risk=0.5

## 运行范围
- benchmarks: textvqa, ocrvqa, chartqa, ocrbench, gqa, ai2d, mme_rw, coco, amber
- n: 1000
- settings: 1 (sweep=single)
- 说明：仅复用已有 diagnostics / predictions 做 rerank+eval，没有重新生成 VLM 输出。

## 缺失输入
- gqa: missing_candidate_diag
- ai2d: missing_candidate_diag
- mme_rw: missing_candidate_diag
- coco: missing_candidate_diag
- amber: missing_candidate_diag

## 候选新默认参数建议
- wsup2p0_wvalid1p0_wbase0p4_wrisk0p75 (avg_delta=0.000500, improved=2, hurt=0)

## 文件输出
- <ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/sensitivity_asca_weights/full4param_9bench_full4p_20260503_033213/wrisk_0p75/sensitivity_raw.csv
- <ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/sensitivity_asca_weights/full4param_9bench_full4p_20260503_033213/wrisk_0p75/sensitivity_summary_by_benchmark.csv
- <ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/sensitivity_asca_weights/full4param_9bench_full4p_20260503_033213/wrisk_0p75/sensitivity_summary_by_setting.csv
- <ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/sensitivity_asca_weights/full4param_9bench_full4p_20260503_033213/wrisk_0p75/sensitivity_changed_cases.csv
- <ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/sensitivity_asca_weights/full4param_9bench_full4p_20260503_033213/wrisk_0p75/sensitivity_report.md

## 口径提醒
- metric 为基于预测表的 sample-level accuracy proxy（统一口径用于参数敏感性相对比较）。
- changed/default_wins/setting_wins/net_vs_default 均按该 sample-level 口径统计。
