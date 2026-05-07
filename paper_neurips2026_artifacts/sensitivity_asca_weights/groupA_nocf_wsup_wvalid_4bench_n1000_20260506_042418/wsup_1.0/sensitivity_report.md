# ASCA Sensitivity Report

## 默认参数
- w_sup=2.0, w_valid=1.0, w_base=0.4, w_risk=0.5

## 运行范围
- benchmarks: chartqa, textvqa, ocrvqa, ocrbench
- n: 1000
- settings: 1 (sweep=single)
- 说明：仅复用已有 diagnostics / predictions 做 rerank+eval，没有重新生成 VLM 输出。

## 候选新默认参数建议
- 当前无 setting 满足 candidate_new_default 全部条件，建议保持默认参数。

## 文件输出
- <ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/sensitivity_asca_weights/groupA_nocf_wsup_wvalid_4bench_n1000_20260506_042418/wsup_1.0/sensitivity_raw.csv
- <ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/sensitivity_asca_weights/groupA_nocf_wsup_wvalid_4bench_n1000_20260506_042418/wsup_1.0/sensitivity_summary_by_benchmark.csv
- <ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/sensitivity_asca_weights/groupA_nocf_wsup_wvalid_4bench_n1000_20260506_042418/wsup_1.0/sensitivity_summary_by_setting.csv
- <ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/sensitivity_asca_weights/groupA_nocf_wsup_wvalid_4bench_n1000_20260506_042418/wsup_1.0/sensitivity_changed_cases.csv
- <ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/sensitivity_asca_weights/groupA_nocf_wsup_wvalid_4bench_n1000_20260506_042418/wsup_1.0/sensitivity_report.md

## 口径提醒
- score/delta_vs_default 使用官方 evaluator（与主表同口径）。
- default_method=nocf（delta 参考分数来源：nocf）。
- changed/default_wins/setting_wins/net_vs_default 始终按 sample-level 口径统计。
