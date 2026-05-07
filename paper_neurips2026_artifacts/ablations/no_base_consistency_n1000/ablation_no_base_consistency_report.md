# no_base_consistency Ablation Report (n=1000)

## 定义
- method = no_base_consistency
- score = 2.0*support + 1.0*valid + 0.0*base - 0.5*risk

## 跑的 benchmark
- textvqa: score=0.771000, full=0.770000, delta=0.001000, changed=29, full_wins=6, ablation_wins=7, net=1
- ocrvqa: score=0.605000, full=0.607000, delta=-0.002000, changed=16, full_wins=2, ablation_wins=0, net=-2
- chartqa: score=0.711000, full=0.713000, delta=-0.002000, changed=38, full_wins=8, ablation_wins=6, net=-2
- ocrbench: score=0.706000, full=0.706000, delta=0.000000, changed=35, full_wins=5, ablation_wins=5, net=0

## 跳过/缺失
- gqa: missing_candidate_diag

## 解释
- 如果 changed_vs_full 很小：base consistency 目前更多是 sparse anchor。
- 如果 delta_vs_full 很小：说明 ASCA 并非简单复制 base answer。
- 如果某 benchmark no_base_consistency 更好：可视为 potential over-anchoring case。

## 产物路径
- <ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/ablations/no_base_consistency_n1000/ablation_no_base_consistency_n1000.csv
- <ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/ablations/no_base_consistency_n1000/ablation_n1000_complete.csv
- <ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/ablations/no_base_consistency_n1000/ablation_n1000_complete_latex.tex
- <ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/ablations/no_base_consistency_n1000/ablation_changed_cases_n1000.csv
- <ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/ablations/no_base_consistency_n1000

## 注意
- 本实验只做 rerank+eval，未重新调用 VLM generation。
