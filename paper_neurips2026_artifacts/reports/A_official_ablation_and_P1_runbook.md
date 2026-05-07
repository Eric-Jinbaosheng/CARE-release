# A-Official Ablation + P1 Analysis Runbook

## 已完成（无需GPU，已产出）

1. Candidate oracle / selection gap
- `paper_neurips2026_artifacts/tables/oracle_gap_summary_n1000.csv`
- `paper_neurips2026_artifacts/tables/oracle_gap_changed_examples_n1000.csv`
- `paper_neurips2026_artifacts/tables/oracle_gap_answer_space_breakdown_n1000.csv`
- `paper_neurips2026_artifacts/reports/oracle_gap_report.md`

2. Answer-space rules / contracts
- `paper_neurips2026_artifacts/reproducibility/answer_space_rules.json`
- `paper_neurips2026_artifacts/reports/answer_space_rules.md`
- `paper_neurips2026_artifacts/tables/answer_space_rules.tex`

3. Compute / cost
- `paper_neurips2026_artifacts/tables/compute_cost_table.csv`
- `paper_neurips2026_artifacts/tables/compute_cost_table.tex`
- `paper_neurips2026_artifacts/reports/compute_cost_report.md`

4. Case study / qualitative
- `paper_neurips2026_artifacts/reports/qualitative_examples.md`
- `paper_neurips2026_artifacts/tables/qualitative_examples_compact.tex`

5. Ablation diagnostics（changed-case / answer-space / subset）
- `paper_neurips2026_artifacts/ablations/affected_subset_n1000/ablation_changed_cases_all.csv`
- `paper_neurips2026_artifacts/ablations/affected_subset_n1000/ablation_affected_subset_summary.csv`
- `paper_neurips2026_artifacts/ablations/affected_subset_n1000/ablation_by_answer_space.csv`
- `paper_neurips2026_artifacts/ablations/affected_subset_n1000/ablation_diagnostic_report.md`

6. no_base_consistency（rerank-only diagnostic版）
- `paper_neurips2026_artifacts/ablations/no_base_consistency_n1000/ablation_no_base_consistency_n1000.csv`
- `paper_neurips2026_artifacts/ablations/no_base_consistency_n1000/ablation_n1000_complete.csv`
- `paper_neurips2026_artifacts/ablations/no_base_consistency_n1000/ablation_n1000_complete_latex.tex`
- `paper_neurips2026_artifacts/ablations/no_base_consistency_n1000/ablation_no_base_consistency_report.md`

## A 方案当前缺口（official ablation，需GPU补跑）
目标 benchmark：TextVQA / OCRVQA / ChartQA / OCRBench

缺 5 个官方评估结果：
1. `test_config_smolvlm2_v91_nocf_ablation_no_base_bias_textvqa`
2. `test_config_smolvlm2_v91_nocf_ablation_no_base_bias_ocrvqa`
3. `test_config_smolvlm2_v91_nocf_ablation_no_base_bias_chartqa`
4. `test_config_smolvlm2_v91_nocf_ablation_no_base_bias_ocrbench`
5. `test_config_smolvlm2_v91_nocf_ablation_majority_vote_chartqa`

说明：这5个需要 official `acc/score` 文件，才能把 ablation 表作为主文正式表。

## GPU补跑命令（不设分区，直接gpu:1）
在 repo 根目录执行：

```bash
cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

ACC="<ANON_ACCOUNT>"
for stem in \
  test_config_smolvlm2_v91_nocf_ablation_no_base_bias_textvqa \
  test_config_smolvlm2_v91_nocf_ablation_no_base_bias_ocrvqa \
  test_config_smolvlm2_v91_nocf_ablation_no_base_bias_chartqa \
  test_config_smolvlm2_v91_nocf_ablation_no_base_bias_ocrbench \
  test_config_smolvlm2_v91_nocf_ablation_majority_vote_chartqa
  do
    CFG="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/paper_neurips2026_artifacts/configs/ablation_configs/${stem}.json"
    CACHE="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/${stem}_n1000"
    mkdir -p "$CACHE"
    sbatch --account="$ACC" \
      --job-name="abl1k-${stem#test_config_smolvlm2_v91_nocf_ablation_}" \
      --export=ALL,CONFIG_PATH="$CFG",SUBSET_LEN=1000,CACHE_PATH="$CACHE" \
      scripts/sbatch_clean.sh "$CFG"
  done
```

如果你想分账号并发提交，可把 `ACC` 换成你空闲账号分别提交。

## 补跑完成后，一键更新表格（无需GPU）
```bash
cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

python paper_neurips2026_artifacts/scripts/rebuild_metric_index.py \
  --repo-root <ANON_ROOT>/peking/smolvlm2_paper/ets_clean

python paper_neurips2026_artifacts/scripts/collect_ablation_results.py
python paper_neurips2026_artifacts/scripts/run_ablation_no_base_consistency_n1000.py --n 1000
python paper_neurips2026_artifacts/scripts/ablation_affected_subset_analysis.py --n 1000
python paper_neurips2026_artifacts/scripts/make_paper_tables.py --textvqa-cf-override 72.28
```

## 一致性检查（必须过）
- official ablation 表里的 `Full ASCA` 必须等于 Table 1 的 `ASCA`。
- 若不相等，不放主文，只放 appendix + 注明口径。
