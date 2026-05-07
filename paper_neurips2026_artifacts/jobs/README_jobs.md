# Jobs README

## Standard
All scripts assume:
- `cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean`
- `scripts/sbatch_clean.sh`
- `CONFIG_PATH`, `SUBSET_LEN`, `CACHE_PATH` exported per job
- logs written under `logs/`

## Priority order
1. TextVQA routed CF n=1000 (if still missing)
2. NoCF ablations n=200
3. Bootstrap CI from existing outputs (CPU)
4. Compute/latency report (CPU)
5. CF coverage-risk analysis (CPU)
6. Qualitative extraction (CPU)
7. NoCF ablations n=1000 priority
8. Second backbone n=200

## Submit commands
```bash
cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean
bash paper_neurips2026_artifacts/jobs/submit_textvqa_routed_n1000.sh
bash paper_neurips2026_artifacts/jobs/submit_ablation_n200.sh
bash paper_neurips2026_artifacts/jobs/submit_ablation_n1000_priority.sh
bash paper_neurips2026_artifacts/jobs/submit_second_backbone_n200.sh
```

## Monitor
```bash
squeue -u <ANON_USER>
```

## Collect after completion
```bash
python paper_neurips2026_artifacts/scripts/collect_ablation_results.py
python paper_neurips2026_artifacts/scripts/make_paper_tables.py --textvqa-cf-override 72.28
python paper_neurips2026_artifacts/scripts/bootstrap_ci.py --n-boot 10000
python paper_neurips2026_artifacts/scripts/compute_cost_report.py
python paper_neurips2026_artifacts/scripts/cf_coverage_risk_analysis.py --textvqa-cf-override 72.28
```
