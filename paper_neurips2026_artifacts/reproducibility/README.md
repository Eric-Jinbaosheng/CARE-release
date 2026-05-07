# Reproducibility (Anonymous-Friendly)

## Environment
- Python 3.x
- Run from repo root:
  - `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean`

## Main table regeneration (from existing results)
```bash
python paper_neurips2026_artifacts/scripts/make_paper_tables.py --textvqa-cf-override 72.28
python paper_neurips2026_artifacts/scripts/bootstrap_ci.py --n-boot 10000
python paper_neurips2026_artifacts/scripts/compute_cost_report.py
python paper_neurips2026_artifacts/scripts/cf_coverage_risk_analysis.py --textvqa-cf-override 72.28
python paper_neurips2026_artifacts/scripts/export_answer_space_rules.py
python paper_neurips2026_artifacts/scripts/extract_qualitative_examples.py
```

## n=200 smoke jobs
```bash
bash paper_neurips2026_artifacts/jobs/submit_ablation_n200.sh
bash paper_neurips2026_artifacts/jobs/submit_second_backbone_n200.sh
```

## n=1000 jobs
```bash
bash paper_neurips2026_artifacts/jobs/submit_textvqa_routed_n1000.sh
bash paper_neurips2026_artifacts/jobs/submit_ablation_n1000_priority.sh
```

## Collect post-run
```bash
python paper_neurips2026_artifacts/scripts/collect_ablation_results.py
python paper_neurips2026_artifacts/scripts/make_paper_tables.py --textvqa-cf-override 72.28
```

## Directory assumptions
- configs: `benchmark_configs/`
- results: `benchmark_results/`
- logs: `logs/`
- runtime cache: `.runtime_cache/`
- artifact outputs: `paper_neurips2026_artifacts/{tables,figures,reports,supplement,reproducibility}`

## Expected outputs
- Main tables: `paper_neurips2026_artifacts/tables/*.tex` and `*.csv`
- Figures: `paper_neurips2026_artifacts/figures/*.pdf`
- Reports: `paper_neurips2026_artifacts/reports/*.md`

## Hardware notes
- GPU jobs use Slurm wrapper `scripts/sbatch_clean.sh`.
- CPU-only scripts can run locally after results exist.
