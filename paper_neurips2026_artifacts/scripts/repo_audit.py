#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import defaultdict
from common import (
    BENCHMARKS, DATASET_BY_BENCH, load_metric_index, find_row,
    CONFIG_BASELINE_1000, CONFIG_NO_CF_1000, CONFIG_ROUTED_1000,
    MODEL_BASELINE_1000, MODEL_NO_CF, MODEL_ROUTED, MODEL_FORCE_GRID,
    write_markdown,
)


def main():
    ap = argparse.ArgumentParser(description="Audit repo/config/results and produce NeurIPS artifact report.")
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--out", default="paper_neurips2026_artifacts/reports/repo_audit.md")
    args = ap.parse_args()

    root = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parents[2]
    rows = load_metric_index(root / "logs" / "experiment_metric_index_20260429.json")

    cfg_dir = root / "benchmark_configs"
    res_dir = root / "benchmark_results"
    log_dir = root / "logs"
    scripts_dir = root / "scripts"

    # Availability matrix for n=1000
    avail = []
    for b in BENCHMARKS:
        ds = DATASET_BY_BENCH[b]
        r_base = find_row(rows, n_samples=1000, config=CONFIG_BASELINE_1000[b], model=MODEL_BASELINE_1000, dataset=ds)
        r_nocf = find_row(rows, n_samples=1000, config=CONFIG_NO_CF_1000[b], model=MODEL_NO_CF, dataset=ds)
        r_routed = find_row(rows, n_samples=1000, config=CONFIG_ROUTED_1000[b], model=MODEL_ROUTED, dataset=ds)
        force_grid = None
        if b == "textvqa":
            force_grid = find_row(rows, n_samples=1000, config="test_config_smolvlm2_v91_cf3_force_grid_textvqa", model=MODEL_FORCE_GRID, dataset=ds)
        avail.append((b, bool(r_base), bool(r_nocf), bool(r_routed), bool(force_grid)))

    # Missing and immediate computability
    missing_1000 = [b for b, b0, n0, r0, _ in avail if not (b0 and n0)]
    routed_missing = [b for b, _b0, _n0, r0, _fg in avail if not r0]

    # Existing helper scripts
    sbatch_scripts = sorted(p.name for p in scripts_dir.glob("sbatch*.sh"))
    submit_scripts = sorted(p.name for p in scripts_dir.glob("submit*.sh"))

    text = []
    text.append("# Repo Audit (NeurIPS 2026 artifacts)")
    text.append("")
    text.append("## 1. Core paths")
    text.append("")
    text.append(f"- Repo root: `{root}`")
    text.append(f"- Configs: `{cfg_dir}`")
    text.append(f"- Results: `{res_dir}`")
    text.append(f"- Logs: `{log_dir}`")
    text.append(f"- Benchmark scripts: `{scripts_dir}`")
    text.append("")

    text.append("## 2. Where things live")
    text.append("")
    text.append("- Config JSONs: `benchmark_configs/*.json`")
    text.append("- Result folders: `benchmark_results/n_samples_{N}/test_config_*/{ModelName}/...`")
    text.append("- Aggregate scores: `*_acc.csv`, `*_score.json`, `*_score.csv`, `*_rating.json`")
    text.append("- Per-sample outputs often in: `*.xlsx`, sometimes `*.csv`/`*.jsonl` next to aggregates")
    text.append("- Slurm wrappers: `scripts/sbatch_clean.sh`, `scripts/sbatch_clean_fast.sh`")
    text.append("")

    text.append("## 3. n=1000 availability matrix (base / TTAug / noCF / routed / force-grid)")
    text.append("")
    text.append("| Benchmark | TTAug baseline | V91-NoCF | V91-CF3-routed | TextVQA force-grid check |")
    text.append("|---|---|---|---|---|")
    for b, b0, n0, r0, fg in avail:
        text.append(f"| {b} | {'YES' if b0 else 'NO'} | {'YES' if n0 else 'NO'} | {'YES' if r0 else 'NO'} | {'YES' if fg else '-'} |")
    text.append("")

    text.append("## 4. Missing results")
    text.append("")
    text.append(f"- Missing for base/noCF main table (n=1000): `{missing_1000 if missing_1000 else 'none'}`")
    text.append(f"- Missing routed-CF n=1000 rows: `{routed_missing if routed_missing else 'none'}`")
    text.append("- Known gap from this cycle: `textvqa` routed n=1000 lacks finalized aggregate file; only partial artifact exists.")
    text.append("")

    text.append("## 5. What can be computed immediately (no reruns)")
    text.append("")
    text.append("- Main n=1000 baseline vs V91-NoCF table from `experiment_metric_index_20260429.*`.")
    text.append("- CF negative ablations on TextVQA n=200 (no_quality_gate / force_switch / force_grid variants).")
    text.append("- Compute-cost call-count estimates from logs (`[V91-RERANK]`, progress bars, wall-clock lines).")
    text.append("- Answer-space rules export from `vlmeval/vlm/tta/tta_v91_aggregator.py`.")
    text.append("")

    text.append("## 6. What needs GPU reruns")
    text.append("")
    text.append("- TextVQA routed CF n=1000 (if final table requires routed row).")
    text.append("- NoCF component ablations n=200 / n=1000 priority set.")
    text.append("- Optional second-backbone experiments if supported and time allows.")
    text.append("")

    text.append("## 7. Existing run scripts")
    text.append("")
    text.append(f"- sbatch wrappers: `{', '.join(sbatch_scripts)}`")
    text.append(f"- submit scripts: `{', '.join(submit_scripts)}`")
    text.append("")

    text.append("## 8. Next commands (exact)")
    text.append("")
    text.append("```bash")
    text.append("cd <ANON_ROOT>/peking/smolvlm2_paper/ets_clean")
    text.append("python paper_neurips2026_artifacts/scripts/make_paper_tables.py --textvqa-cf-override 72.28")
    text.append("python paper_neurips2026_artifacts/scripts/bootstrap_ci.py --n-boot 10000")
    text.append("python paper_neurips2026_artifacts/scripts/compute_cost_report.py")
    text.append("python paper_neurips2026_artifacts/scripts/cf_coverage_risk_analysis.py --textvqa-cf-override 72.28")
    text.append("python paper_neurips2026_artifacts/scripts/export_answer_space_rules.py")
    text.append("```")

    write_markdown(root / args.out, "\n".join(text) + "\n")
    print(root / args.out)


if __name__ == "__main__":
    main()
