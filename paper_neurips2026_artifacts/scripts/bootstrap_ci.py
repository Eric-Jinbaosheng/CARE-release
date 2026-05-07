#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common import (
    BENCHMARKS,
    DATASET_BY_BENCH,
    CONFIG_BASELINE_1000,
    CONFIG_NO_CF_1000,
    CONFIG_ROUTED_1000,
    MODEL_BASELINE_1000,
    MODEL_NO_CF,
    MODEL_ROUTED,
    MODEL_FORCE_GRID,
    load_metric_index,
    find_row,
    bootstrap_mean_ci,
    bootstrap_delta_ci,
    find_candidate_sample_files,
    load_sample_score_map,
    write_csv,
    to_latex_table,
    format_num,
)


def _result_dir(root: Path, row: dict) -> Path:
    return root / "benchmark_results" / f"n_samples_{row['n_samples']}" / row["config"] / row["model"]


def _best_sample_map(root: Path, row: Optional[dict]) -> Tuple[Optional[Dict[str, float]], str, str]:
    if row is None:
        return None, "missing_row", ""
    d = _result_dir(root, row)
    if not d.exists():
        return None, "missing_result_dir", str(d)
    files = find_candidate_sample_files(d)
    for f in files:
        m, status = load_sample_score_map(f)
        if m:
            return m, "ok", str(f)
    return None, "aggregate_only", (str(files[0]) if files else "")


def _add_result(out_rows: List[dict], bench: str, metric: str, baseline: str, method: str,
                baseline_score: Optional[float], method_score: Optional[float],
                pairing: str, ci: str, pval: str, note: str):
    delta = None
    if baseline_score is not None and method_score is not None:
        delta = method_score - baseline_score
    out_rows.append({
        "Benchmark": bench,
        "Metric": metric,
        "Baseline": format_num(baseline_score),
        "Method": format_num(method_score),
        "Delta": format_num(delta),
        "95% CI": ci,
        "p-value": pval,
        "Pairing status": pairing,
        "Notes": note,
    })


def main():
    ap = argparse.ArgumentParser(description="Bootstrap CI and paired significance from existing benchmark outputs.")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    root = Path(args.repo_root)
    rows = load_metric_index(root / "logs" / "experiment_metric_index_20260429.json")

    main_rows = []
    cf_rows = []

    for b in BENCHMARKS:
        ds = DATASET_BY_BENCH[b]

        r_base = find_row(rows, n_samples=1000, config=CONFIG_BASELINE_1000[b], model=MODEL_BASELINE_1000, dataset=ds)
        r_nocf = find_row(rows, n_samples=1000, config=CONFIG_NO_CF_1000[b], model=MODEL_NO_CF, dataset=ds)
        if not r_base or not r_nocf:
            _add_result(main_rows, b, (r_nocf or r_base or {}).get("metric", "NA"),
                        "TTAug", "V91-NoCF",
                        r_base["score"] if r_base else None,
                        r_nocf["score"] if r_nocf else None,
                        "missing", "NA (aggregate only)", "NA", "missing row")
            continue

        base_map, base_status, base_src = _best_sample_map(root, r_base)
        nocf_map, nocf_status, nocf_src = _best_sample_map(root, r_nocf)
        metric = r_nocf.get("metric", "Overall")

        if base_map and nocf_map:
            common_ids = sorted(set(base_map) & set(nocf_map))
            if common_ids:
                a = [nocf_map[i] for i in common_ids]
                bvals = [base_map[i] for i in common_ids]
                d, lo, hi, p = bootstrap_delta_ci(a, bvals, n_boot=args.n_boot, seed=args.seed, alpha=args.alpha)
                ci = f"[{format_num(lo)}, {format_num(hi)}]"
                _add_result(main_rows, b, metric, "TTAug", "V91-NoCF",
                            r_base["score"], r_nocf["score"],
                            f"paired:{len(common_ids)}", ci, f"{p:.4g}",
                            f"base:{Path(base_src).name}; nocf:{Path(nocf_src).name}")
            else:
                _add_result(main_rows, b, metric, "TTAug", "V91-NoCF",
                            r_base["score"], r_nocf["score"],
                            "no_overlap", "NA (aggregate only)", "NA",
                            "no common sample IDs")
        else:
            _add_result(main_rows, b, metric, "TTAug", "V91-NoCF",
                        r_base["score"], r_nocf["score"],
                        f"{base_status}/{nocf_status}", "NA (aggregate only)", "NA",
                        f"base:{base_src}; nocf:{nocf_src}")

        # CF rows: routed vs noCF
        r_routed = find_row(rows, n_samples=1000, config=CONFIG_ROUTED_1000[b], model=MODEL_ROUTED, dataset=ds)
        if r_routed:
            routed_map, routed_status, routed_src = _best_sample_map(root, r_routed)
            if nocf_map and routed_map:
                common_ids = sorted(set(nocf_map) & set(routed_map))
                if common_ids:
                    a = [routed_map[i] for i in common_ids]
                    bvals = [nocf_map[i] for i in common_ids]
                    d, lo, hi, p = bootstrap_delta_ci(a, bvals, n_boot=args.n_boot, seed=args.seed, alpha=args.alpha)
                    ci = f"[{format_num(lo)}, {format_num(hi)}]"
                    _add_result(cf_rows, b, metric, "V91-NoCF", "V91-CF3-routed",
                                r_nocf["score"], r_routed["score"],
                                f"paired:{len(common_ids)}", ci, f"{p:.4g}",
                                f"routed:{Path(routed_src).name}")
                else:
                    _add_result(cf_rows, b, metric, "V91-NoCF", "V91-CF3-routed",
                                r_nocf["score"], r_routed["score"],
                                "no_overlap", "NA (aggregate only)", "NA",
                                "no common sample IDs")
            else:
                _add_result(cf_rows, b, metric, "V91-NoCF", "V91-CF3-routed",
                            r_nocf["score"], r_routed["score"],
                            routed_status, "NA (aggregate only)", "NA",
                            f"routed:{routed_src}")
        else:
            _add_result(cf_rows, b, metric, "V91-NoCF", "V91-CF3-routed",
                        r_nocf["score"], None, "missing", "NA (aggregate only)", "NA",
                        "missing routed row")

    # TextVQA force-grid vs noCF focused row
    ds = DATASET_BY_BENCH["textvqa"]
    r_nocf = find_row(rows, n_samples=1000, config=CONFIG_NO_CF_1000["textvqa"], model=MODEL_NO_CF, dataset=ds)
    r_fg = find_row(rows, n_samples=1000, config="test_config_smolvlm2_v91_cf3_force_grid_textvqa", model=MODEL_FORCE_GRID, dataset=ds)
    if r_nocf and r_fg:
        _add_result(cf_rows, "textvqa(force_grid)", r_fg.get("metric", "Overall"), "V91-NoCF", "V91-CF3-force-grid",
                    r_nocf["score"], r_fg["score"], "aggregate", "NA (aggregate only)", "NA",
                    "focused check with recorded rescue/harm in logs")

    out_dir = root / "paper_neurips2026_artifacts"
    tables = out_dir / "tables"
    reports = out_dir / "reports"

    main_fields = ["Benchmark", "Metric", "Baseline", "Method", "Delta", "95% CI", "p-value", "Pairing status", "Notes"]
    write_csv(tables / "bootstrap_ci_main.csv", main_rows, main_fields)
    write_csv(tables / "bootstrap_ci_cf.csv", cf_rows, main_fields)

    to_latex_table(
        tables / "bootstrap_ci_main.tex",
        ["Benchmark", "Metric", "Baseline", "Method", "$\\Delta$", "95\\% CI", "p-value", "Pairing"],
        [[r["Benchmark"], r["Metric"], r["Baseline"], r["Method"], r["Delta"], r["95% CI"], r["p-value"], r["Pairing status"]] for r in main_rows],
        caption="Bootstrap confidence intervals for V91-NoCF vs deterministic TTAug.",
        label="tab:bootstrap_main",
    )
    to_latex_table(
        tables / "bootstrap_ci_cf.tex",
        ["Benchmark", "Metric", "NoCF", "CF method", "$\\Delta$", "95\\% CI", "p-value", "Pairing"],
        [[r["Benchmark"], r["Metric"], r["Baseline"], r["Method"], r["Delta"], r["95% CI"], r["p-value"], r["Pairing status"]] for r in cf_rows],
        caption="Bootstrap confidence intervals for CF variants vs V91-NoCF.",
        label="tab:bootstrap_cf",
    )

    rep = []
    rep.append("# Bootstrap CI Report")
    rep.append("")
    rep.append(f"- n_boot: {args.n_boot}")
    rep.append(f"- seed: {args.seed}")
    rep.append("- Rule: if per-sample pairing cannot be recovered from available files, CI/p-value is reported as `NA (aggregate only)`.")
    rep.append("")
    rep.append("## Output files")
    rep.append("")
    rep.append(f"- `{tables / 'bootstrap_ci_main.csv'}`")
    rep.append(f"- `{tables / 'bootstrap_ci_main.tex'}`")
    rep.append(f"- `{tables / 'bootstrap_ci_cf.csv'}`")
    rep.append(f"- `{tables / 'bootstrap_ci_cf.tex'}`")

    (reports / "bootstrap_ci_report.md").write_text("\n".join(rep) + "\n")
    print(tables / "bootstrap_ci_main.csv")
    print(tables / "bootstrap_ci_cf.csv")


if __name__ == "__main__":
    main()
