#!/usr/bin/env python3
import argparse
from pathlib import Path

from common import load_metric_index, write_csv, to_latex_table, format_num


ABLS = [
    "frequency_only", "no_format", "no_base_bias", "no_length_risk",
    "no_answer_space", "majority_vote", "base_only", "first_view",
]
BENCHES = ["textvqa", "ocrvqa", "gqa", "chartqa", "ocrbench"]
DATASET = {
    "textvqa": "TextVQA_VAL",
    "ocrvqa": "OCRVQA_TEST",
    "gqa": "GQA_TestDev_Balanced",
    "chartqa": "ChartQA_TEST",
    "ocrbench": "OCRBench",
}


def find_score(rows, n, config, dataset):
    for r in rows:
        if int(r.get("n_samples", -1)) == int(n) and r.get("config") == config and r.get("dataset") == dataset:
            return float(r.get("score")), r.get("metric", "Overall")
    return None, None


def make_table(rows, n, out_csv, out_tex):
    rec = []
    for b in BENCHES:
        ds = DATASET[b]
        nocf_cfg = f"test_config_smolvlm2_v91_nocf_{b}"
        s_ref, metric = find_score(rows, n, nocf_cfg, ds)
        for a in ABLS:
            cfg = f"test_config_smolvlm2_v91_nocf_ablation_{a}_{b}"
            s, m = find_score(rows, n, cfg, ds)
            d = None if (s is None or s_ref is None) else (s - s_ref)
            rec.append({
                "Benchmark": b,
                "Ablation": a,
                "Metric": metric or m or "NA",
                "NoCF": format_num(s_ref),
                "AblationScore": format_num(s),
                "Delta_vs_NoCF": format_num(d),
                "n": str(n),
            })

    fields = ["Benchmark", "Ablation", "Metric", "NoCF", "AblationScore", "Delta_vs_NoCF", "n"]
    write_csv(out_csv, rec, fields)
    to_latex_table(
        out_tex,
        ["Benchmark", "Ablation", "Metric", "NoCF", "Ablation", "$\\Delta$", "n"],
        [[r[c] for c in fields] for r in rec],
        caption=f"V91-NoCF ablations on n={n} subsets.",
        label=f"tab:nocf_ablation_n{n}",
    )


def main():
    ap = argparse.ArgumentParser(description="Collect V91-NoCF ablation results into paper tables.")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()

    root = Path(args.repo_root)
    rows = load_metric_index(root / "logs" / "experiment_metric_index_20260429.json")

    tables = root / "paper_neurips2026_artifacts" / "tables"
    reports = root / "paper_neurips2026_artifacts" / "reports"

    make_table(rows, 200, tables / "nocf_ablation_n200.csv", tables / "nocf_ablation_n200.tex")
    make_table(rows, 1000, tables / "nocf_ablation_n1000.csv", tables / "nocf_ablation_n1000.tex")

    rep = []
    rep.append("# NoCF Ablation Report")
    rep.append("")
    rep.append("This report summarizes current availability of NoCF component ablations.")
    rep.append("Rows with `NA` indicate jobs not run yet or missing in metric index.")
    rep.append("")
    rep.append("Interpretation guidance:")
    rep.append("- `frequency_only` tests whether view frequency alone explains gains.")
    rep.append("- `no_format` tests format-validity contribution.")
    rep.append("- `no_base_bias` tests stabilizing anchor effect.")
    rep.append("- `no_length_risk` tests degeneration control.")
    rep.append("- `majority_vote` is a weaker heuristic baseline.")
    (reports / "nocf_ablation_report.md").write_text("\n".join(rep) + "\n")

    print(tables / "nocf_ablation_n200.csv")
    print(tables / "nocf_ablation_n1000.csv")


if __name__ == "__main__":
    main()
