#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from common import load_metric_index, find_row


def parse_setting_from_cfg(cfg: str):
    # test_config_smolvlm2_v91_nocf_sens_<setting>_<bench>
    p = cfg.replace("test_config_smolvlm2_v91_nocf_sens_", "")
    if p.endswith("_textvqa"):
        return p[:-8], "textvqa"
    if p.endswith("_chartqa"):
        return p[:-8], "chartqa"
    return p, "unknown"


def main():
    ap = argparse.ArgumentParser(description="Summarize official GroupA true-pipeline sensitivity.")
    ap.add_argument("--repo-root", default="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out-csv", default="paper_neurips2026_artifacts/sensitivity_asca_weights/groupA_truepipeline_official_summary.csv")
    args = ap.parse_args()

    root = Path(args.repo_root)
    rows = load_metric_index(root / "logs" / "experiment_metric_index_20260429.json")

    routed_ref = {
        "textvqa": ("test_config_smolvlm2_v91_cf3_routed_textvqa", "V91CF3Routed_SmolVLM2_2B", "TextVQA_VAL"),
        "chartqa": ("test_config_smolvlm2_v91_cf3_routed_chartqa", "V91CF3Routed_SmolVLM2_2B", "ChartQA_TEST"),
    }
    sens_cfgs = [
        r for r in rows
        if int(r.get("n_samples", -1)) == args.n
        and str(r.get("config", "")).startswith("test_config_smolvlm2_v91_nocf_sens_")
    ]

    out = []
    for r in sens_cfgs:
        cfg = str(r["config"])
        setting, bench = parse_setting_from_cfg(cfg)
        rr = routed_ref.get(bench)
        if rr is None:
            continue
        ref = find_row(rows, n_samples=args.n, config=rr[0], model=rr[1], dataset=rr[2])
        ref_score = float(ref["score"]) if ref else None
        score = float(r["score"])
        out.append({
            "benchmark": bench,
            "setting": setting,
            "score": f"{score:.6f}",
            "routed_default_score": ("NA" if ref_score is None else f"{ref_score:.6f}"),
            "delta_vs_routed_default": ("NA" if ref_score is None else f"{(score-ref_score):.6f}"),
            "config": cfg,
            "model": r["model"],
            "metric": r.get("metric", "Overall"),
        })

    out.sort(key=lambda x: (x["benchmark"], x["setting"]))
    out_path = root / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()) if out else [
            "benchmark", "setting", "score", "routed_default_score",
            "delta_vs_routed_default", "config", "model", "metric"
        ])
        w.writeheader()
        for row in out:
            w.writerow(row)
    print(f"[DONE] {out_path}")


if __name__ == "__main__":
    main()

