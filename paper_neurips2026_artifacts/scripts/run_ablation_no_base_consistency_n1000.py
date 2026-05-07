#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from common import read_table_file
from run_asca_sensitivity import (
    DATASET_TAG,
    normalize_answer,
    norm_text,
    is_correct,
    discover_xlsx,
    discover_diag,
    load_diag_map,
    extract_support,
    rerank_one,
)

TARGET_BENCHMARKS = ["textvqa", "ocrvqa", "gqa", "chartqa", "ocrbench"]
SETTING = {
    "method": "no_base_consistency",
    "w_sup": 2.0,
    "w_valid": 1.0,
    "w_base": 0.0,
    "w_risk": 0.5,
}

EXISTING_ABLATIONS = [
    ("full", "test_config_smolvlm2_v91_nocf_{b}", "V91NoCF_SmolVLM2_2B"),
    ("frequency_only", "test_config_smolvlm2_v91_nocf_ablation_frequency_only_{b}", "V91NoCFAbl_frequency_only_SmolVLM2_2B"),
    ("majority_vote", "test_config_smolvlm2_v91_nocf_ablation_majority_vote_{b}", "V91NoCFAbl_majority_vote_SmolVLM2_2B"),
    ("no_format", "test_config_smolvlm2_v91_nocf_ablation_no_format_{b}", "V91NoCFAbl_no_format_SmolVLM2_2B"),
    ("no_length_risk", "test_config_smolvlm2_v91_nocf_ablation_no_length_risk_{b}", "V91NoCFAbl_no_length_risk_SmolVLM2_2B"),
]


def parse_acc_csv(path: Path):
    if not path or not path.exists():
        return None
    try:
        rows = list(csv.reader(path.open()))
        if len(rows) >= 2 and rows[1]:
            return float(rows[1][0])
    except Exception:
        return None
    return None


def find_acc_csv_for_config(repo_root: Path, n: int, config: str):
    d = repo_root / "benchmark_results" / f"n_samples_{n}" / config
    if not d.exists():
        return None
    cands = sorted(d.glob("*/*/*_acc.csv"))
    if not cands:
        return None
    return cands[0]


def proxy_score_from_rows(rows):
    correct = 0
    n = len(rows)
    for r in rows:
        gt = str(r.get("answer", ""))
        pred = str(r.get("prediction", ""))
        if is_correct(gt, pred, r):
            correct += 1
    return (correct / n) if n > 0 else None


def find_or_none(repo_root: Path, n: int, cfg: str, model: str, bench: str):
    ds = DATASET_TAG.get(bench, "")
    return discover_xlsx(repo_root, n, cfg, model, ds)


def format2(x):
    if x is None:
        return "NA"
    return f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser(description="Run no_base_consistency ablation using cached candidates only (no VLM generation).")
    ap.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--benchmarks", default=",".join(TARGET_BENCHMARKS))
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    out_root = Path(args.output_dir) if args.output_dir else (repo_root / "paper_neurips2026_artifacts" / "ablations" / "no_base_consistency_n1000")
    out_root.mkdir(parents=True, exist_ok=True)

    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    summary_rows = []
    changed_rows = []
    merged_rows = []
    skip_notes = []

    for b in benches:
        bench_dir = out_root / b
        bench_dir.mkdir(parents=True, exist_ok=True)
        run_log = bench_dir / "run.log"

        def log(msg):
            with run_log.open("a") as f:
                f.write(msg + "\n")
            print(msg)

        run_log.write_text("")
        log(f"[INFO] benchmark={b} n={args.n}")
        log("[INFO] method = no_base_consistency")
        log(f"[INFO] w_sup = {SETTING['w_sup']}")
        log(f"[INFO] w_valid = {SETTING['w_valid']}")
        log(f"[INFO] w_base = {SETTING['w_base']}")
        log(f"[INFO] w_risk = {SETTING['w_risk']}")

        cfg_full = f"test_config_smolvlm2_v91_nocf_{b}"
        model_full = "V91NoCF_SmolVLM2_2B"
        full_xlsx = find_or_none(repo_root, args.n, cfg_full, model_full, b)
        if full_xlsx is None:
            log("[WARN] missing full ASCA xlsx; skip")
            skip_notes.append((b, "missing_full_xlsx"))
            continue
        log(f"[INFO] full_xlsx={full_xlsx}")

        # use no_format diag as primary (most complete), fallback to others.
        main_diag_path = None
        for v in ["no_format", "no_length_risk", "frequency_only", "majority_vote", "no_base_bias"]:
            p = discover_diag(repo_root, f"test_config_smolvlm2_v91_nocf_ablation_{v}_{b}_n{args.n}")
            if p is not None:
                main_diag_path = p
                break
        if main_diag_path is None:
            log("[WARN] missing candidate diagnostics; skip")
            skip_notes.append((b, "missing_candidate_diag"))
            continue
        log(f"[INFO] candidate_diag={main_diag_path}")

        aux_diags = {}
        for v in ["no_format", "no_length_risk", "frequency_only", "majority_vote", "no_base_bias"]:
            p = discover_diag(repo_root, f"test_config_smolvlm2_v91_nocf_ablation_{v}_{b}_n{args.n}")
            if p is not None:
                aux_diags[v] = load_diag_map(p)

        main_diag = load_diag_map(main_diag_path)
        full_rows = read_table_file(full_xlsx)

        default_pred = {}
        gt_map = {}
        q_map = {}
        row_map = {}
        for i, r in enumerate(full_rows, 1):
            sid = str(i)
            default_pred[sid] = str(r.get("prediction", ""))
            gt_map[sid] = str(r.get("answer", ""))
            q_map[sid] = str(r.get("question", ""))
            row_map[sid] = r

        # frequency-only predictions (for changed_vs_frequency_only)
        freq_xlsx = find_or_none(
            repo_root, args.n,
            f"test_config_smolvlm2_v91_nocf_ablation_frequency_only_{b}",
            "V91NoCFAbl_frequency_only_SmolVLM2_2B",
            b,
        )
        freq_pred = {}
        if freq_xlsx is not None:
            for i, r in enumerate(read_table_file(freq_xlsx), 1):
                freq_pred[str(i)] = str(r.get("prediction", ""))
            log(f"[INFO] frequency_xlsx={freq_xlsx}")

        pred_rows = []
        setting_pred = {}
        missing_diag_samples = 0
        missing_support_candidates = 0

        for sid in default_pred.keys():
            diag = main_diag.get(sid)
            if diag is None:
                missing_diag_samples += 1
                setting_pred[sid] = default_pred[sid]
                continue

            support_map = {}
            for _, dmap in aux_diags.items():
                d = dmap.get(sid)
                if d is None:
                    continue
                sm = extract_support(d.get("scored_top"))
                for k, v in sm.items():
                    support_map[k] = v

            cand_list = diag.get("candidate_list") or []
            for c in cand_list:
                nrm = normalize_answer(c)
                if nrm and nrm not in support_map:
                    missing_support_candidates += 1

            pred, _, _ = rerank_one(diag, q_map.get(sid, ""), support_map, SETTING)
            if pred is None:
                pred = default_pred[sid]
            setting_pred[sid] = pred

        changed_vs_full = 0
        default_wins = 0
        ablation_wins = 0
        correct = 0
        changed_vs_freq = 0 if freq_pred else None

        for sid, dp in default_pred.items():
            sp = setting_pred.get(sid, dp)
            if norm_text(sp) != norm_text(dp):
                changed_vs_full += 1

            row = row_map[sid]
            gt = gt_map[sid]
            d_ok = is_correct(gt, dp, row)
            s_ok = is_correct(gt, sp, row)
            if d_ok and not s_ok:
                default_wins += 1
            elif s_ok and not d_ok:
                ablation_wins += 1
            if s_ok:
                correct += 1

            if changed_vs_freq is not None:
                fp = freq_pred.get(sid)
                if fp is not None and norm_text(fp) != norm_text(sp):
                    changed_vs_freq += 1

            pred_rows.append({
                "sample_id": sid,
                "prediction": sp,
                "full_prediction": dp,
                "changed_vs_full": int(norm_text(sp) != norm_text(dp)),
            })

        score_proxy = correct / max(1, len(default_pred))
        full_score = proxy_score_from_rows(full_rows)
        metric_name = "accuracy_proxy"

        delta_vs_full = (score_proxy - full_score) if (score_proxy is not None and full_score is not None) else None

        pred_path = bench_dir / "predictions.csv"
        with pred_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sample_id", "prediction", "full_prediction", "changed_vs_full"])
            w.writeheader(); w.writerows(pred_rows)

        eval_res = {
            "benchmark": b,
            "n": len(default_pred),
            "method": "no_base_consistency",
            "weights": SETTING,
            "metric": metric_name,
            "score": score_proxy,
            "full_score": full_score,
            "delta_vs_full": delta_vs_full,
            "changed_vs_full": changed_vs_full,
            "full_wins": default_wins,
            "ablation_wins": ablation_wins,
            "net_vs_full": ablation_wins - default_wins,
            "changed_vs_frequency_only": changed_vs_freq,
            "candidate_diag": str(main_diag_path),
            "missing_diag_samples": missing_diag_samples,
            "missing_support_candidates": missing_support_candidates,
        }
        (bench_dir / "eval_results.json").write_text(json.dumps(eval_res, indent=2))

        summary_rows.append({
            "benchmark": b,
            "n": len(default_pred),
            "method": "no_base_consistency",
            "w_sup": SETTING["w_sup"],
            "w_valid": SETTING["w_valid"],
            "w_base": SETTING["w_base"],
            "w_risk": SETTING["w_risk"],
            "metric": metric_name,
            "score": f"{score_proxy:.6f}",
            "full_score": ("NA" if full_score is None else f"{full_score:.6f}"),
            "delta_vs_full": ("NA" if delta_vs_full is None else f"{delta_vs_full:.6f}"),
            "changed_vs_full": changed_vs_full,
            "full_wins": default_wins,
            "ablation_wins": ablation_wins,
            "net_vs_full": ablation_wins - default_wins,
            "output_path": str(bench_dir),
        })

        changed_rows.append({
            "benchmark": b,
            "ablation": "no_base_consistency",
            "changed": changed_vs_full,
            "full_wins": default_wins,
            "ablation_wins": ablation_wins,
            "net_vs_full": ablation_wins - default_wins,
        })

        # frequency_only changed-case diagnostic vs full
        if freq_xlsx is not None:
            fr_rows = read_table_file(freq_xlsx)
            freq_map = {str(i): str(r.get("prediction", "")) for i, r in enumerate(fr_rows, 1)}
            ch = fw = aw = 0
            for sid, fp in freq_map.items():
                if sid not in default_pred:
                    continue
                dp = default_pred[sid]
                if norm_text(fp) == norm_text(dp):
                    continue
                ch += 1
                row = row_map[sid]
                gt = gt_map[sid]
                d_ok = is_correct(gt, dp, row)
                f_ok = is_correct(gt, fp, row)
                if d_ok and not f_ok:
                    fw += 1
                elif f_ok and not d_ok:
                    aw += 1
            changed_rows.append({
                "benchmark": b,
                "ablation": "frequency_only",
                "changed": ch,
                "full_wins": fw,
                "ablation_wins": aw,
                "net_vs_full": aw - fw,
            })

    # write summary csv
    summary_csv = out_root / "ablation_no_base_consistency_n1000.csv"
    with summary_csv.open("w", newline="") as f:
        fields = [
            "benchmark", "n", "method", "w_sup", "w_valid", "w_base", "w_risk",
            "metric", "score", "full_score", "delta_vs_full", "changed_vs_full",
            "full_wins", "ablation_wins", "net_vs_full", "output_path",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    # changed-case table
    changed_csv = out_root / "ablation_changed_cases_n1000.csv"
    with changed_csv.open("w", newline="") as f:
        fields = ["benchmark", "ablation", "changed", "full_wins", "ablation_wins", "net_vs_full"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in changed_rows:
            w.writerow(r)

    # merged ablation table
    # Use a unified sample-level proxy score from xlsx for comparability across
    # Full and ablations, including no_base_consistency rerank-only outputs.
    score_map = {}
    for b in TARGET_BENCHMARKS:
        for col, cfg_tpl, model in EXISTING_ABLATIONS:
            cfg = cfg_tpl.format(b=b)
            x = find_or_none(repo_root, args.n, cfg, model, b)
            score = None
            if x is not None:
                rows = read_table_file(x)
                score = proxy_score_from_rows(rows)
            score_map[(b, col)] = score

    no_base_scores = {}
    for r in summary_rows:
        no_base_scores[r["benchmark"]] = float(r["score"]) if r.get("score") not in (None, "NA") else None

    merged_path = out_root / "ablation_n1000_complete.csv"
    merged_fields = ["benchmark", "full", "frequency_only", "majority_vote", "no_format", "no_base_consistency", "no_length_risk"]
    with merged_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=merged_fields)
        w.writeheader()
        for b in TARGET_BENCHMARKS:
            row = {
                "benchmark": b,
                "full": score_map.get((b, "full")),
                "frequency_only": score_map.get((b, "frequency_only")),
                "majority_vote": score_map.get((b, "majority_vote")),
                "no_format": score_map.get((b, "no_format")),
                "no_base_consistency": no_base_scores.get(b),
                "no_length_risk": score_map.get((b, "no_length_risk")),
            }
            w.writerow(row)
            merged_rows.append(row)

    # latex table (2 decimals)
    tex_path = out_root / "ablation_n1000_complete_latex.tex"
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Benchmark & Full & Freq. only & Majority & No format & No base & No length risk \\\\")
    lines.append("\\midrule")
    for r in merged_rows:
        vals = [
            r["benchmark"],
            format2(r["full"]),
            format2(r["frequency_only"]),
            format2(r["majority_vote"]),
            format2(r["no_format"]),
            format2(r["no_base_consistency"]),
            format2(r["no_length_risk"]),
        ]
        lines.append("{} & {} & {} & {} & {} & {} & {} \\\\".format(*vals))
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{ASCA n=1000 ablation table including no-base-consistency (w_base=0.0).}")
    lines.append("\\label{tab:ablation_n1000_complete}")
    lines.append("\\end{table}")
    tex_path.write_text("\n".join(lines) + "\n")

    # report
    report = []
    report.append("# no_base_consistency Ablation Report (n=1000)")
    report.append("")
    report.append("## 定义")
    report.append("- method = no_base_consistency")
    report.append("- score = 2.0*support + 1.0*valid + 0.0*base - 0.5*risk")
    report.append("")
    report.append("## 跑的 benchmark")
    if summary_rows:
        for r in summary_rows:
            report.append(
                f"- {r['benchmark']}: score={r['score']}, full={r['full_score']}, "
                f"delta={r['delta_vs_full']}, changed={r['changed_vs_full']}, "
                f"full_wins={r['full_wins']}, ablation_wins={r['ablation_wins']}, net={r['net_vs_full']}"
            )
    else:
        report.append("- 无成功 benchmark")
    report.append("")

    if skip_notes:
        report.append("## 跳过/缺失")
        for b, why in skip_notes:
            report.append(f"- {b}: {why}")
        report.append("")

    report.append("## 解释")
    report.append("- 如果 changed_vs_full 很小：base consistency 目前更多是 sparse anchor。")
    report.append("- 如果 delta_vs_full 很小：说明 ASCA 并非简单复制 base answer。")
    report.append("- 如果某 benchmark no_base_consistency 更好：可视为 potential over-anchoring case。")
    report.append("")

    report.append("## 产物路径")
    report.append(f"- {summary_csv}")
    report.append(f"- {merged_path}")
    report.append(f"- {tex_path}")
    report.append(f"- {changed_csv}")
    report.append(f"- {out_root}")
    report.append("")
    report.append("## 注意")
    report.append("- 本实验只做 rerank+eval，未重新调用 VLM generation。")
    (out_root / "ablation_no_base_consistency_report.md").write_text("\n".join(report) + "\n")

    print(f"[DONE] summary={summary_csv}")
    print(f"[DONE] merged={merged_path}")
    print(f"[DONE] latex={tex_path}")
    print(f"[DONE] changed={changed_csv}")
    print(f"[DONE] report={out_root / 'ablation_no_base_consistency_report.md'}")


if __name__ == "__main__":
    main()
