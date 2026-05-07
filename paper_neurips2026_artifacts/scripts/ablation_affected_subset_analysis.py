#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

from common import read_table_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_asca_sensitivity import (  # noqa: E402
    DATASET_TAG,
    _format_validity,
    _length_risk,
    build_features,
    discover_diag,
    discover_xlsx,
    extract_support,
    is_correct,
    load_diag_map,
    normalize_answer,
    norm_text,
)


BENCHMARKS = ["textvqa", "ocrvqa", "gqa", "chartqa", "ocrbench"]
ABLATIONS = [
    ("frequency_only", "test_config_smolvlm2_v91_nocf_ablation_frequency_only_{b}", "V91NoCFAbl_frequency_only_SmolVLM2_2B"),
    ("no_format", "test_config_smolvlm2_v91_nocf_ablation_no_format_{b}", "V91NoCFAbl_no_format_SmolVLM2_2B"),
    ("no_base_consistency", None, None),  # handled from artifact predictions
    ("no_length_risk", "test_config_smolvlm2_v91_nocf_ablation_no_length_risk_{b}", "V91NoCFAbl_no_length_risk_SmolVLM2_2B"),
    ("majority_vote", "test_config_smolvlm2_v91_nocf_ablation_majority_vote_{b}", "V91NoCFAbl_majority_vote_SmolVLM2_2B"),
]


def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def find_primary_diag(repo_root: Path, benchmark: str, n: int):
    for v in ["no_format", "no_length_risk", "frequency_only", "majority_vote", "no_base_bias"]:
        p = discover_diag(repo_root, f"test_config_smolvlm2_v91_nocf_ablation_{v}_{benchmark}_n{n}")
        if p is not None:
            return p
    return None


def load_full_rows(repo_root: Path, benchmark: str, n: int):
    ds = DATASET_TAG.get(benchmark, "")
    xlsx = discover_xlsx(
        repo_root,
        n,
        f"test_config_smolvlm2_v91_nocf_{benchmark}",
        "V91NoCF_SmolVLM2_2B",
        ds,
    )
    if xlsx is None:
        return None, None
    rows = read_table_file(xlsx)
    return xlsx, rows


def load_ablation_predictions(repo_root: Path, out_root: Path, benchmark: str, n: int, ablation: str):
    if ablation == "no_base_consistency":
        p = out_root.parent / "no_base_consistency_n1000" / benchmark / "predictions.csv"
        if not p.exists():
            return None, None
        rows = read_table_file(p)
        pred = {}
        for r in rows:
            sid = str(r.get("sample_id", "")).strip()
            if sid:
                pred[sid] = str(r.get("prediction", ""))
        return p, pred

    cfg_tpl = None
    model = None
    for name, c, m in ABLATIONS:
        if name == ablation:
            cfg_tpl = c
            model = m
            break
    if cfg_tpl is None:
        return None, None

    ds = DATASET_TAG.get(benchmark, "")
    xlsx = discover_xlsx(repo_root, n, cfg_tpl.format(b=benchmark), model, ds)
    if xlsx is None:
        return None, None

    rows = read_table_file(xlsx)
    pred = {}
    for i, r in enumerate(rows, 1):
        pred[str(i)] = str(r.get("prediction", ""))
    return xlsx, pred


def compute_sample_meta(diag_obj, question):
    if not isinstance(diag_obj, dict):
        return {
            "answer_space": "unknown",
            "low_margin_le_1_8": False,
            "low_margin_le_2_8": False,
            "format_affected": False,
            "length_risk_affected": False,
        }

    support_map = extract_support(diag_obj.get("scored_top"))
    feats, answer_space, base_norm = build_features(diag_obj, question or "", support_map)
    supports = sorted([to_float(v) for v in [f.get("view_freq", 0.0) for f in feats.values()]], reverse=True)
    if len(supports) >= 2:
        margin = supports[0] - supports[1]
    else:
        margin = 1.0

    fmt_vals = set()
    base_len = feats.get(base_norm, {}).get("length_words", 0)
    risk_vals = []
    for cand in feats.keys():
        fmt_vals.add(_format_validity(cand, feats, answer_space))
        risk_vals.append(_length_risk(cand, feats, answer_space, base_len))

    format_affected = len(fmt_vals) >= 2
    length_affected = any(r > 0 for r in risk_vals) and any(abs(r) <= 1e-12 for r in risk_vals)

    return {
        "answer_space": answer_space or "unknown",
        "low_margin_le_1_8": margin <= (1.0 / 8.0),
        "low_margin_le_2_8": margin <= (2.0 / 8.0),
        "format_affected": format_affected,
        "length_risk_affected": length_affected,
    }


def safe_div(a, b):
    if b == 0:
        return None
    return a / b


def fmt6(x):
    if x is None:
        return "NA"
    return f"{x:.6f}"


def main():
    ap = argparse.ArgumentParser(description="Affected-subset ablation analysis from existing n=1000 outputs (no VLM regeneration).")
    ap.add_argument("--repo_root", default=str(REPO_ROOT))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    ap.add_argument(
        "--output_dir",
        default=None,
        help="Default: paper_neurips2026_artifacts/ablations/affected_subset_n1000",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    out_root = Path(args.output_dir) if args.output_dir else (repo_root / "paper_neurips2026_artifacts" / "ablations" / "affected_subset_n1000")
    out_root.mkdir(parents=True, exist_ok=True)

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    changed_rows = []
    subset_rows = []
    space_rows = []
    missing_notes = []

    for b in benchmarks:
        full_xlsx, full_rows = load_full_rows(repo_root, b, args.n)
        if full_rows is None:
            missing_notes.append((b, "missing_full_xlsx"))
            continue

        diag_path = find_primary_diag(repo_root, b, args.n)
        diag_map = load_diag_map(diag_path) if diag_path is not None else {}
        if diag_path is None:
            missing_notes.append((b, "missing_diag"))

        full_pred = {}
        gt_map = {}
        q_map = {}
        row_map = {}
        meta_map = {}

        for i, r in enumerate(full_rows, 1):
            sid = str(i)
            full_pred[sid] = str(r.get("prediction", ""))
            gt_map[sid] = str(r.get("answer", ""))
            q_map[sid] = str(r.get("question", ""))
            row_map[sid] = r
            meta_map[sid] = compute_sample_meta(diag_map.get(sid), q_map[sid])

        for ablation, _, _ in ABLATIONS:
            src_path, abl_pred = load_ablation_predictions(repo_root, out_root, b, args.n, ablation)
            if abl_pred is None:
                missing_notes.append((b, f"missing_{ablation}"))
                continue

            sample_ids = [sid for sid in full_pred.keys() if sid in abl_pred]
            if not sample_ids:
                missing_notes.append((b, f"no_overlap_{ablation}"))
                continue

            changed = 0
            full_wins = 0
            abl_wins = 0
            changed_correct_full = 0
            changed_correct_abl = 0

            # prepare answer-space aggregators
            by_space = {}
            for sid in sample_ids:
                sp = meta_map[sid]["answer_space"]
                if sp not in by_space:
                    by_space[sp] = {
                        "n_total": 0,
                        "changed": 0,
                        "full_wins": 0,
                        "ablation_wins": 0,
                        "full_correct_total": 0,
                        "ablation_correct_total": 0,
                        "full_correct_changed": 0,
                        "ablation_correct_changed": 0,
                    }

            for sid in sample_ids:
                dp = full_pred[sid]
                apred = abl_pred[sid]
                gt = gt_map[sid]
                row = row_map[sid]

                d_ok = bool(is_correct(gt, dp, row))
                a_ok = bool(is_correct(gt, apred, row))
                changed_flag = norm_text(dp) != norm_text(apred)

                sp = meta_map[sid]["answer_space"]
                agg = by_space[sp]
                agg["n_total"] += 1
                agg["full_correct_total"] += int(d_ok)
                agg["ablation_correct_total"] += int(a_ok)

                if changed_flag:
                    changed += 1
                    agg["changed"] += 1
                    changed_correct_full += int(d_ok)
                    changed_correct_abl += int(a_ok)
                    agg["full_correct_changed"] += int(d_ok)
                    agg["ablation_correct_changed"] += int(a_ok)

                    if d_ok and not a_ok:
                        full_wins += 1
                        agg["full_wins"] += 1
                    elif a_ok and not d_ok:
                        abl_wins += 1
                        agg["ablation_wins"] += 1

            changed_subset_full_acc = safe_div(changed_correct_full, changed)
            changed_subset_ablation_acc = safe_div(changed_correct_abl, changed)
            changed_subset_delta = (
                None if (changed_subset_full_acc is None or changed_subset_ablation_acc is None)
                else (changed_subset_ablation_acc - changed_subset_full_acc)
            )

            changed_rows.append({
                "benchmark": b,
                "ablation": ablation,
                "n": len(sample_ids),
                "source_path": str(src_path),
                "changed": changed,
                "full_wins": full_wins,
                "ablation_wins": abl_wins,
                "full_net_gain": full_wins - abl_wins,
                "changed_subset_full_acc": fmt6(changed_subset_full_acc),
                "changed_subset_ablation_acc": fmt6(changed_subset_ablation_acc),
                "changed_subset_delta": fmt6(changed_subset_delta),
            })

            # Diagnostic subsets
            subsets = [
                ("low_margin_le_1_8", lambda m: m["low_margin_le_1_8"]),
                ("low_margin_le_2_8", lambda m: m["low_margin_le_2_8"]),
                ("format_affected", lambda m: m["format_affected"]),
                ("length_risk_affected", lambda m: m["length_risk_affected"]),
            ]
            for subset_name, fn in subsets:
                sids = [sid for sid in sample_ids if fn(meta_map[sid])]
                if not sids:
                    subset_rows.append({
                        "benchmark": b,
                        "ablation": ablation,
                        "subset_name": subset_name,
                        "subset_n": 0,
                        "changed": 0,
                        "full_wins": 0,
                        "ablation_wins": 0,
                        "full_net_gain": 0,
                        "changed_subset_full_acc": "NA",
                        "changed_subset_ablation_acc": "NA",
                        "changed_subset_delta": "NA",
                    })
                    continue

                schanged = sfw = saw = 0
                scf = sca = 0
                for sid in sids:
                    dp = full_pred[sid]
                    apred = abl_pred[sid]
                    row = row_map[sid]
                    gt = gt_map[sid]
                    d_ok = bool(is_correct(gt, dp, row))
                    a_ok = bool(is_correct(gt, apred, row))
                    if norm_text(dp) != norm_text(apred):
                        schanged += 1
                        scf += int(d_ok)
                        sca += int(a_ok)
                        if d_ok and not a_ok:
                            sfw += 1
                        elif a_ok and not d_ok:
                            saw += 1

                sf_acc = safe_div(scf, schanged)
                sa_acc = safe_div(sca, schanged)
                sdelta = None if (sf_acc is None or sa_acc is None) else (sa_acc - sf_acc)

                subset_rows.append({
                    "benchmark": b,
                    "ablation": ablation,
                    "subset_name": subset_name,
                    "subset_n": len(sids),
                    "changed": schanged,
                    "full_wins": sfw,
                    "ablation_wins": saw,
                    "full_net_gain": sfw - saw,
                    "changed_subset_full_acc": fmt6(sf_acc),
                    "changed_subset_ablation_acc": fmt6(sa_acc),
                    "changed_subset_delta": fmt6(sdelta),
                })

            # By answer space
            for sp, agg in sorted(by_space.items()):
                ch = agg["changed"]
                cf_acc = safe_div(agg["full_correct_changed"], ch)
                ca_acc = safe_div(agg["ablation_correct_changed"], ch)
                cdelta = None if (cf_acc is None or ca_acc is None) else (ca_acc - cf_acc)

                space_rows.append({
                    "benchmark": b,
                    "ablation": ablation,
                    "answer_space": sp,
                    "subset_n": agg["n_total"],
                    "changed": ch,
                    "full_wins": agg["full_wins"],
                    "ablation_wins": agg["ablation_wins"],
                    "full_net_gain": agg["full_wins"] - agg["ablation_wins"],
                    "full_acc_overall": fmt6(safe_div(agg["full_correct_total"], agg["n_total"])),
                    "ablation_acc_overall": fmt6(safe_div(agg["ablation_correct_total"], agg["n_total"])),
                    "changed_subset_full_acc": fmt6(cf_acc),
                    "changed_subset_ablation_acc": fmt6(ca_acc),
                    "changed_subset_delta": fmt6(cdelta),
                })

    # outputs
    changed_csv = out_root / "ablation_changed_cases_all.csv"
    with changed_csv.open("w", newline="") as f:
        fields = [
            "benchmark", "ablation", "n", "source_path", "changed", "full_wins", "ablation_wins",
            "full_net_gain", "changed_subset_full_acc", "changed_subset_ablation_acc", "changed_subset_delta",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(changed_rows)

    subset_csv = out_root / "ablation_affected_subset_summary.csv"
    with subset_csv.open("w", newline="") as f:
        fields = [
            "benchmark", "ablation", "subset_name", "subset_n", "changed", "full_wins", "ablation_wins",
            "full_net_gain", "changed_subset_full_acc", "changed_subset_ablation_acc", "changed_subset_delta",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(subset_rows)

    by_space_csv = out_root / "ablation_by_answer_space.csv"
    with by_space_csv.open("w", newline="") as f:
        fields = [
            "benchmark", "ablation", "answer_space", "subset_n", "changed", "full_wins", "ablation_wins",
            "full_net_gain", "full_acc_overall", "ablation_acc_overall",
            "changed_subset_full_acc", "changed_subset_ablation_acc", "changed_subset_delta",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(space_rows)

    # report
    overall_path = repo_root / "paper_neurips2026_artifacts" / "ablations" / "no_base_consistency_n1000" / "ablation_n1000_complete.csv"
    overall_text = ""
    if overall_path.exists():
        overall_text = overall_path.read_text()

    lines = []
    lines.append("# Ablation Affected-Subset Analysis (n=1000)")
    lines.append("")
    lines.append("## 保留的 overall ablation（原表）")
    lines.append("")
    if overall_text:
        lines.append("```csv")
        lines.append(overall_text.strip())
        lines.append("```")
    else:
        lines.append("- 未找到 overall 表。")
    lines.append("")
    lines.append("## Changed-case 总结")
    lines.append("")
    if changed_rows:
        lines.append("```csv")
        with changed_csv.open() as f:
            lines.append(f.read().strip())
        lines.append("```")
    else:
        lines.append("- 无可用 changed-case 结果。")
    lines.append("")
    lines.append("## 说明")
    lines.append("- changed/full_wins/ablation_wins/full_net_gain 均按 sample-level correctness 计算。")
    lines.append("- low-margin 子集定义：top1 support - top2 support <= 1/8 或 <= 2/8。")
    lines.append("- format-affected 子集：candidate pool 里 contract validity 至少两档。")
    lines.append("- length-risk-affected 子集：candidate pool 同时存在 risk=0 与 risk>0 候选。")
    if missing_notes:
        lines.append("")
        lines.append("## 缺失/跳过")
        for b, reason in missing_notes:
            lines.append(f"- {b}: {reason}")

    report_md = out_root / "ablation_diagnostic_report.md"
    report_md.write_text("\n".join(lines) + "\n")

    print(f"[DONE] {changed_csv}")
    print(f"[DONE] {subset_csv}")
    print(f"[DONE] {by_space_csv}")
    print(f"[DONE] {report_md}")


if __name__ == "__main__":
    main()

