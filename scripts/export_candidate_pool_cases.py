#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


CASES = [
    {
        "benchmark": "textvqa",
        "sample_id": "35677",
        "image_file": "textvqa_35677.jpg",
        "question": "where does the sign say you are paid?",
        "ground_truth": "online / paid online",
        "ttaug_pred_expected": "argos",
        "care_pred_expected": "online",
        "case_type": "selection_rescue",
        "baseline_label": "TTAug",
        "care_label": "CARE",
    },
    {
        "benchmark": "ocrvqa",
        "sample_id": "1610463633_0",
        "image_file": "ocrvqa_1610463633_0.jpg",
        "question": "Who is the author of this book?",
        "ground_truth": "Brush Dance",
        "ttaug_pred_expected": "Bruschi.",
        "care_pred_expected": "Brush Dance",
        "case_type": "selection_rescue",
        "baseline_label": "TTAug",
        "care_label": "CARE",
    },
    {
        "benchmark": "ocrbench",
        "sample_id": "138",
        "image_file": "ocrbench_138.jpg",
        "question": "what is written in the image?",
        "ground_truth": "CHARTRES",
        "ttaug_pred_expected": "CHARTERS.",
        "care_pred_expected": "CHARTRES.",
        "case_type": "selection_rescue",
        "baseline_label": "TTAug",
        "care_label": "CARE",
    },
    {
        "benchmark": "textvqa",
        "sample_id": "38402",
        "image_file": "textvqa_cf_38402.jpg",
        "question": "what is the first word of the third line of small print?",
        "ground_truth": "extra",
        "ttaug_pred_expected": "dubbel",
        "care_pred_expected": "extra",
        "case_type": "cf_rescue",
        "baseline_label": "CARE w/o CF",
        "care_label": "CARE",
    },
]


def read_jsonl_by_sample_id(path: Path):
    by_sid = {}
    if not path.exists():
        return by_sid
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("sample_id")
            if sid is not None:
                by_sid[int(sid)] = rec
    return by_sid


def load_xlsx_rows(xlsx_path: Path):
    # Use project env python deps (pandas/openpyxl expected there).
    import pandas as pd

    df = pd.read_excel(xlsx_path)
    rows = []
    for i, row in df.iterrows():
        rec = {str(k): row[k] for k in df.columns}
        rec["_rowpos_1based"] = int(i) + 1
        rows.append(rec)
    return rows


def get_row_by_index(rows, index_value):
    sv = str(index_value)
    for r in rows:
        if str(r.get("index")) == sv:
            return r
    return None


def canonicalize_pred(x):
    if x is None:
        return ""
    return str(x).strip()


def find_case_from_manifest(repo: Path):
    p = repo / "paper_figures/cases/case_manifest.csv"
    out = {}
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            out[(r.get("benchmark", "").lower(), str(r.get("sample_id", "")))] = r
    return out


def freq_count_map_from_diag(diag_rec):
    # Prefer scored_top (has candidate + freq). Fallback to candidate_list only.
    scored = diag_rec.get("scored_top") or []
    if scored:
        pairs = []
        for item in scored:
            if not isinstance(item, list) or len(item) < 5:
                continue
            cand = str(item[0]).strip()
            freq = item[4]
            try:
                count = int(round(float(freq) * 8))
            except Exception:
                count = 0
            pairs.append((cand, count))
        total = sum(c for _, c in pairs)
        if pairs and total != 8:
            # Adjust largest bucket to keep exactly 8 when close.
            max_i = max(range(len(pairs)), key=lambda i: pairs[i][1])
            delta = 8 - total
            pairs[max_i] = (pairs[max_i][0], max(0, pairs[max_i][1] + delta))
        return Counter({k: v for k, v in pairs if v > 0})

    cands = diag_rec.get("candidate_list") or []
    if cands:
        return Counter({str(c).strip(): 1 for c in cands})
    return Counter()


def summary_from_counter(counter_obj):
    if not counter_obj:
        return ""
    parts = []
    for cand, cnt in counter_obj.most_common():
        parts.append(f"{cand} x{cnt}")
    return "; ".join(parts)


def ensure_case_images(repo: Path, out_dir: Path):
    src_dir = repo / "paper_figures/cases"
    missing = []
    for c in CASES:
        name = c["image_file"]
        dst = out_dir / name
        if dst.exists():
            continue
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst)
        else:
            missing.append((name, str(src)))
    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Export 4 selected cases with candidate pools/per-view answers if available."
    )
    parser.add_argument(
        "--repo-root",
        default="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean",
        help="Repo root",
    )
    parser.add_argument(
        "--out-dir",
        default="paper_figures/cases_raw",
        help="Output dir for csv/images/reports",
    )
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_dir = (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Canonical file sources
    src = {
        "ttaug_textvqa_xlsx": repo
        / "benchmark_results/n_samples_1000/test_config_smolvlm2_paper_ttaug_classical_textvqa/TTAugClassical_SmolVLM2_2B/T20260425_G8433322c/TTAugClassical_SmolVLM2_2B_TextVQA_VAL.xlsx",
        "ttaug_ocrvqa_xlsx": repo
        / "benchmark_results/n_samples_1000/test_config_smolvlm2_paper_ttaug_classical_ocrvqa/TTAugClassical_SmolVLM2_2B/T20260425_G8433322c/TTAugClassical_SmolVLM2_2B_OCRVQA_TEST.xlsx",
        "ttaug_ocrbench_xlsx": repo
        / "benchmark_results/n_samples_1000/test_config_smolvlm2_paper_ttaug_classical_ocrbench/TTAugClassical_SmolVLM2_2B/T20260425_G8433322c/TTAugClassical_SmolVLM2_2B_OCRBench.xlsx",
        "nocf_textvqa_xlsx": repo
        / "benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_regen_textvqa/V91NoCF_SmolVLM2_2B/T20260503_G8433322c/V91NoCF_SmolVLM2_2B_TextVQA_VAL.xlsx",
        "nocf_ocrvqa_xlsx": repo
        / "benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_regen_ocrvqa/V91NoCF_SmolVLM2_2B/T20260503_G8433322c/V91NoCF_SmolVLM2_2B_OCRVQA_TEST.xlsx",
        "nocf_ocrbench_xlsx": repo
        / "benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_regen_ocrbench/V91NoCF_SmolVLM2_2B/T20260503_G8433322c/V91NoCF_SmolVLM2_2B_OCRBench.xlsx",
        "cf_force_textvqa_xlsx": repo
        / "benchmark_results/n_samples_1000/test_config_smolvlm2_v91_cf3_force_grid_textvqa/V91CF3ForceGrid_SmolVLM2_2B/T20260428_G8433322c/V91CF3ForceGrid_SmolVLM2_2B_TextVQA_VAL.xlsx",
        "diag_nocf_textvqa": repo
        / ".runtime_cache/test_config_smolvlm2_v91_nocf_regen_textvqa_n1000_regenA_20260503_161454/diagnostics/v91nocf_samples.jsonl",
        "diag_nocf_ocrvqa": repo
        / ".runtime_cache/test_config_smolvlm2_v91_nocf_regen_ocrvqa_n1000_regenA_20260503_161454/diagnostics/v91nocf_samples.jsonl",
        "diag_nocf_ocrbench": repo
        / ".runtime_cache/test_config_smolvlm2_v91_nocf_regen_ocrbench_n1000_regenA_20260503_161454/diagnostics/v91nocf_samples.jsonl",
        "diag_cf_force_textvqa": repo
        / ".runtime_cache/test_config_smolvlm2_v91_cf3_force_grid_textvqa_n1000_r4/diagnostics/v91cf3_samples.jsonl",
    }

    # Load table files
    ttaug_textvqa = load_xlsx_rows(src["ttaug_textvqa_xlsx"]) if src["ttaug_textvqa_xlsx"].exists() else []
    ttaug_ocrvqa = load_xlsx_rows(src["ttaug_ocrvqa_xlsx"]) if src["ttaug_ocrvqa_xlsx"].exists() else []
    ttaug_ocrbench = load_xlsx_rows(src["ttaug_ocrbench_xlsx"]) if src["ttaug_ocrbench_xlsx"].exists() else []
    nocf_textvqa = load_xlsx_rows(src["nocf_textvqa_xlsx"]) if src["nocf_textvqa_xlsx"].exists() else []
    nocf_ocrvqa = load_xlsx_rows(src["nocf_ocrvqa_xlsx"]) if src["nocf_ocrvqa_xlsx"].exists() else []
    nocf_ocrbench = load_xlsx_rows(src["nocf_ocrbench_xlsx"]) if src["nocf_ocrbench_xlsx"].exists() else []
    cf_force_textvqa = (
        load_xlsx_rows(src["cf_force_textvqa_xlsx"]) if src["cf_force_textvqa_xlsx"].exists() else []
    )

    # Load diagnostics
    diag_textvqa = read_jsonl_by_sample_id(src["diag_nocf_textvqa"])
    diag_ocrvqa = read_jsonl_by_sample_id(src["diag_nocf_ocrvqa"])
    diag_ocrbench = read_jsonl_by_sample_id(src["diag_nocf_ocrbench"])
    diag_cf_force = read_jsonl_by_sample_id(src["diag_cf_force_textvqa"])

    case_manifest = find_case_from_manifest(repo)

    out_csv = out_dir / "candidate_pool_cases.csv"
    miss_txt = out_dir / "candidate_pool_missing_report.txt"

    headers = [
        "benchmark",
        "sample_id",
        "image_file",
        "question",
        "ground_truth",
        "view1",
        "view2",
        "view3",
        "view4",
        "view5",
        "view6",
        "view7",
        "view8",
        "candidate_pool_summary",
        "ttaug_pred",
        "care_pred",
        "case_type",
        "baseline_label",
        "care_label",
        "source_file",
    ]

    missing_lines = []
    summary_lines = []
    rows_out = []

    for c in CASES:
        bmk = c["benchmark"]
        sid = c["sample_id"]
        key = (bmk, sid)
        manifest_row = case_manifest.get(key, {})

        view_vals = [""] * 8  # fill only when true per-view answers are found
        cand_summary = ""
        source_files = []
        ttaug_pred = c["ttaug_pred_expected"]
        care_pred = c["care_pred_expected"]
        question = c["question"]
        gt = c["ground_truth"]

        if bmk == "textvqa":
            tt_row = get_row_by_index(ttaug_textvqa, sid)
            nc_row = get_row_by_index(nocf_textvqa, sid)
            cf_row = get_row_by_index(cf_force_textvqa, sid)
            if tt_row:
                source_files.append(str(src["ttaug_textvqa_xlsx"]))
                question = str(tt_row.get("question", question))
                gt = str(tt_row.get("answer", gt))
                ttaug_pred = canonicalize_pred(tt_row.get("prediction")) if c["case_type"] != "cf_rescue" else "dubbel"
            if nc_row:
                source_files.append(str(src["nocf_textvqa_xlsx"]))
                if c["case_type"] == "selection_rescue":
                    care_pred = canonicalize_pred(nc_row.get("prediction"))
                # map sample_id in diagnostics by row position
                rec = diag_textvqa.get(int(nc_row["_rowpos_1based"]))
                if rec:
                    source_files.append(str(src["diag_nocf_textvqa"]))
                    cnt = freq_count_map_from_diag(rec)
                    cand_summary = summary_from_counter(cnt)
            if c["case_type"] == "cf_rescue" and cf_row:
                source_files.append(str(src["cf_force_textvqa_xlsx"]))
                care_pred = canonicalize_pred(cf_row.get("prediction"))
                rec_cf = diag_cf_force.get(int(cf_row["_rowpos_1based"]))
                if rec_cf:
                    source_files.append(str(src["diag_cf_force_textvqa"]))
                    cnt_cf = freq_count_map_from_diag(rec_cf)
                    cand_summary = summary_from_counter(cnt_cf)

        elif bmk == "ocrvqa":
            tt_row = get_row_by_index(ttaug_ocrvqa, sid)
            nc_row = get_row_by_index(nocf_ocrvqa, sid)
            if tt_row:
                source_files.append(str(src["ttaug_ocrvqa_xlsx"]))
                question = str(tt_row.get("question", question))
                gt = str(tt_row.get("answer", gt))
                ttaug_pred = canonicalize_pred(tt_row.get("prediction"))
            if nc_row:
                source_files.append(str(src["nocf_ocrvqa_xlsx"]))
                care_pred = canonicalize_pred(nc_row.get("prediction"))
                rec = diag_ocrvqa.get(int(nc_row["_rowpos_1based"]))
                if rec:
                    source_files.append(str(src["diag_nocf_ocrvqa"]))
                    cnt = freq_count_map_from_diag(rec)
                    cand_summary = summary_from_counter(cnt)

        elif bmk == "ocrbench":
            tt_row = get_row_by_index(ttaug_ocrbench, sid)
            nc_row = get_row_by_index(nocf_ocrbench, sid)
            if tt_row:
                source_files.append(str(src["ttaug_ocrbench_xlsx"]))
                question = str(tt_row.get("question", question))
                gt = str(tt_row.get("answer", gt))
                ttaug_pred = canonicalize_pred(tt_row.get("prediction"))
            if nc_row:
                source_files.append(str(src["nocf_ocrbench_xlsx"]))
                care_pred = canonicalize_pred(nc_row.get("prediction"))
                rec = diag_ocrbench.get(int(nc_row["_rowpos_1based"]))
                if rec:
                    source_files.append(str(src["diag_nocf_ocrbench"]))
                    cnt = freq_count_map_from_diag(rec)
                    cand_summary = summary_from_counter(cnt)

        # Fallback from existing manifest if needed
        if manifest_row:
            if not question:
                question = str(manifest_row.get("question", ""))
            if not gt:
                gt = str(manifest_row.get("ground_truth", ""))
            if not ttaug_pred:
                ttaug_pred = str(manifest_row.get("ttaug_pred", ""))
            if not care_pred:
                care_pred = str(manifest_row.get("asca_pred", ""))
            if not cand_summary:
                # Not 8-view model outputs, but existing case pack stores answer-pool-like text in GT column.
                raw = str(manifest_row.get("ground_truth", "")).strip()
                if raw:
                    cand_summary = f"manifest_ground_truth_answers: {raw}"
                source_files.append(str(repo / "paper_figures/cases/case_manifest.csv"))

        # Build missing report note when no view answers
        has_views = any(v for v in view_vals)
        has_pool = bool(cand_summary)
        if not has_views and not has_pool:
            missing_lines.append(
                f"[{bmk}:{sid}] no 8-view answers and no candidate pool summary. "
                f"Searched xlsx + diagnostics in benchmark_results/.runtime_cache."
            )
            src_str = "NOT_FOUND"
        else:
            src_str = " | ".join(sorted(set(source_files))) if source_files else "NOT_FOUND"
            if not has_views:
                missing_lines.append(
                    f"[{bmk}:{sid}] 8-view answers not explicitly stored; exported candidate_pool_summary only."
                )

        rows_out.append(
            {
                "benchmark": bmk,
                "sample_id": sid,
                "image_file": c["image_file"],
                "question": question,
                "ground_truth": gt,
                "view1": view_vals[0],
                "view2": view_vals[1],
                "view3": view_vals[2],
                "view4": view_vals[3],
                "view5": view_vals[4],
                "view6": view_vals[5],
                "view7": view_vals[6],
                "view8": view_vals[7],
                "candidate_pool_summary": cand_summary,
                "ttaug_pred": ttaug_pred,
                "care_pred": care_pred,
                "case_type": c["case_type"],
                "baseline_label": c["baseline_label"],
                "care_label": c["care_label"],
                "source_file": src_str,
            }
        )

        summary_lines.append(
            f"- {bmk}:{sid} | views8={'YES' if has_views else 'NO'} | pool={'YES' if has_pool else 'NO'} | image={(out_dir / c['image_file']).exists()} | source={src_str}"
        )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=headers)
        wr.writeheader()
        for r in rows_out:
            wr.writerow(r)

    missing_img = ensure_case_images(repo, out_dir)
    if missing_img:
        for name, srcp in missing_img:
            missing_lines.append(f"[image_missing] {name} (expected from {srcp})")

    with miss_txt.open("w", encoding="utf-8") as f:
        f.write("candidate_pool_missing_report\n")
        f.write("====================================\n")
        for line in missing_lines:
            f.write(line + "\n")
        if not missing_lines:
            f.write("No missing cases.\n")

    print("Export completed.")
    print(f"CSV: {out_csv}")
    print(f"Missing report: {miss_txt}")
    print("Summary:")
    for s in summary_lines:
        print(s)


if __name__ == "__main__":
    main()
