#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def _norm_text(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_answers_field(ans):
    if ans is None:
        return []
    s = str(ans).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return [str(x) for x in v]
        except Exception:
            pass
    return [s]


def is_correct(pred, gt_field):
    p = _norm_text(pred)
    gts = parse_answers_field(gt_field)
    if not gts:
        return False
    g_norm = [_norm_text(x) for x in gts]
    return p in g_norm


def read_jsonl_diag(path: Path):
    out = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            sid = rec.get("sample_id")
            if sid is not None:
                out[int(sid)] = rec
    return out


def pool_summary_from_diag(rec, n_views=8):
    if not rec:
        return ""
    scored_top = rec.get("scored_top") or []
    if scored_top:
        pairs = []
        for it in scored_top:
            if not isinstance(it, list) or len(it) < 5:
                continue
            cand = str(it[0]).strip()
            try:
                cnt = int(round(float(it[4]) * n_views))
            except Exception:
                cnt = 0
            if cnt > 0:
                pairs.append((cand, cnt))
        if pairs:
            total = sum(c for _, c in pairs)
            if total != n_views:
                imax = max(range(len(pairs)), key=lambda i: pairs[i][1])
                pairs[imax] = (pairs[imax][0], max(0, pairs[imax][1] + (n_views - total)))
            c = Counter({k: v for k, v in pairs if v > 0})
            return "; ".join([f"{k} x{v}" for k, v in c.most_common()])
    cands = rec.get("candidate_list") or []
    if cands:
        uniq = []
        for x in cands:
            x = str(x).strip()
            if x and x not in uniq:
                uniq.append(x)
        return "; ".join([f"{x} x1" for x in uniq[:8]])
    return ""


def short_q(q, max_len=66):
    q = str(q).strip()
    return q if len(q) <= max_len else q[: max_len - 1] + "…"


def short_gt(gt):
    vals = parse_answers_field(gt)
    if not vals:
        return str(gt)
    if len(vals) == 1:
        return vals[0]
    return " / ".join(vals[:2])


def aspect_score(w, h):
    if not w or not h:
        return 0.0
    r = w / float(h)
    targets = [1.0, 1.25, 1.5]
    d = min(abs(r - t) for t in targets)
    return max(0.0, 1.0 - d)


def load_df(path):
    import pandas as pd

    df = pd.read_excel(path)
    df = df.copy()
    df["_rowpos_1b"] = range(1, len(df) + 1)
    return df


def find_image_path(benchmark, row, repo: Path):
    if benchmark == "textvqa":
        p = str(row.get("image_path", "")).strip()
        if p:
            return p
    if benchmark == "ocrvqa":
        fn = str(row.get("image_path", "")).strip()
        if fn:
            candidates = [
                f"<ANON_HOME_PATH><ANON_USER>/LMUData/images/OCRVQA_TEST/{fn}",
                f"<ANON_HOME_PATH><ANON_USER>/LMUData/images/OCRVQA/{fn}",
                str(repo / "paper_figures/cases" / f"ocrvqa_{row.get('index')}.jpg"),
            ]
            for p in candidates:
                if Path(p).exists():
                    return p
            return candidates[0]
    if benchmark == "ocrbench":
        sid = str(row.get("index"))
        candidates = [
            f"<ANON_HOME_PATH><ANON_USER>/LMUData/images/OCRBench/{sid}.jpg",
            f"<ANON_HOME_PATH><ANON_USER>/LMUData/images/OCRBench/{sid}.png",
            str(repo / "paper_figures/cases" / f"ocrbench_{sid}.jpg"),
        ]
        for p in candidates:
            if Path(p).exists():
                return p
        return candidates[0]
    return ""


def copy_image(src, dst):
    p = Path(src)
    if not p.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dst)
    return True


def make_contact_sheet(rows, out_path: Path, thumb_w=420, thumb_h=300):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return False, "PIL unavailable"

    n = len(rows)
    if n == 0:
        return False, "no rows"
    cols = 3
    cell_w = thumb_w
    cell_h = thumb_h + 120
    pad = 18
    rws = math.ceil(n / cols)
    canvas = Image.new("RGB", (cols * cell_w + (cols + 1) * pad, rws * cell_h + (rws + 1) * pad), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for i, r in enumerate(rows):
        rr, cc = divmod(i, cols)
        x0 = pad + cc * (cell_w + pad)
        y0 = pad + rr * (cell_h + pad)
        img_p = Path(r["copied_image"])
        if img_p.exists():
            img = Image.open(img_p).convert("RGB")
            img.thumbnail((thumb_w, thumb_h))
            ix = x0 + (thumb_w - img.width) // 2
            iy = y0 + (thumb_h - img.height) // 2
            canvas.paste(img, (ix, iy))
            draw.rectangle([x0, y0, x0 + thumb_w, y0 + thumb_h], outline=(180, 180, 180), width=1)
        else:
            draw.rectangle([x0, y0, x0 + thumb_w, y0 + thumb_h], outline=(200, 0, 0), width=2)
            draw.text((x0 + 8, y0 + 8), "image missing", fill=(180, 0, 0), font=font)

        text = (
            f"{r['rank']:02d} {r['benchmark']} {r['sample_id']}\n"
            f"Q: {short_q(r['question'], 58)}\n"
            f"GT: {short_q(r['ground_truth'], 58)}\n"
            f"B:{short_q(r['base_pred'],18)}  T:{short_q(r['ttaug_pred'],18)}  C:{short_q(r['care_pred'],18)}"
        )
        draw.text((x0 + 4, y0 + thumb_h + 6), text, fill=(20, 20, 20), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return True, ""


def main():
    parser = argparse.ArgumentParser(description="Build qualitative case-grid materials from existing outputs.")
    parser.add_argument(
        "--repo-root",
        default="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean",
        help="Repo root",
    )
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_dir = repo / "paper_figures/case_grid_materials"
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    # Fixed sources (official/same protocol set)
    src = {
        "base_textvqa": repo
        / "benchmark_results/n_samples_1000/test_config_smolvlm2_base_textvqa/Base_SmolVLM2_2B/T20260425_G8433322c/Base_SmolVLM2_2B_TextVQA_VAL.xlsx",
        "base_ocrvqa": repo
        / "benchmark_results/n_samples_1000/test_config_smolvlm2_base_ocrvqa/Base_SmolVLM2_2B/T20260425_G8433322c/Base_SmolVLM2_2B_OCRVQA_TEST.xlsx",
        "base_ocrbench": repo
        / "benchmark_results/n_samples_1000/test_config_smolvlm2_base_ocrbench/Base_SmolVLM2_2B/T20260425_G8433322c/Base_SmolVLM2_2B_OCRBench.xlsx",
        "ttaug_textvqa": repo
        / "benchmark_results/n_samples_1000/test_config_smolvlm2_paper_ttaug_classical_textvqa/TTAugClassical_SmolVLM2_2B/T20260425_G8433322c/TTAugClassical_SmolVLM2_2B_TextVQA_VAL.xlsx",
        "ttaug_ocrvqa": repo
        / "benchmark_results/n_samples_1000/test_config_smolvlm2_paper_ttaug_classical_ocrvqa/TTAugClassical_SmolVLM2_2B/T20260425_G8433322c/TTAugClassical_SmolVLM2_2B_OCRVQA_TEST.xlsx",
        "ttaug_ocrbench": repo
        / "benchmark_results/n_samples_1000/test_config_smolvlm2_paper_ttaug_classical_ocrbench/TTAugClassical_SmolVLM2_2B/T20260425_G8433322c/TTAugClassical_SmolVLM2_2B_OCRBench.xlsx",
        "care_textvqa": repo
        / "benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_regen_textvqa/V91NoCF_SmolVLM2_2B/T20260503_G8433322c/V91NoCF_SmolVLM2_2B_TextVQA_VAL.xlsx",
        "care_ocrvqa": repo
        / "benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_regen_ocrvqa/V91NoCF_SmolVLM2_2B/T20260503_G8433322c/V91NoCF_SmolVLM2_2B_OCRVQA_TEST.xlsx",
        "care_ocrbench": repo
        / "benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_regen_ocrbench/V91NoCF_SmolVLM2_2B/T20260503_G8433322c/V91NoCF_SmolVLM2_2B_OCRBench.xlsx",
        "cf_textvqa": repo
        / "benchmark_results/n_samples_1000/test_config_smolvlm2_v91_cf3_force_grid_textvqa/V91CF3ForceGrid_SmolVLM2_2B/T20260428_G8433322c/V91CF3ForceGrid_SmolVLM2_2B_TextVQA_VAL.xlsx",
        "diag_textvqa": repo
        / ".runtime_cache/test_config_smolvlm2_v91_nocf_regen_textvqa_n1000_regenA_20260503_161454/diagnostics/v91nocf_samples.jsonl",
        "diag_ocrvqa": repo
        / ".runtime_cache/test_config_smolvlm2_v91_nocf_regen_ocrvqa_n1000_regenA_20260503_161454/diagnostics/v91nocf_samples.jsonl",
        "diag_ocrbench": repo
        / ".runtime_cache/test_config_smolvlm2_v91_nocf_regen_ocrbench_n1000_regenA_20260503_161454/diagnostics/v91nocf_samples.jsonl",
    }

    missing = []
    for k, p in src.items():
        if not p.exists():
            missing.append(f"missing source: {k} -> {p}")

    if missing:
        (out_dir / "missing_report.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
        print("Some sources missing. Wrote missing_report and exited.")
        return

    import pandas as pd
    from PIL import Image

    b_text = load_df(src["base_textvqa"])
    t_text = load_df(src["ttaug_textvqa"])
    c_text = load_df(src["care_textvqa"])
    f_text = load_df(src["cf_textvqa"])

    b_ocrvqa = load_df(src["base_ocrvqa"])
    t_ocrvqa = load_df(src["ttaug_ocrvqa"])
    c_ocrvqa = load_df(src["care_ocrvqa"])

    b_ocrb = load_df(src["base_ocrbench"])
    t_ocrb = load_df(src["ttaug_ocrbench"])
    c_ocrb = load_df(src["care_ocrbench"])

    d_text = read_jsonl_diag(src["diag_textvqa"])
    d_ocrvqa = read_jsonl_diag(src["diag_ocrvqa"])
    d_ocrb = read_jsonl_diag(src["diag_ocrbench"])

    candidates = []
    skip_reasons = []

    def harvest(benchmark, bdf, tdf, cdf, diag_map, fdf=None):
        b_by = {str(r["index"]): r for _, r in bdf.iterrows()}
        t_by = {str(r["index"]): r for _, r in tdf.iterrows()}
        c_by = {str(r["index"]): r for _, r in cdf.iterrows()}
        f_by = {str(r["index"]): r for _, r in fdf.iterrows()} if fdf is not None else {}
        sids = set(c_by.keys()) & set(t_by.keys())
        for sid in sids:
            br = b_by.get(sid)
            tr = t_by.get(sid)
            cr = c_by.get(sid)
            if tr is None or cr is None:
                continue
            q = str(cr.get("question", tr.get("question", "")))
            gt_field = cr.get("answer", tr.get("answer", ""))
            base_pred = br.get("prediction", "") if br is not None else ""
            ttaug_pred = tr.get("prediction", "")
            care_pred = cr.get("prediction", "")
            cf_pred = f_by.get(sid, {}).get("prediction", "") if sid in f_by else ""
            image_path = find_image_path(benchmark, cr, repo)
            if not image_path:
                skip_reasons.append(f"[{benchmark}:{sid}] missing image_path field")
                continue
            pimg = Path(image_path)
            if not pimg.exists():
                skip_reasons.append(f"[{benchmark}:{sid}] image not found: {image_path}")
                continue

            base_ok = is_correct(base_pred, gt_field) if base_pred != "" else False
            t_ok = is_correct(ttaug_pred, gt_field)
            c_ok = is_correct(care_pred, gt_field)
            cf_ok = is_correct(cf_pred, gt_field) if cf_pred else False

            diag = diag_map.get(int(cr["_rowpos_1b"]))
            pool = pool_summary_from_diag(diag)
            pool_n = len([x for x in pool.split(";") if x.strip()]) if pool else 99

            # image quality/shape score
            try:
                with Image.open(pimg) as im:
                    w, h = im.size
            except Exception:
                w, h = 0, 0

            reason = []
            ctype = None
            if (not t_ok) and c_ok and (not base_ok):
                ctype = "base_ttaug_wrong_care_right"
                reason.append("Base/TTAug wrong, CARE right")
            elif (not t_ok) and c_ok:
                ctype = "ttaug_wrong_care_right"
                reason.append("TTAug wrong, CARE right")
            elif benchmark == "textvqa" and (not c_ok) and cf_pred and cf_ok:
                ctype = "care_cf_rescue"
                reason.append("CARE w/o CF wrong, CARE with CF right")
            elif t_ok and (not c_ok):
                ctype = "harm_case_for_appendix"
                reason.append("TTAug right, CARE wrong")

            if ctype is None:
                continue

            q_len = len(q.split())
            gt_short = short_gt(gt_field)
            gt_len = len(_norm_text(gt_short).split())

            score = 0.0
            score += 3.0 if ctype in ("base_ttaug_wrong_care_right", "ttaug_wrong_care_right") else 0.0
            score += 2.0 if ctype == "care_cf_rescue" else 0.0
            score += 1.5 * aspect_score(w, h)
            score += 1.0 if q_len <= 10 else 0.0
            score += 1.0 if gt_len <= 3 else 0.0
            score += max(0.0, 1.5 - 0.2 * pool_n)
            if benchmark == "textvqa":
                score += 0.25
            if benchmark == "ocrvqa":
                if _norm_text(gt_short) in ("yes", "no"):
                    ctype = "yesno_rescue" if "right" in ctype else ctype
                else:
                    ctype = "entity_rescue" if "right" in ctype else ctype
            if benchmark == "ocrbench" and "right" in ctype:
                ctype = "ocr_character_rescue"

            candidates.append(
                {
                    "benchmark": benchmark,
                    "sample_id": sid,
                    "image_path": str(pimg),
                    "question": q,
                    "ground_truth": str(gt_field),
                    "answer_pool_summary": pool,
                    "base_pred": str(base_pred),
                    "ttaug_pred": str(ttaug_pred),
                    "care_pred": str(care_pred),
                    "nocf_pred": str(care_pred),
                    "cf_pred": str(cf_pred),
                    "case_type": ctype,
                    "why_selected": "; ".join(reason),
                    "source_file": " | ".join(
                        [
                            str(src[f"base_{benchmark}"]),
                            str(src[f"ttaug_{benchmark}"]),
                            str(src[f"care_{benchmark}"]),
                            str(src[f"diag_{benchmark}"]),
                        ]
                        + ([str(src["cf_textvqa"])] if benchmark == "textvqa" else [])
                    ),
                    "_score": score,
                    "_q_len": q_len,
                    "_gt_len": gt_len,
                    "_pool_n": pool_n,
                    "_w": w,
                    "_h": h,
                }
            )

    harvest("textvqa", b_text, t_text, c_text, d_text, f_text)
    harvest("ocrvqa", b_ocrvqa, t_ocrvqa, c_ocrvqa, d_ocrvqa, None)
    harvest("ocrbench", b_ocrb, t_ocrb, c_ocrb, d_ocrb, None)

    # Enforce minimum mix
    by_bmk = defaultdict(list)
    for r in candidates:
        by_bmk[r["benchmark"]].append(r)
    for k in by_bmk:
        by_bmk[k].sort(key=lambda x: x["_score"], reverse=True)

    selected = []
    selected_ids = set()

    def pick_from(benchmark, n):
        arr = by_bmk.get(benchmark, [])
        cnt = 0
        for r in arr:
            key = (r["benchmark"], r["sample_id"], r["case_type"])
            if key in selected_ids:
                continue
            selected.append(r)
            selected_ids.add(key)
            cnt += 1
            if cnt >= n:
                break

    # Mandatory quotas
    pick_from("textvqa", 3)
    pick_from("ocrvqa", 2)
    pick_from("ocrbench", 2)

    # Ensure at least one CF rescue
    cf_rescues = [r for r in candidates if r["case_type"] == "care_cf_rescue"]
    cf_rescues.sort(key=lambda x: x["_score"], reverse=True)
    if cf_rescues:
        k = ("textvqa", cf_rescues[0]["sample_id"], cf_rescues[0]["case_type"])
        if k not in selected_ids:
            selected.append(cf_rescues[0])
            selected_ids.add(k)

    # Add 1-2 harms
    harms = [r for r in candidates if r["case_type"] == "harm_case_for_appendix"]
    harms.sort(key=lambda x: x["_score"], reverse=True)
    for h in harms[:2]:
        k = (h["benchmark"], h["sample_id"], h["case_type"])
        if k not in selected_ids:
            selected.append(h)
            selected_ids.add(k)

    # Fill to >=10 with best remaining
    rest = sorted(candidates, key=lambda x: x["_score"], reverse=True)
    for r in rest:
        if len(selected) >= 12:
            break
        k = (r["benchmark"], r["sample_id"], r["case_type"])
        if k in selected_ids:
            continue
        selected.append(r)
        selected_ids.add(k)

    # Final rank sort by score then readability
    selected.sort(key=lambda x: (x["_score"], -x["_q_len"], -x["_gt_len"], -x["_pool_n"]), reverse=True)

    # Export and copy images
    csv_path = out_dir / "case_grid_candidates.csv"
    md_path = out_dir / "case_grid_candidates.md"
    miss_path = out_dir / "missing_report.txt"
    sheet_path = out_dir / "contact_sheet.jpg"

    csv_cols = [
        "rank",
        "benchmark",
        "sample_id",
        "image_file",
        "image_path",
        "question",
        "ground_truth",
        "answer_pool_summary",
        "base_pred",
        "ttaug_pred",
        "care_pred",
        "nocf_pred",
        "cf_pred",
        "case_type",
        "why_selected",
        "source_file",
    ]

    out_rows = []
    md_lines = ["# Case Grid Candidates", ""]
    image_missing = []
    for i, r in enumerate(selected, start=1):
        ext = Path(r["image_path"]).suffix.lower() if Path(r["image_path"]).suffix else ".jpg"
        image_file = f"{i:02d}_{r['benchmark']}_{r['sample_id']}{ext}"
        dst = img_dir / image_file
        ok = copy_image(r["image_path"], dst)
        if not ok:
            image_missing.append(f"[{r['benchmark']}:{r['sample_id']}] cannot copy image from {r['image_path']}")
        out = {
            "rank": i,
            "benchmark": r["benchmark"],
            "sample_id": r["sample_id"],
            "image_file": image_file,
            "image_path": r["image_path"],
            "question": r["question"],
            "ground_truth": r["ground_truth"],
            "answer_pool_summary": r["answer_pool_summary"],
            "base_pred": r["base_pred"],
            "ttaug_pred": r["ttaug_pred"],
            "care_pred": r["care_pred"],
            "nocf_pred": r["nocf_pred"],
            "cf_pred": r["cf_pred"],
            "case_type": r["case_type"],
            "why_selected": r["why_selected"],
            "source_file": r["source_file"],
            "copied_image": str(dst),
        }
        out_rows.append(out)

        md_lines.extend(
            [
                f"## {i:02d} {r['benchmark'].upper()} sample {r['sample_id']}",
                f"- Image: paper_figures/case_grid_materials/images/{image_file}",
                f"- Q: {r['question']}",
                f"- GT: {short_gt(r['ground_truth'])}",
                f"- Answer pool: {r['answer_pool_summary'] if r['answer_pool_summary'] else '(missing)'}",
                f"- Base: {r['base_pred']}",
                f"- TTAug: {r['ttaug_pred']}",
                f"- CARE: {r['care_pred']}",
                f"- Type: {r['case_type']}",
                f"- Why good: {r['why_selected']}",
                "",
            ]
        )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=csv_cols)
        wr.writeheader()
        for r in out_rows:
            wr.writerow({k: r.get(k, "") for k in csv_cols})

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    ok_sheet, err_sheet = make_contact_sheet(out_rows, sheet_path)

    miss_lines = []
    miss_lines.extend(skip_reasons)
    miss_lines.extend(image_missing)
    if not ok_sheet:
        miss_lines.append(f"[contact_sheet] not created: {err_sheet}")
    miss_path.write_text("\n".join(miss_lines) + ("\n" if miss_lines else ""), encoding="utf-8")

    # Summary print
    bcount = Counter(r["benchmark"] for r in out_rows)
    pool_count = sum(1 for r in out_rows if r["answer_pool_summary"])
    base_count = sum(1 for r in out_rows if str(r["base_pred"]).strip() != "")
    print(f"Found candidates: {len(out_rows)}")
    print(f"By benchmark: {dict(bcount)}")
    print(f"With answer_pool_summary: {pool_count}")
    print(f"With base_pred: {base_count}")
    print(f"contact_sheet: {sheet_path}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
