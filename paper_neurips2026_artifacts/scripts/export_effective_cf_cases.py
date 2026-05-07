#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter


def draw_overlay(img: Image.Image, rel_box, ctrl_box) -> Image.Image:
    out = img.copy().convert("RGB")
    dr = ImageDraw.Draw(out)
    if rel_box and len(rel_box) == 4:
        dr.rectangle([int(x) for x in rel_box], outline=(255, 50, 50), width=4)
    if ctrl_box and len(ctrl_box) == 4:
        dr.rectangle([int(x) for x in ctrl_box], outline=(0, 200, 255), width=4)
    return out


def blur_box(img: Image.Image, box, radius=10) -> Image.Image:
    out = img.copy().convert("RGB")
    if not box or len(box) != 4:
        return out
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(out.width, x1))
    x2 = max(0, min(out.width, x2))
    y1 = max(0, min(out.height, y1))
    y2 = max(0, min(out.height, y2))
    if x2 <= x1 or y2 <= y1:
        return out
    crop = out.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(radius=radius))
    out.paste(crop, (x1, y1))
    return out


def pick_candidate_debug(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    dbg = row.get("cf_candidate_debug") or {}
    if not isinstance(dbg, dict) or not dbg:
        return None, None

    # Priority 1: cf_winner
    cf_winner = str(row.get("cf_winner", "")).strip()
    if cf_winner and cf_winner in dbg:
        return cf_winner, dbg.get(cf_winner)

    # Priority 2: no_cf_winner
    no_cf_winner = str(row.get("no_cf_winner", "")).strip()
    if no_cf_winner and no_cf_winner in dbg:
        return no_cf_winner, dbg.get(no_cf_winner)

    # Priority 3: max cf_score candidate
    cf_score = row.get("cf_score") or {}
    if isinstance(cf_score, dict) and cf_score:
        cand = max(cf_score.items(), key=lambda kv: kv[1])[0]
        if cand in dbg:
            return cand, dbg.get(cand)

    # fallback first
    k = next(iter(dbg.keys()))
    return k, dbg.get(k)


def to_builtin(x):
    if isinstance(x, dict):
        return {str(k): to_builtin(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_builtin(v) for v in x]
    if isinstance(x, tuple):
        return [to_builtin(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def resolve_image_path(benchmark: str, dfr) -> Optional[Path]:
    raw = str(dfr.get("image_path", "")).strip() if "image_path" in dfr else ""
    if raw:
        p = Path(raw)
        if p.exists():
            return p
        # common relative path in OCRVQA outputs, e.g. 1492612766.jpg
        p2 = Path("<ANON_HOME_PATH><ANON_USER>/LMUData/images/OCRVQA") / raw
        if p2.exists():
            return p2

    idx = str(dfr.get("index", "")).strip()
    if benchmark == "ocrbench" and idx:
        p = Path("<ANON_HOME_PATH><ANON_USER>/LMUData/images/OCRBench") / f"{idx}.jpg"
        if p.exists():
            return p
    if benchmark == "chartqa" and idx:
        p = Path("<ANON_HOME_PATH><ANON_USER>/LMUData/images/ChartQA_TEST") / f"{idx}.jpg"
        if p.exists():
            return p
    return None


def process_source(
    benchmark: str,
    variant: str,
    diag_path: Path,
    xlsx_path: Path,
    out_root: Path,
    max_cases: int,
):
    if not diag_path.exists() or not xlsx_path.exists():
        return []

    out_dir = out_root / f"{variant}_{benchmark}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(x) for x in diag_path.open() if x.strip()]
    df = pd.read_excel(xlsx_path)

    exported = []
    picked = 0

    for r in rows:
        used = bool(r.get("cf_used"))
        changed = bool(r.get("cf_final_changed"))
        if not (used or changed):
            continue

        sid = r.get("sample_id")
        if sid is None:
            continue
        try:
            row_id = int(sid)
        except Exception:
            continue

        if row_id < 1 or row_id > len(df):
            continue

        cand, dbg = pick_candidate_debug(r)
        if not dbg:
            continue

        rel_box = dbg.get("rel_box")
        ctrl_box = dbg.get("ctrl_box")
        if not (isinstance(rel_box, (list, tuple)) and isinstance(ctrl_box, (list, tuple))):
            continue

        dfr = df.iloc[row_id - 1]
        img_path = resolve_image_path(benchmark, dfr)
        if img_path is None or (not img_path.exists()):
            continue
        image_path = str(img_path)

        case_id = f"row{row_id:04d}"
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(img_path).convert("RGB")
        img.save(case_dir / "00_original.jpg", quality=95)
        draw_overlay(img, rel_box, ctrl_box).save(case_dir / "01_cf_boxes_overlay.jpg", quality=95)
        blur_box(img, rel_box, radius=10).save(case_dir / "02_rel_drop_blur.jpg", quality=95)
        blur_box(img, ctrl_box, radius=10).save(case_dir / "03_ctrl_drop_blur.jpg", quality=95)

        meta = {
            "benchmark": benchmark,
            "variant": variant,
            "row_id": row_id,
            "index": dfr.get("index", ""),
            "question": str(dfr.get("question", "")),
            "answer": str(dfr.get("answer", "")),
            "prediction": str(dfr.get("prediction", "")),
            "image_path": image_path,
            "cf_used": used,
            "cf_final_changed": changed,
            "block_reason": r.get("block_reason", ""),
            "answer_space": r.get("answer_space", ""),
            "cf_margin": r.get("cf_margin", ""),
            "no_cf_winner": r.get("no_cf_winner", ""),
            "cf_winner": r.get("cf_winner", ""),
            "final_answer": r.get("final_answer", ""),
            "picked_cf_candidate": cand,
            "rel_box": rel_box,
            "ctrl_box": ctrl_box,
            "rel_drop": dbg.get("rel_drop", ""),
            "ctrl_drop": dbg.get("ctrl_drop", ""),
            "source_diag": str(diag_path),
            "source_xlsx": str(xlsx_path),
            "case_dir": str(case_dir),
        }
        meta = to_builtin(meta)

        with (case_dir / "cf_case_meta.json").open("w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        with (case_dir / "cf_case_summary.txt").open("w") as f:
            for k in [
                "benchmark", "variant", "row_id", "index", "question", "answer", "prediction",
                "cf_used", "cf_final_changed", "block_reason", "answer_space", "cf_margin",
                "no_cf_winner", "cf_winner", "final_answer", "picked_cf_candidate",
                "rel_box", "ctrl_box", "rel_drop", "ctrl_drop", "image_path"
            ]:
                f.write(f"{k}: {meta.get(k)}\n")

        exported.append(meta)
        picked += 1
        if picked >= max_cases:
            break

    return exported


def main():
    ap = argparse.ArgumentParser(description="Export effective CF case visuals from existing diagnostics.")
    ap.add_argument("--repo_root", default="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean")
    ap.add_argument("--out_dir", default="paper_figures/cf_effective_cases")
    ap.add_argument("--max_cases_per_source", type=int, default=12)
    args = ap.parse_args()

    repo = Path(args.repo_root)
    out_root = repo / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    sources = [
        {
            "benchmark": "textvqa",
            "variant": "routed",
            "diag": ".runtime_cache/test_config_smolvlm2_v91_cf3_routed_textvqa_n1000_r6/diagnostics/v91cf3_samples.jsonl",
            "xlsx": "benchmark_results/n_samples_1000/test_config_smolvlm2_v91_cf3_routed_textvqa/V91CF3Routed_SmolVLM2_2B/T20260430_G8433322c/V91CF3Routed_SmolVLM2_2B_TextVQA_VAL.xlsx",
        },
        {
            "benchmark": "ocrbench",
            "variant": "routed",
            "diag": ".runtime_cache/test_config_smolvlm2_v91_cf3_routed_ocrbench_n1000_noreuse/diagnostics/v91cf3_samples.jsonl",
            "xlsx": "benchmark_results/n_samples_1000/test_config_smolvlm2_v91_cf3_routed_ocrbench/V91CF3Routed_SmolVLM2_2B/T20260504_G8433322c/V91CF3Routed_SmolVLM2_2B_OCRBench.xlsx",
        },
        {
            "benchmark": "chartqa",
            "variant": "routed",
            "diag": ".runtime_cache/test_config_smolvlm2_v91_cf3_routed_chartqa_n1000_noreuse/diagnostics/v91cf3_samples.jsonl",
            "xlsx": "benchmark_results/n_samples_1000/test_config_smolvlm2_v91_cf3_routed_chartqa/V91CF3Routed_SmolVLM2_2B/T20260504_G8433322c/V91CF3Routed_SmolVLM2_2B_ChartQA_TEST.xlsx",
        },
        {
            "benchmark": "ocrvqa",
            "variant": "allcf_ungated",
            "diag": ".runtime_cache/test_config_smolvlm2_v91_cf3_allcf_ocrvqa_n1000_20260504_043132/diagnostics/v91cf3_samples.jsonl",
            "xlsx": "benchmark_results/n_samples_1000/test_config_smolvlm2_v91_cf3_allcf_ocrvqa/V91CF3AllCF_SmolVLM2_2B/T20260504_G8433322c/V91CF3AllCF_SmolVLM2_2B_OCRVQA_TEST.xlsx",
        },
        {
            "benchmark": "ocrbench",
            "variant": "allcf_ungated",
            "diag": ".runtime_cache/test_config_smolvlm2_v91_cf3_allcf_ocrbench_n1000_20260504_043132/diagnostics/v91cf3_samples.jsonl",
            "xlsx": "benchmark_results/n_samples_1000/test_config_smolvlm2_v91_cf3_allcf_ocrbench/V91CF3AllCF_SmolVLM2_2B/T20260504_G8433322c/V91CF3AllCF_SmolVLM2_2B_OCRBench.xlsx",
        },
        {
            "benchmark": "chartqa",
            "variant": "allcf_ungated",
            "diag": ".runtime_cache/test_config_smolvlm2_v91_cf3_allcf_chartqa_n1000_20260504_043132/diagnostics/v91cf3_samples.jsonl",
            "xlsx": "benchmark_results/n_samples_1000/test_config_smolvlm2_v91_cf3_allcf_chartqa/V91CF3AllCF_SmolVLM2_2B/T20260504_G8433322c/V91CF3AllCF_SmolVLM2_2B_ChartQA_TEST.xlsx",
        },
    ]

    all_rows = []
    for s in sources:
        all_rows.extend(
            process_source(
                benchmark=s["benchmark"],
                variant=s["variant"],
                diag_path=repo / s["diag"],
                xlsx_path=repo / s["xlsx"],
                out_root=out_root,
                max_cases=args.max_cases_per_source,
            )
        )

    fields = [
        "benchmark", "variant", "row_id", "index", "question", "answer", "prediction", "image_path",
        "cf_used", "cf_final_changed", "block_reason", "answer_space", "cf_margin",
        "no_cf_winner", "cf_winner", "final_answer", "picked_cf_candidate",
        "rel_box", "ctrl_box", "rel_drop", "ctrl_drop", "case_dir", "source_diag", "source_xlsx"
    ]

    with (out_root / "cf_effective_cases_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    by_group = {}
    for r in all_rows:
        k = f"{r['variant']}_{r['benchmark']}"
        by_group[k] = by_group.get(k, 0) + 1

    with (out_root / "README.txt").open("w") as f:
        f.write("CF effective case pack (existing diagnostics only).\n")
        f.write("Contains only cases with cf_used=True or cf_final_changed=True and available candidate-level rel/ctrl boxes.\n\n")
        f.write(f"total_exported: {len(all_rows)}\n")
        for k in sorted(by_group.keys()):
            f.write(f"{k}: {by_group[k]}\n")

    print(f"[DONE] {out_root}")
    print(f"[TOTAL_EXPORTED] {len(all_rows)}")
    for k in sorted(by_group.keys()):
        print(f"  {k}: {by_group[k]}")


if __name__ == "__main__":
    main()
