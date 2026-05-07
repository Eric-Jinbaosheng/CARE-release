#!/usr/bin/env python3
import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]


class FallbackImageAugment:
    """Lightweight augmentation fallback for login-node export."""

    def __init__(self, n_augmentations=9, aug_strength="medium"):
        self.n_augmentations = max(1, int(n_augmentations))

    def _aug(self, img: Image.Image, i: int) -> Image.Image:
        x = img.copy().convert("RGB")
        mode = i % 8
        if mode == 0:
            x = ImageEnhance.Brightness(x).enhance(0.85)
        elif mode == 1:
            x = ImageEnhance.Brightness(x).enhance(1.15)
        elif mode == 2:
            x = ImageEnhance.Contrast(x).enhance(0.85)
        elif mode == 3:
            x = ImageEnhance.Contrast(x).enhance(1.15)
        elif mode == 4:
            x = ImageEnhance.Color(x).enhance(0.85)
        elif mode == 5:
            x = ImageEnhance.Color(x).enhance(1.15)
        elif mode == 6:
            x = x.filter(ImageFilter.GaussianBlur(radius=1.4))
        elif mode == 7:
            x = ImageOps.autocontrast(x)
        return x

    def __call__(self, input_images):
        if isinstance(input_images, Image.Image):
            input_images = [input_images]
        augmented_images = [[] for _ in range(self.n_augmentations - 1)]
        for img in input_images:
            for i in range(self.n_augmentations - 1):
                augmented_images[i].append(self._aug(img, i))
        # match upstream shape: N aug views + original view
        augmented_images.append(input_images)
        return augmented_images, ["fallback"]


def load_image_augment_class(repo_root: Path):
    p = repo_root / "vlmeval" / "vlm" / "tta" / "image_augment.py"
    if not p.exists():
        return FallbackImageAugment
    try:
        spec = importlib.util.spec_from_file_location("local_image_augment", str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod.ImageAugment
    except Exception:
        return FallbackImageAugment


def pick_existing_image(row, repo_root: Path):
    cands = []
    ip = (row.get("image_path") or "").strip()
    if ip:
        cands.append(Path(ip))
    imf = (row.get("image_file") or "").strip()
    if imf:
        cands.append(repo_root / "paper_figures" / "cases_raw" / imf)
        cands.append(repo_root / "paper_figures" / "case_grid_materials" / "images" / imf)
        # rank-prefixed in case_grid_materials/images
        for p in (repo_root / "paper_figures" / "case_grid_materials" / "images").glob(f"*_{imf}"):
            cands.append(p)
    for p in cands:
        if p.exists() and p.is_file():
            return p
    return None


def parse_source_paths(source_file: str):
    out = {"nocf_diag": None, "nocf_xlsx": None, "cf_xlsx": None}
    if not source_file:
        return out
    parts = [x.strip() for x in source_file.split("|")]
    for p in parts:
        if p.endswith("v91nocf_samples.jsonl"):
            out["nocf_diag"] = p
        if "v91_nocf_regen" in p and p.endswith(".xlsx"):
            out["nocf_xlsx"] = p
        if ("v91_cf3" in p) and p.endswith(".xlsx"):
            out["cf_xlsx"] = p
    return out


def load_diag(diag_path: Path):
    m = {}
    if not diag_path or not diag_path.exists():
        return m
    with diag_path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except Exception:
                continue
            sid = str(o.get("sample_id", "")).strip()
            if sid:
                m[sid] = o
    return m


def read_xlsx_rows(xlsx_path: Path):
    import pandas as pd
    if not xlsx_path or not xlsx_path.exists():
        return []
    df = pd.read_excel(xlsx_path)
    return df.to_dict(orient="records")


def find_row_number_by_case(rows, row_case):
    sid = str(row_case.get("sample_id", "")).strip()
    q = (row_case.get("question") or "").strip().lower()
    # first try index column exact
    for i, r in enumerate(rows, 1):
        idx = str(r.get("index", "")).strip()
        if sid and idx and sid == idx:
            return i
    # fallback question match
    for i, r in enumerate(rows, 1):
        rq = str(r.get("question", "")).strip().lower()
        if q and rq == q:
            return i
    return None


def draw_cf_overlay(img: Image.Image, rel_box, ctrl_box):
    out = img.copy().convert("RGB")
    dr = ImageDraw.Draw(out)
    if rel_box and len(rel_box) == 4:
        dr.rectangle(rel_box, outline=(255, 0, 0), width=4)
    if ctrl_box and len(ctrl_box) == 4:
        dr.rectangle(ctrl_box, outline=(0, 200, 255), width=4)
    return out


def blur_box(img: Image.Image, box, radius=10):
    from PIL import ImageFilter
    out = img.copy().convert("RGB")
    if not box or len(box) != 4:
        return out
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(out.width, x1)); x2 = max(0, min(out.width, x2))
    y1 = max(0, min(out.height, y1)); y2 = max(0, min(out.height, y2))
    if x2 <= x1 or y2 <= y1:
        return out
    crop = out.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(radius=radius))
    out.paste(crop, (x1, y1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_csv", default="paper_figures/case_grid_materials/case_grid_candidates.csv")
    ap.add_argument("--out_dir", default="paper_figures/benchmark_aug_cf_pack")
    ap.add_argument("--per_benchmark", type=int, default=2)
    args = ap.parse_args()

    repo = REPO_ROOT
    case_csv = repo / args.case_csv
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    # select several cases: textvqa/ocrvqa/ocrbench priority
    rows = list(csv.DictReader(case_csv.open()))
    buckets = defaultdict(list)
    for r in rows:
        b = (r.get("benchmark") or "").strip().lower()
        buckets[b].append(r)

    selected = []
    for b in ["textvqa", "ocrvqa", "ocrbench", "chartqa"]:
        if b in buckets:
            selected.extend(buckets[b][: args.per_benchmark])

    # prepare routed diag for CF visuals (textvqa only likely)
    import glob
    cf_diag_path = None
    cands = sorted(glob.glob(str(repo / ".runtime_cache/test_config_smolvlm2_v91_cf3_routed_textvqa*/diagnostics/v91cf3_samples.jsonl")))
    if cands:
        # max lines
        best = None; best_n = -1
        for p in cands:
            n = sum(1 for ln in open(p) if ln.strip())
            if n > best_n:
                best_n = n; best = p
        cf_diag_path = Path(best)
    cf_diag = load_diag(cf_diag_path) if cf_diag_path else {}

    summary_rows = []
    cand_rows = []

    # cache xlsx/diag per path
    xlsx_cache = {}
    diag_cache = {}

    # image augmenter: 8 aug + original
    ImageAugmentClass = load_image_augment_class(repo)
    augmenter = ImageAugmentClass(n_augmentations=9, aug_strength="medium")

    for rank, r in enumerate(selected, 1):
        b = (r.get("benchmark") or "").strip().lower()
        sid = str(r.get("sample_id") or "").strip()
        case_id = f"{rank:02d}_{b}_{sid}"
        case_dir = out_dir / "images" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        src_img = pick_existing_image(r, repo)
        if src_img is None:
            summary_rows.append({"case_id": case_id, "status": "missing_image"})
            continue

        img = Image.open(src_img).convert("RGB")
        img.save(case_dir / "00_original.jpg", quality=95)

        # deterministic per case
        np.random.seed(abs(hash(case_id)) % (2**32))
        aug_sets, applied = augmenter(img)
        # aug_sets: len 9 -> first 8 augmented, last original list
        for i in range(min(8, len(aug_sets)-1)):
            aug_img = aug_sets[i][0] if isinstance(aug_sets[i], list) else aug_sets[i]
            aug_img.save(case_dir / f"aug_view{i+1:02d}.jpg", quality=95)

        # candidate info from nocf diag using row mapping
        sp = parse_source_paths(r.get("source_file", ""))
        row_num = None
        cand_list = []
        scored_top = []
        if sp["nocf_xlsx"] and sp["nocf_diag"]:
            nx = Path(sp["nocf_xlsx"])
            nd = Path(sp["nocf_diag"])
            if nx not in xlsx_cache:
                xlsx_cache[nx] = read_xlsx_rows(nx)
            if nd not in diag_cache:
                diag_cache[nd] = load_diag(nd)
            row_num = find_row_number_by_case(xlsx_cache[nx], r)
            if row_num is not None:
                d = diag_cache[nd].get(str(row_num), {})
                cand_list = d.get("candidate_list", []) or []
                scored_top = d.get("scored_top", []) or []

        # save candidate txt
        with (case_dir / "candidate_summary.txt").open("w") as f:
            f.write(f"benchmark: {b}\n")
            f.write(f"sample_id: {sid}\n")
            f.write(f"question: {r.get('question','')}\n")
            f.write(f"ground_truth: {r.get('ground_truth','')}\n")
            f.write(f"base_pred: {r.get('base_pred','')}\n")
            f.write(f"ttaug_pred: {r.get('ttaug_pred','')}\n")
            f.write(f"care_pred: {r.get('care_pred','')}\n")
            f.write(f"answer_pool_summary: {r.get('answer_pool_summary','')}\n")
            f.write(f"mapped_row_num: {row_num}\n")
            f.write("\n[candidate_list]\n")
            for c in cand_list:
                f.write(f"- {c}\n")
            f.write("\n[scored_top]\n")
            for x in scored_top[:10]:
                f.write(f"- {x}\n")

        # CF visuals (if textvqa and mapped row exists in routed cf diag)
        cf_ok = False
        if b == "textvqa" and row_num is not None and str(row_num) in cf_diag:
            dcf = cf_diag[str(row_num)]
            rel_box = dcf.get("rel_box")
            ctrl_box = dcf.get("ctrl_box")
            overlay = draw_cf_overlay(img, rel_box, ctrl_box)
            overlay.save(case_dir / "cf_boxes_overlay.jpg", quality=95)
            blur_box(img, rel_box, radius=10).save(case_dir / "cf_rel_drop_blur.jpg", quality=95)
            blur_box(img, ctrl_box, radius=10).save(case_dir / "cf_ctrl_drop_blur.jpg", quality=95)
            with (case_dir / "cf_summary.txt").open("w") as f:
                for k in ["answer_space","block_reason","cf_margin","cf_verifier_candidate","cf_verifier_general_gap","mask_quality","control_quality","cf_used","final_answer","no_cf_winner","cf_winner"]:
                    f.write(f"{k}: {dcf.get(k)}\n")
                f.write(f"rel_box: {rel_box}\nctrl_box: {ctrl_box}\n")
            cf_ok = True

        summary_rows.append({
            "case_id": case_id,
            "benchmark": b,
            "sample_id": sid,
            "src_image": str(src_img),
            "output_dir": str(case_dir),
            "mapped_row_num": row_num,
            "candidate_count": len(cand_list),
            "cf_visualized": int(cf_ok),
        })

        cand_rows.append({
            "case_id": case_id,
            "benchmark": b,
            "sample_id": sid,
            "question": r.get("question", ""),
            "ground_truth": r.get("ground_truth", ""),
            "base_pred": r.get("base_pred", ""),
            "ttaug_pred": r.get("ttaug_pred", ""),
            "care_pred": r.get("care_pred", ""),
            "answer_pool_summary": r.get("answer_pool_summary", ""),
            "mapped_row_num": row_num,
            "candidate_list": json.dumps(cand_list, ensure_ascii=False),
            "scored_top": json.dumps(scored_top[:10], ensure_ascii=False),
        })

    # write tables
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id","benchmark","sample_id","src_image","output_dir","mapped_row_num","candidate_count","cf_visualized"])
        w.writeheader(); w.writerows(summary_rows)

    with (out_dir / "candidate_and_cf_metadata.csv").open("w", newline="") as f:
        fields = ["case_id","benchmark","sample_id","question","ground_truth","base_pred","ttaug_pred","care_pred","answer_pool_summary","mapped_row_num","candidate_list","scored_top"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(cand_rows)

    # quick README
    with (out_dir / "README.txt").open("w") as f:
        f.write("Generated pack: original + 8 augmented views + candidate summary + CF visuals (if available).\n")
        f.write(f"CF diag source: {cf_diag_path}\n")
        f.write(f"num_cases: {len(summary_rows)}\n")

    print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    main()
