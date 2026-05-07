#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DATASET_PATTERNS: List[Tuple[str, str]] = [
    ("TextVQA_VAL", "TextVQA_VAL"),
    ("OCRVQA_TEST", "OCRVQA_TEST"),
    ("OCRBench", "OCRBench"),
    ("GQA_TestDev_Balanced", "GQA_TestDev_Balanced"),
    ("ChartQA_TEST", "ChartQA_TEST"),
    ("AI2D_TEST", "AI2D_TEST"),
    ("MME-RealWorld-Lite", "MME-RealWorld-Lite"),
    ("COCO_VAL", "COCO_VAL"),
    ("AMBER", "AMBER"),
]


def infer_dataset_from_name(name: str) -> Optional[str]:
    for pat, ds in DATASET_PATTERNS:
        if pat in name:
            return ds
    return None


def read_csv_score(path: Path) -> Tuple[Optional[str], Optional[float]]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return None, None
    headers = rows[0]
    vals = rows[1]
    # Prefer common metric columns.
    metric_col = None
    for cand in ["Overall", "Avg ACC", "Final Score Norm", "score", "acc"]:
        if cand in headers:
            metric_col = headers.index(cand)
            break
    if metric_col is None:
        for i, v in enumerate(vals):
            try:
                float(v)
                metric_col = i
                break
            except Exception:
                continue
    if metric_col is None:
        return None, None
    try:
        score = float(vals[metric_col])
    except Exception:
        return None, None
    metric = headers[metric_col] if metric_col < len(headers) else "Overall"
    return metric, score


def read_json_score(path: Path) -> Tuple[Optional[str], Optional[float]]:
    obj = json.loads(path.read_text())
    # Common priority keys in this repo.
    priority = [
        "Overall",
        "ROUGE_L",
        "Final Score Norm",
        "Final Score",
        "Avg ACC",
        "CIDEr",
    ]
    if isinstance(obj, dict):
        for k in priority:
            if k in obj:
                try:
                    return k, float(obj[k])
                except Exception:
                    continue
        for k, v in obj.items():
            try:
                return str(k), float(v)
            except Exception:
                continue
    return None, None


def parse_metric_file(path: Path) -> Tuple[Optional[str], Optional[float]]:
    name = path.name
    if name.endswith(".csv"):
        return read_csv_score(path)
    if name.endswith(".json"):
        return read_json_score(path)
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild experiment metric index from benchmark_results.")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out-stem", default="experiment_metric_index_20260429")
    args = ap.parse_args()

    root = Path(args.repo_root)
    bench_root = root / "benchmark_results"
    logs_root = root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []

    for ns_dir in sorted(bench_root.glob("n_samples_*")):
        m = re.match(r"n_samples_(\d+)$", ns_dir.name)
        if not m:
            continue
        n_samples = int(m.group(1))
        for cfg_dir in sorted([p for p in ns_dir.iterdir() if p.is_dir()]):
            config = cfg_dir.name
            for model_dir in sorted([p for p in cfg_dir.iterdir() if p.is_dir()]):
                model = model_dir.name
                metric_files = []
                metric_files.extend(model_dir.rglob("*_acc.csv"))
                metric_files.extend(model_dir.rglob("*_score.csv"))
                metric_files.extend(model_dir.rglob("*_score.json"))
                metric_files.extend(model_dir.rglob("*_rating.json"))
                metric_files.extend(model_dir.rglob("*_score.xlsx"))  # ignored parse, but keep source if needed
                if not metric_files:
                    continue

                # Keep best (latest mtime) metric file per dataset.
                by_dataset: Dict[str, Path] = {}
                for mf in metric_files:
                    if not mf.exists():
                        continue
                    ds = infer_dataset_from_name(mf.name)
                    if ds is None:
                        continue
                    prev = by_dataset.get(ds)
                    if prev is None:
                        by_dataset[ds] = mf
                        continue
                    try:
                        newer = mf.stat().st_mtime > prev.stat().st_mtime
                    except FileNotFoundError:
                        newer = True
                    if newer:
                        by_dataset[ds] = mf

                for ds, mf in by_dataset.items():
                    metric, score = parse_metric_file(mf)
                    # For xlsx-only cases we skip; scripts consuming this index rely on score present.
                    if score is None:
                        continue
                    rows.append(
                        {
                            "n_samples": n_samples,
                            "config": config,
                            "model": model,
                            "dataset": ds,
                            "metric": metric or "Overall",
                            "score": score,
                            "source": mf.name,
                        }
                    )

    rows.sort(key=lambda r: (int(r["n_samples"]), str(r["config"]), str(r["dataset"]), str(r["model"])))

    out_json = logs_root / f"{args.out_stem}.json"
    out_csv = logs_root / f"{args.out_stem}.csv"
    out_json.write_text(json.dumps(rows, indent=2))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_samples", "config", "model", "dataset", "metric", "score", "source"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(str(out_json))
    print(str(out_csv))
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
