#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BENCHMARKS = [
    "ocrbench", "gqa", "textvqa", "chartqa", "ocrvqa", "ai2d", "mme_rw", "coco", "amber"
]

DATASET_BY_BENCH = {
    "ocrbench": "OCRBench",
    "gqa": "GQA_TestDev_Balanced",
    "textvqa": "TextVQA_VAL",
    "chartqa": "ChartQA_TEST",
    "ocrvqa": "OCRVQA_TEST",
    "ai2d": "AI2D_TEST",
    "mme_rw": "MME-RealWorld-Lite",
    "coco": "COCO_VAL",
    "amber": "AMBER",
}

CONFIG_BASELINE_1000 = {b: f"test_config_smolvlm2_paper_ttaug_classical_{b}" for b in BENCHMARKS}
CONFIG_NO_CF_1000 = {b: f"test_config_smolvlm2_v91_nocf_{b}" for b in BENCHMARKS}
CONFIG_ROUTED_1000 = {b: f"test_config_smolvlm2_v91_cf3_routed_{b}" for b in BENCHMARKS}

MODEL_BASELINE_1000 = "TTAugClassical_SmolVLM2_2B"
MODEL_BASELINE_200 = "TTAugClassical_SmolVLM2_2B_deterministic"
MODEL_NO_CF = "V91NoCF_SmolVLM2_2B"
MODEL_ROUTED = "V91CF3Routed_SmolVLM2_2B"
MODEL_FORCE_GRID = "V91CF3ForceGrid_SmolVLM2_2B"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def artifacts_root() -> Path:
    return repo_root() / "paper_neurips2026_artifacts"


def load_metric_index(index_path: Optional[Path] = None) -> List[dict]:
    if index_path is None:
        index_path = repo_root() / "logs" / "experiment_metric_index_20260429.json"
    with open(index_path, "r") as f:
        return json.load(f)


def find_row(rows: List[dict], *, n_samples: int, config: str, model: str, dataset: str) -> Optional[dict]:
    for r in rows:
        if (
            int(r.get("n_samples", -1)) == int(n_samples)
            and r.get("config") == config
            and r.get("model") == model
            and r.get("dataset") == dataset
        ):
            return r
    return None


def maybe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def format_num(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    if abs(x) >= 10:
        return f"{x:.2f}"
    if abs(x) >= 1:
        return f"{x:.3f}"
    return f"{x:.4f}"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]):
    ensure_dir(path.parent)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def to_latex_table(path: Path, headers: List[str], rows: List[List[str]], caption: str = "", label: str = ""):
    ensure_dir(path.parent)
    cols = "l" + "c" * (len(headers) - 1)
    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append(f"\\begin{{tabular}}{{{cols}}}")
    out.append("\\toprule")
    out.append(" & ".join(headers) + " \\\\")
    out.append("\\midrule")
    for r in rows:
        out.append(" & ".join(r) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    if caption:
        out.append(f"\\caption{{{caption}}}")
    if label:
        out.append(f"\\label{{{label}}}")
    out.append("\\end{table}")
    path.write_text("\n".join(out) + "\n")


def _xlsx_col_to_idx(col: str) -> int:
    n = 0
    for c in col:
        n = n * 26 + ord(c) - 64
    return n - 1


def read_xlsx_rows(path: Path) -> List[List[str]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        sst = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                texts = [t.text or "" for t in si.findall(".//a:t", ns)]
                sst.append("".join(texts))

        wb = ET.fromstring(z.read("xl/workbook.xml"))
        sheets = wb.find("a:sheets", ns)
        first_sheet = sheets.findall("a:sheet", ns)[0]
        rid = first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]

        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels:
            if rel.attrib.get("Id") == rid:
                target = rel.attrib["Target"]
                break
        if target is None:
            return []
        if target.startswith("/"):
            target = target[1:]
        sheet_path = "xl/" + target.split("xl/")[-1] if not target.startswith("xl/") else target

        root = ET.fromstring(z.read(sheet_path))
        data = []
        for row in root.findall(".//a:sheetData/a:row", ns):
            cells = {}
            for c in row.findall("a:c", ns):
                ref = c.attrib.get("r", "")
                m = re.match(r"([A-Z]+)(\d+)", ref)
                if not m:
                    continue
                idx = _xlsx_col_to_idx(m.group(1))
                t = c.attrib.get("t")
                v = c.find("a:v", ns)
                val = "" if v is None else (v.text or "")
                if t == "s" and val != "":
                    try:
                        val = sst[int(val)]
                    except Exception:
                        pass
                cells[idx] = str(val)
            if cells:
                mx = max(cells)
                rowvals = [""] * (mx + 1)
                for i, v in cells.items():
                    rowvals[i] = v
                data.append(rowvals)
        return data


def _choose_id_col(headers: List[str]) -> Optional[int]:
    keys = ["index", "id", "question_id", "image_id", "sample_id"]
    lowered = [h.lower().strip() for h in headers]
    for k in keys:
        if k in lowered:
            return lowered.index(k)
    return 0 if headers else None


def _choose_score_col(headers: List[str]) -> Optional[int]:
    keys = ["score", "correct", "acc", "hit", "em"]
    lowered = [h.lower().strip() for h in headers]
    for k in keys:
        if k in lowered:
            return lowered.index(k)
    return None


def load_sample_score_map(file_path: Path) -> Tuple[Optional[Dict[str, float]], str]:
    ext = file_path.suffix.lower()
    try:
        if ext == ".csv":
            with open(file_path, newline="") as f:
                rr = list(csv.reader(f))
            if len(rr) < 2:
                return None, "csv_no_rows"
            headers = rr[0]
            id_col = _choose_id_col(headers)
            score_col = _choose_score_col(headers)
            if score_col is None:
                return None, "csv_no_score_col"
            out = {}
            for r in rr[1:]:
                if id_col is None or id_col >= len(r) or score_col >= len(r):
                    continue
                sid = str(r[id_col])
                sc = maybe_float(r[score_col])
                if sc is not None:
                    out[sid] = sc
            return (out if out else None), "ok"
        if ext == ".jsonl":
            out = {}
            for ln in file_path.read_text().splitlines():
                if not ln.strip():
                    continue
                obj = json.loads(ln)
                sid = str(obj.get("index", obj.get("id", obj.get("question_id", obj.get("image_id", "")))))
                sc = maybe_float(obj.get("score", obj.get("correct", obj.get("acc"))))
                if sid and sc is not None:
                    out[sid] = sc
            return (out if out else None), "ok"
        if ext == ".xlsx":
            rows = read_xlsx_rows(file_path)
            if len(rows) < 2:
                return None, "xlsx_no_rows"
            headers = rows[0]
            id_col = _choose_id_col(headers)
            score_col = _choose_score_col(headers)
            if score_col is None:
                return None, "xlsx_no_score_col"
            out = {}
            for r in rows[1:]:
                if id_col is None or id_col >= len(r) or score_col >= len(r):
                    continue
                sid = str(r[id_col])
                sc = maybe_float(r[score_col])
                if sc is not None:
                    out[sid] = sc
            return (out if out else None), "ok"
    except Exception as e:
        return None, f"error:{type(e).__name__}"
    return None, "unsupported_ext"


def bootstrap_mean_ci(values: List[float], n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for __ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    point = sum(values) / n
    return point, lo, hi


def bootstrap_delta_ci(a: List[float], b: List[float], n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float, float]:
    # paired delta: a - b
    rng = random.Random(seed)
    n = min(len(a), len(b))
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    deltas = [a[i] - b[i] for i in range(n)]
    boots = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for __ in range(n)]
        boots.append(sum(deltas[i] for i in idxs) / n)
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    point = sum(deltas) / n
    p = 2 * min(
        sum(1 for x in boots if x <= 0) / n_boot,
        sum(1 for x in boots if x >= 0) / n_boot,
    )
    return point, lo, hi, p


def find_candidate_sample_files(result_dir: Path) -> List[Path]:
    # Prefer per-sample sheets over aggregate summary files.
    files = []
    for p in sorted(result_dir.glob("*")):
        if not p.is_file():
            continue
        n = p.name.lower()
        if n.endswith(".json") and "score" in n:
            continue
        if n.endswith("_acc.csv") or n.endswith("_score.csv"):
            continue
        if n.endswith("_rating.json") or n.endswith("_score.xlsx"):
            continue
        if p.suffix.lower() in (".csv", ".xlsx", ".jsonl"):
            files.append(p)
    return files


def write_markdown(path: Path, text: str):
    ensure_dir(path.parent)
    path.write_text(text)


def shquote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def parser_with_common(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--repo-root", default=str(repo_root()), help="Path to repo root")
    return p


def find_result_file_for_row(row: dict, benchmark_results_root: Path) -> Optional[Path]:
    """
    Resolve metric file path from a metric-index row.
    """
    if not row:
        return None
    n = row.get("n_samples")
    config = row.get("config")
    model = row.get("model")
    src = row.get("source")
    if n is None or not config or not model or not src:
        return None
    p = (
        Path(benchmark_results_root)
        / f"n_samples_{n}"
        / str(config)
        / str(model)
    )
    if not p.exists():
        return None
    candidate = p / str(src)
    if candidate.exists():
        return candidate
    # fallback: look for basename match
    src_name = Path(str(src)).name
    for q in p.glob("*"):
        if q.is_file() and q.name == src_name:
            return q
    return None


def read_table_file(path: Path) -> List[dict]:
    """
    Read csv/xlsx/json/jsonl into a list-of-dicts as best effort.
    """
    ext = path.suffix.lower()
    if ext == ".csv":
        with open(path, newline="") as f:
            rr = list(csv.DictReader(f))
        return rr
    if ext == ".json":
        obj = json.loads(path.read_text())
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            # common case: {"results":[...]}
            for k in ("results", "data", "rows"):
                v = obj.get(k)
                if isinstance(v, list):
                    return [x for x in v if isinstance(x, dict)]
            return [obj]
        return []
    if ext == ".jsonl":
        out = []
        for ln in path.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out
    if ext == ".xlsx":
        rows = read_xlsx_rows(path)
        if len(rows) < 1:
            return []
        headers = rows[0]
        out = []
        for r in rows[1:]:
            d = {}
            for i, h in enumerate(headers):
                key = str(h).strip() if h is not None else f"col_{i}"
                val = r[i] if i < len(r) else ""
                d[key] = val
            out.append(d)
        return out
    return []
