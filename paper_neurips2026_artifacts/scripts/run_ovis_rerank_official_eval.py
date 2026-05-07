#!/usr/bin/env python3
import argparse
import csv
import importlib
import importlib.machinery
import json
import re
import string
import sys
import types
from pathlib import Path

from common import read_table_file

DATASET_TAG = {
    "textvqa": "TextVQA_VAL",
    "ocrvqa": "OCRVQA_TEST",
    "chartqa": "ChartQA_TEST",
    "gqa": "GQA_TestDev_Balanced",
}

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_NUMERIC_RE = re.compile(
    r"^[\$£€¥]?[+-]?\d+(?:[\.,]\d+)*(?:%|°|usd|gb|kg|mg|cm|mm|km|kmh|hp|ml|l|kw|kwh)?\.?$",
    flags=re.IGNORECASE,
)
_YES_NO = {"yes", "no", "true", "false", "yeah", "nope"}

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def norm_text(s):
    return str(s or "").strip().lower().translate(_PUNCT_TABLE).strip()


def normalize_answer(s):
    return norm_text(s)


def is_numeric(s):
    s = normalize_answer(s)
    if not s:
        return False
    return bool(_NUMERIC_RE.match(s))


def is_yes_no(s):
    return normalize_answer(s) in _YES_NO


def is_alphanumeric_short(s):
    n = normalize_answer(s)
    if not n or len(n) > 32:
        return False
    return bool(re.match(r"^[a-z0-9 \-_/\.]+$", n))


def answer_length(s):
    return len(normalize_answer(s).split())


def ensure_unsloth_stub():
    if "unsloth" in sys.modules:
        return
    m = types.ModuleType("unsloth")
    m.__spec__ = importlib.machinery.ModuleSpec("unsloth", loader=None)
    m.FastVisionModel = object
    m.FastLanguageModel = object
    sys.modules["unsloth"] = m


def import_build_dataset_lightweight():
    if "vlmeval" not in sys.modules:
        pkg = types.ModuleType("vlmeval")
        pkg.__path__ = [str(REPO_ROOT / "vlmeval")]
        pkg.__package__ = "vlmeval"
        pkg.__spec__ = importlib.machinery.ModuleSpec("vlmeval", loader=None, is_package=True)
        sys.modules["vlmeval"] = pkg
    ds_mod = importlib.import_module("vlmeval.dataset")
    return ds_mod.build_dataset


def _flatten_numeric_values(obj):
    vals = []
    if isinstance(obj, (int, float)):
        vals.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            vals.extend(_flatten_numeric_values(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            vals.extend(_flatten_numeric_values(v))
    return vals


def extract_official_score(eval_results, benchmark: str):
    b = benchmark.lower()
    if eval_results is None:
        return None

    if isinstance(eval_results, dict):
        if b == "ocrbench":
            for k in ["Final Score Norm", "Final Score", "Overall", "Avg ACC", "score", "acc"]:
                if k in eval_results:
                    return float(eval_results[k])
        if b == "coco":
            for k in ["ROUGE_L", "CIDEr", "Bleu_4", "Bleu_1"]:
                if k in eval_results:
                    return float(eval_results[k])
        for k in ["Overall", "Avg ACC", "score", "acc", "Final Score Norm", "Final Score"]:
            if k in eval_results:
                return float(eval_results[k])
        vals = _flatten_numeric_values(eval_results)
        return float(vals[0]) if vals else None

    try:
        import pandas as pd
        if isinstance(eval_results, pd.DataFrame):
            df = eval_results.copy()
            if "Overall" in df.columns and len(df) > 0:
                try:
                    return float(df["Overall"].iloc[0])
                except Exception:
                    pass
            if "split" in df.columns:
                key_cols = [c for c in ["Avg ACC", "Overall", "acc", "score"] if c in df.columns]
                if key_cols:
                    sub = df[df["split"].astype(str).str.lower() == "overall"]
                    if len(sub) > 0:
                        for c in key_cols:
                            try:
                                return float(sub[c].iloc[0])
                            except Exception:
                                continue
                    for c in key_cols:
                        try:
                            return float(df[c].iloc[0])
                        except Exception:
                            continue
            for col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s) > 0:
                    return float(s.iloc[0])
    except Exception:
        pass

    return None


def load_diag(path: Path):
    out = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sid = str(obj.get("sample_id", ""))
            if sid:
                out[sid] = obj
    return out


def extract_support(scored_top):
    out = {}
    if not isinstance(scored_top, list):
        return out
    for entry in scored_top:
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        cand = str(entry[0])
        n = normalize_answer(cand)
        if not n:
            continue
        sup = None
        if len(entry) >= 5:
            sup = safe_float(entry[4])
        if sup is None:
            sup = safe_float(entry[1])
        if sup is None:
            continue
        out[n] = sup
    return out


def _format_validity(n_ans, feats, answer_space):
    f = feats[n_ans]
    if answer_space == "yes_no":
        return 1.0 if f["is_yes_no"] else 0.0
    if answer_space == "numeric":
        return 1.0 if f["is_numeric"] else 0.4
    if answer_space == "ocr_text_short":
        return 1.0 if f["is_alnum_short"] else 0.3
    if answer_space == "open_entity":
        return 1.0 if f["length_words"] <= 5 else 0.6
    if answer_space == "chart_or_diagram":
        return 1.0 if (f["is_numeric"] or f["length_words"] <= 6) else 0.5
    if answer_space == "caption_like":
        return 1.0 if f["length_words"] >= 4 else 0.3
    return 1.0


def _length_risk(n_ans, feats, answer_space, base_len_words):
    f = feats[n_ans]
    if answer_space in ("caption_like",):
        return 0.0
    if answer_space in ("yes_no", "multiple_choice", "numeric"):
        return 0.5 if f["length_words"] > 4 else 0.0
    if answer_space in ("ocr_text_short", "open_entity"):
        if base_len_words and f["length_words"] > base_len_words * 2:
            return 0.5
    return 0.0


def build_features(diag_obj, support_map):
    candidate_list = diag_obj.get("candidate_list") or []
    base_answer = diag_obj.get("base_answer", "")
    answer_space = diag_obj.get("answer_space", "unknown")

    base_norm = normalize_answer(base_answer)
    feats = {}
    for cand in candidate_list:
        n = normalize_answer(cand)
        if not n or n in feats:
            continue
        sup = float(support_map.get(n, 0.0))
        feats[n] = {
            "raw": str(cand),
            "view_freq": sup,
            "is_base": (n == base_norm),
            "is_numeric": is_numeric(n),
            "is_yes_no": is_yes_no(n),
            "is_alnum_short": is_alphanumeric_short(n),
            "length_words": answer_length(n),
        }
    return feats, answer_space, base_norm


def rerank_with_weights(diag_obj, w):
    scored_top = diag_obj.get("scored_top", [])
    support_map = extract_support(scored_top)
    feats, answer_space, base_norm = build_features(diag_obj, support_map)
    if not feats:
        return None

    base_len = feats.get(base_norm, {}).get("length_words", 0)
    scored = {}
    for n_ans, f in feats.items():
        fv = _format_validity(n_ans, feats, answer_space)
        lr = _length_risk(n_ans, feats, answer_space, base_len)
        s = (
            float(w["w_sup"]) * float(f["view_freq"])
            + float(w["w_valid"]) * float(fv)
            + (float(w["w_base"]) if f["is_base"] else 0.0)
            - float(w["w_risk"]) * float(lr)
        )
        scored[n_ans] = {
            "score": s,
            "view_freq": float(f["view_freq"]),
            "fmt_validity": float(fv),
            "is_base": bool(f["is_base"]),
            "length_words": int(f["length_words"]),
            "raw": f["raw"],
        }

    best = sorted(
        scored.items(),
        key=lambda kv: (
            kv[1]["score"], kv[1]["view_freq"], kv[1]["fmt_validity"],
            1 if kv[1]["is_base"] else 0, -kv[1]["length_words"],
        ),
        reverse=True,
    )[0][0]
    return scored[best]["raw"]


def discover_source_xlsx(repo, n, bench):
    d = repo / "benchmark_results" / f"n_samples_{n}" / f"test_config_ovis2_2b_asca_{bench}" / "ASCA_Ovis2_2B"
    if not d.exists():
        return None
    cands = [
        p for p in d.glob("**/*.xlsx")
        if not p.name.endswith("_score.xlsx") and "openai_result" not in p.name and "auxmatch" not in p.name
    ]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime)
    return cands[-1]


def write_prediction_csv(src_rows, predictions, out_csv):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # union all columns + prediction
    keys = set()
    for r in src_rows:
        keys.update(r.keys())
    cols = list(keys)
    if "prediction" in cols:
        cols.remove("prediction")
    cols.append("prediction")
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(src_rows, 1):
            sid = str(i)
            row = {k: r.get(k, "") for k in cols}
            row["prediction"] = predictions.get(sid, str(r.get("prediction", "")))
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Ovis rerank with existing generation cache + official evaluator")
    ap.add_argument("--repo_root", default="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--benchmarks", default="textvqa,ocrvqa,chartqa,gqa")
    ap.add_argument("--w_sup", type=float, default=2.0)
    ap.add_argument("--w_valid", type=float, default=0.25)
    ap.add_argument("--w_base", type=float, default=0.4)
    ap.add_argument("--w_risk", type=float, default=0.5)
    ap.add_argument("--output_dir", default="paper_neurips2026_artifacts/second_backbone/ovis2_rerank_wvalid025_official")
    args = ap.parse_args()

    repo = Path(args.repo_root)
    outdir = repo / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    ensure_unsloth_stub()
    build_dataset = import_build_dataset_lightweight()

    w = {
        "w_sup": args.w_sup,
        "w_valid": args.w_valid,
        "w_base": args.w_base,
        "w_risk": args.w_risk,
    }

    rows_out = []
    benches = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    for b in benches:
        dataset_tag = DATASET_TAG.get(b)
        if not dataset_tag:
            print(f"[SKIP] unknown benchmark {b}")
            continue

        src_xlsx = discover_source_xlsx(repo, args.n, b)
        if src_xlsx is None:
            print(f"[SKIP] {b} missing source xlsx")
            continue

        diag_path = repo / ".runtime_cache" / f"test_config_ovis2_2b_asca_{b}_n{args.n}" / "diagnostics" / "asca_samples.jsonl"
        if not diag_path.exists():
            # fallback to latest diagregen_noreuse
            cands = sorted((repo / ".runtime_cache").glob(f"test_config_ovis2_2b_asca_{b}_n{args.n}_diagregen_noreuse_*/diagnostics/asca_samples.jsonl"))
            if cands:
                diag_path = cands[-1]
        if not diag_path.exists():
            print(f"[SKIP] {b} missing diagnostics")
            continue

        src_rows = read_table_file(src_xlsx)
        diag = load_diag(diag_path)

        preds = {}
        changed = 0
        for i, r in enumerate(src_rows, 1):
            sid = str(i)
            src_pred = str(r.get("prediction", ""))
            d = diag.get(sid)
            if d is None:
                preds[sid] = src_pred
                continue
            p = rerank_with_weights(d, w)
            if p is None:
                p = src_pred
            if normalize_answer(p) == normalize_answer(src_pred):
                p = src_pred
            preds[sid] = p
            if normalize_answer(p) != normalize_answer(src_pred):
                changed += 1

        out_csv = outdir / "official_eval_inputs" / b / f"asca_wvalid{str(args.w_valid).replace('.', 'p')}_{dataset_tag}.csv"
        write_prediction_csv(src_rows, preds, out_csv)

        dataset = build_dataset(dataset_tag)
        if dataset is None:
            print(f"[SKIP] build_dataset({dataset_tag}) returned None")
            continue

        # official eval for tuned rerank
        tuned_ret = dataset.evaluate(str(out_csv))
        tuned_score = extract_official_score(tuned_ret, b)

        # full/original official score from source xlsx
        full_ret = dataset.evaluate(str(src_xlsx))
        full_score = extract_official_score(full_ret, b)

        delta = None if (tuned_score is None or full_score is None) else tuned_score - full_score
        rows_out.append({
            "benchmark": b,
            "dataset": dataset_tag,
            "n": len(src_rows),
            "w_sup": args.w_sup,
            "w_valid": args.w_valid,
            "w_base": args.w_base,
            "w_risk": args.w_risk,
            "full_score": full_score,
            "tuned_score": tuned_score,
            "delta_tuned_minus_full": delta,
            "changed_vs_full": changed,
            "source_xlsx": str(src_xlsx),
            "source_diag": str(diag_path),
            "tuned_eval_input": str(out_csv),
        })
        print(f"[DONE] {b}: full={full_score} tuned={tuned_score} delta={delta} changed={changed}")

    out_sum = outdir / "ovis_rerank_official_summary.csv"
    with out_sum.open("w", newline="") as f:
        fn = [
            "benchmark", "dataset", "n", "w_sup", "w_valid", "w_base", "w_risk",
            "full_score", "tuned_score", "delta_tuned_minus_full", "changed_vs_full",
            "source_xlsx", "source_diag", "tuned_eval_input",
        ]
        wcsv = csv.DictWriter(f, fieldnames=fn)
        wcsv.writeheader()
        wcsv.writerows(rows_out)
    print(f"[DONE] summary: {out_sum}")


if __name__ == "__main__":
    main()
