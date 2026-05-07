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
    "ocrbench": "OCRBench",
    "ai2d": "AI2D_TEST",
    "mme_rw": "MME-RealWorld-Lite",
    "amber": "AMBER",
    "coco": "COCO_VAL",
}

ALLOW_SPACES = {"ocr_text_short", "open_entity"}

WEIGHT_SETTINGS = [
    {"name": "wA", "w_sup": 2.0, "w_valid": 1.0, "w_base": 0.8, "w_risk": 0.75},
    {"name": "wB", "w_sup": 2.0, "w_valid": 1.5, "w_base": 0.8, "w_risk": 0.75},
    {"name": "wC", "w_sup": 1.5, "w_valid": 1.0, "w_base": 0.8, "w_risk": 0.75},
    {"name": "wD", "w_sup": 2.0, "w_valid": 1.0, "w_base": 1.0, "w_risk": 1.0},
]

GATE_SETTINGS = [
    {"name": "g1", "margin_tau": 0.30, "min_support": 0.25},
    {"name": "g2", "margin_tau": 0.60, "min_support": 0.25},
]

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


def format_validity(n_ans, feats, answer_space):
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


def length_risk(n_ans, feats, answer_space, base_len_words):
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
        feats[n] = {
            "raw": str(cand),
            "view_freq": float(support_map.get(n, 0.0)),
            "is_base": (n == base_norm),
            "is_numeric": is_numeric(n),
            "is_yes_no": is_yes_no(n),
            "is_alnum_short": is_alphanumeric_short(n),
            "length_words": answer_length(n),
        }
    return feats, answer_space, base_norm


def score_candidates(diag_obj, w):
    scored_top = diag_obj.get("scored_top", [])
    support_map = extract_support(scored_top)
    feats, answer_space, base_norm = build_features(diag_obj, support_map)
    if not feats:
        return [], answer_space
    base_len = feats.get(base_norm, {}).get("length_words", 0)

    scored = []
    for n_ans, f in feats.items():
        fv = format_validity(n_ans, feats, answer_space)
        lr = length_risk(n_ans, feats, answer_space, base_len)
        s = (
            float(w["w_sup"]) * float(f["view_freq"])
            + float(w["w_valid"]) * float(fv)
            + (float(w["w_base"]) if f["is_base"] else 0.0)
            - float(w["w_risk"]) * float(lr)
        )
        scored.append(
            {
                "norm": n_ans,
                "raw": f["raw"],
                "score": s,
                "support": float(f["view_freq"]),
                "fmt": float(fv),
                "is_base": bool(f["is_base"]),
                "len_words": int(f["length_words"]),
            }
        )
    scored.sort(
        key=lambda x: (x["score"], x["support"], x["fmt"], 1 if x["is_base"] else 0, -x["len_words"]),
        reverse=True,
    )
    return scored, answer_space


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


def discover_diag(repo, n, bench):
    p = repo / ".runtime_cache" / f"test_config_ovis2_2b_asca_{bench}_n{n}" / "diagnostics" / "asca_samples.jsonl"
    if p.exists():
        return p
    cands = sorted((repo / ".runtime_cache").glob(f"test_config_ovis2_2b_asca_{bench}_n{n}_diagregen_noreuse_*/diagnostics/asca_samples.jsonl"))
    return cands[-1] if cands else None


def write_prediction_file(src_rows, predictions, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = set()
    for r in src_rows:
        keys.update(r.keys())
    cols = list(keys)
    if "prediction" in cols:
        cols.remove("prediction")
    cols.append("prediction")
    rows = []
    for i, r in enumerate(src_rows, 1):
        sid = str(i)
        row = {k: r.get(k, "") for k in cols}
        row["prediction"] = predictions.get(sid, str(r.get("prediction", "")))
        rows.append(row)

    if out_path.suffix.lower() == ".xlsx":
        import pandas as pd
        pd.DataFrame(rows, columns=cols).to_excel(out_path, index=False)
        return

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Ovis tuned/gated rerank sweep with official eval (no new generation)")
    ap.add_argument("--repo_root", default="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--benchmarks", default="textvqa,ocrvqa,chartqa,gqa,ocrbench,ai2d,mme_rw,amber,coco")
    ap.add_argument("--output_dir", default="paper_neurips2026_artifacts/second_backbone/ovis2_tuned_gated_sweep")
    ap.add_argument("--allow_spaces", default="ocr_text_short", help="comma-separated answer_space allowlist")
    ap.add_argument("--weight_mode", choices=["grid", "strict_single"], default="strict_single")
    ap.add_argument("--margin_taus", default="0.6", help="comma-separated margin taus")
    ap.add_argument("--min_supports", default="0.75", help="comma-separated min supports")
    args = ap.parse_args()

    repo = Path(args.repo_root)
    outdir = repo / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    benches = [x.strip() for x in args.benchmarks.split(",") if x.strip()]

    ensure_unsloth_stub()
    build_dataset = import_build_dataset_lightweight()

    rows_out = []
    allow_spaces = {x.strip() for x in args.allow_spaces.split(",") if x.strip()}
    if not allow_spaces:
        allow_spaces = {"ocr_text_short"}

    if args.weight_mode == "strict_single":
        weight_settings = [{"name": "wStrict", "w_sup": 2.0, "w_valid": 1.0, "w_base": 1.0, "w_risk": 1.0}]
    else:
        weight_settings = WEIGHT_SETTINGS

    margin_taus = [float(x.strip()) for x in args.margin_taus.split(",") if x.strip()]
    min_supports = [float(x.strip()) for x in args.min_supports.split(",") if x.strip()]
    gate_settings = []
    gid = 1
    for m in margin_taus:
        for s in min_supports:
            gate_settings.append({"name": f"g{gid}", "margin_tau": m, "min_support": s})
            gid += 1
    for b in benches:
        dataset_tag = DATASET_TAG.get(b)
        if not dataset_tag:
            continue
        src_xlsx = discover_source_xlsx(repo, args.n, b)
        diag_path = discover_diag(repo, args.n, b)
        if src_xlsx is None or diag_path is None:
            print(f"[SKIP] {b} missing src_xlsx/diag")
            rows_out.append(
                {
                    "benchmark": b,
                    "dataset": dataset_tag,
                    "n": 0,
                    "setting": "NA",
                    "w_sup": "",
                    "w_valid": "",
                    "w_base": "",
                    "w_risk": "",
                    "margin_tau": "",
                    "min_support": "",
                    "full_score": "",
                    "tuned_score": "",
                    "delta_tuned_minus_full": "",
                    "gate_pass_count": "",
                    "switch_count": "",
                    "blocked_space_count": "",
                    "source_xlsx": str(src_xlsx) if src_xlsx is not None else "",
                    "source_diag": str(diag_path) if diag_path is not None else "",
                    "tuned_eval_input": "",
                }
            )
            continue

        src_rows = read_table_file(src_xlsx)
        diag = load_diag(diag_path)

        dataset = build_dataset(dataset_tag)
        if dataset is None:
            print(f"[SKIP] {b} build_dataset({dataset_tag})=None")
            continue
        full_ret = dataset.evaluate(str(src_xlsx))
        full_score = extract_official_score(full_ret, b)
        print(f"[BASE] {b} full_score={full_score}")

        for w in weight_settings:
            for g in gate_settings:
                setting_name = f"{w['name']}_{g['name']}"
                preds = {}
                switch_count = 0
                gate_pass = 0
                blocked_space = 0
                for i, r in enumerate(src_rows, 1):
                    sid = str(i)
                    src_pred = str(r.get("prediction", ""))
                    d = diag.get(sid)
                    if d is None:
                        preds[sid] = src_pred
                        continue
                    scored, answer_space = score_candidates(d, w)
                    if len(scored) == 0:
                        preds[sid] = src_pred
                        continue
                    top1 = scored[0]
                    top2_score = scored[1]["score"] if len(scored) > 1 else -1e9
                    margin = top1["score"] - top2_score
                    if answer_space not in allow_spaces:
                        blocked_space += 1
                        preds[sid] = src_pred
                        continue
                    if top1["support"] < g["min_support"] or margin < g["margin_tau"]:
                        preds[sid] = src_pred
                        continue
                    gate_pass += 1
                    tuned = top1["raw"]
                    if normalize_answer(tuned) == normalize_answer(src_pred):
                        tuned = src_pred
                    else:
                        switch_count += 1
                    preds[sid] = tuned

                src_suffix = src_xlsx.suffix.lower() if src_xlsx is not None else ".csv"
                if src_suffix not in (".csv", ".xlsx"):
                    src_suffix = ".csv"
                out_eval = outdir / "official_eval_inputs" / b / setting_name / f"{setting_name}_{dataset_tag}{src_suffix}"
                write_prediction_file(src_rows, preds, out_eval)
                tuned_ret = dataset.evaluate(str(out_eval))
                tuned_score = extract_official_score(tuned_ret, b)
                delta = None if (tuned_score is None or full_score is None) else tuned_score - full_score
                rows_out.append(
                    {
                        "benchmark": b,
                        "dataset": dataset_tag,
                        "n": len(src_rows),
                        "setting": setting_name,
                        "w_sup": w["w_sup"],
                        "w_valid": w["w_valid"],
                        "w_base": w["w_base"],
                        "w_risk": w["w_risk"],
                        "margin_tau": g["margin_tau"],
                        "min_support": g["min_support"],
                        "full_score": full_score,
                        "tuned_score": tuned_score,
                        "delta_tuned_minus_full": delta,
                        "gate_pass_count": gate_pass,
                        "switch_count": switch_count,
                        "blocked_space_count": blocked_space,
                        "source_xlsx": str(src_xlsx),
                        "source_diag": str(diag_path),
                        "tuned_eval_input": str(out_eval),
                    }
                )
                print(
                    f"[DONE] {b} {setting_name}: tuned={tuned_score} delta={delta} "
                    f"switch={switch_count} gate_pass={gate_pass} blocked={blocked_space}"
                )

    out_csv = outdir / "ovis_tuned_gated_sweep_summary.csv"
    with out_csv.open("w", newline="") as f:
        fn = [
            "benchmark", "dataset", "n", "setting", "w_sup", "w_valid", "w_base", "w_risk",
            "margin_tau", "min_support", "full_score", "tuned_score", "delta_tuned_minus_full",
            "gate_pass_count", "switch_count", "blocked_space_count",
            "source_xlsx", "source_diag", "tuned_eval_input",
        ]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(rows_out)
    print(f"[DONE] summary: {out_csv}")


if __name__ == "__main__":
    main()
