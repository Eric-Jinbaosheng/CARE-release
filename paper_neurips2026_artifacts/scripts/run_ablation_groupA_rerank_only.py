#!/usr/bin/env python3
import argparse
import csv
import json
import importlib
import importlib.machinery
import re
import string
import sys
import types
from pathlib import Path
from typing import Dict, List, Tuple

from common import read_table_file

DEFAULT_BENCHMARKS = ["textvqa", "ocrvqa", "chartqa", "ocrbench", "gqa", "ai2d", "mme_rw", "amber", "coco"]
DATASET_TAG = {
    "textvqa": "TextVQA_VAL",
    "ocrvqa": "OCRVQA_TEST",
    "chartqa": "ChartQA_TEST",
    "ocrbench": "OCRBench",
    "gqa": "GQA_TestDev_Balanced",
    "ai2d": "AI2D_TEST",
    "mme_rw": "MME-RealWorld-Lite",
    "amber": "AMBER",
    "coco": "COCO_VAL",
}

SETTINGS = {
    "full_asca": dict(w_sup=2.0, w_valid=1.0, w_base=0.4, w_risk=0.5),
    "without_consistency_module": dict(w_sup=0.0, w_valid=1.0, w_base=0.0, w_risk=0.5),
    "without_answer_space_module": dict(w_sup=2.0, w_valid=0.0, w_base=0.4, w_risk=0.0),
}

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_MCQ_LETTER_RE = re.compile(r"^\(?([A-Ea-e])\)?(?:[\.:\)\s]|$)")
_YESNO_SET = {"yes", "no", "true", "false"}
_NUMERIC_RE = re.compile(
    r"^[\$£€¥]?[+-]?\d+(?:[\.,]\d+)*(?:%|°|usd|gb|kg|mg|cm|mm|km|kmh|hp|ml|l|kw|kwh)?\.?$",
    flags=re.IGNORECASE,
)
_DATE_RE = re.compile(r"\b(19|20)\d{2}\b|\b\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}\b")
_YES_NO = {"yes", "no", "true", "false", "yeah", "nope"}
_MCQ_FULL_RE = re.compile(r"^[abcdefgh][\.\):]?$", flags=re.IGNORECASE)
_OPTION_RE = re.compile(r"\b([A-H])[\.\):]\s+([^\n]+)")

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
    if _NUMERIC_RE.match(s):
        return True
    return bool(_DATE_RE.search(s))


def is_yes_no(s):
    return normalize_answer(s) in _YES_NO


def is_mcq_letter(s):
    s = (s or "").strip()
    return bool(_MCQ_FULL_RE.match(s))


def is_alphanumeric_short(s):
    n = normalize_answer(s)
    if not n or len(n) > 32:
        return False
    return bool(re.match(r"^[a-z0-9 \-_/\.]+$", n))


def answer_length(s):
    return len(normalize_answer(s).split())


def extract_options(question):
    if not question:
        return []
    options = []
    for m in _OPTION_RE.finditer(question):
        letter = m.group(1).upper()
        text = m.group(2).strip()
        text = re.split(r"\s+[A-H][\.\):]\s+", text)[0].strip()
        options.append((letter, text))
    return options


def _format_validity(n_ans, feats, answer_space):
    f = feats[n_ans]
    if answer_space == "yes_no":
        return 1.0 if f["is_yes_no"] else 0.0
    if answer_space == "multiple_choice":
        return 1.0 if (f["is_mcq_letter"] or f["from_options"]) else 0.2
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


def first_letter(pred):
    p = str(pred or "").strip()
    m = _MCQ_LETTER_RE.match(p)
    return m.group(1).upper() if m else None


def bool_from_pred(pred):
    p = norm_text(pred)
    if p in {"yes", "true"}:
        return True
    if p in {"no", "false"}:
        return False
    if p.startswith("yes"):
        return True
    if p.startswith("no"):
        return False
    return None


def is_correct(gt, pred, row):
    g = str(gt or "")
    p = str(pred or "")

    if "A" in row and "B" in row and len(g.strip()) == 1 and g.strip().upper() in "ABCDE":
        pl = first_letter(p)
        if pl is not None:
            return pl == g.strip().upper()

    ng = norm_text(g)
    if ng in _YESNO_SET:
        bp = bool_from_pred(p)
        if bp is not None:
            return bp == (ng in {"yes", "true"})

    if g.startswith("[") and g.endswith("]"):
        try:
            vals = json.loads(g.replace("'", '"'))
            if isinstance(vals, list):
                pn = norm_text(p)
                return pn in {norm_text(v) for v in vals}
        except Exception:
            pass

    return norm_text(g) == norm_text(p)


def discover_xlsx(repo_root: Path, n: int, cfg_stem: str, model_name: str, dataset_tag: str):
    d = repo_root / "benchmark_results" / f"n_samples_{n}" / cfg_stem / model_name
    if not d.exists():
        return None
    cands = [
        p for p in d.glob("*.xlsx")
        if not p.name.endswith("_score.xlsx") and "openai_result" not in p.name and "auxmatch" not in p.name
    ]
    if dataset_tag:
        cands = [p for p in cands if dataset_tag.lower() in p.name.lower()]
    if not cands:
        return None
    cands.sort()
    return cands[0]


def load_source_rows(xlsx_path: Path):
    try:
        import pandas as pd
        df = pd.read_excel(xlsx_path)
        rows = [{k: ("" if v is None else v) for k, v in r.items()} for r in df.to_dict(orient="records")]
        return df.columns.tolist(), rows
    except Exception:
        rows = read_table_file(xlsx_path)
        if not rows:
            return [], []
        headers = list(rows[0].keys())
        return headers, rows


def write_prediction_xlsx(src_headers: List[str], src_rows: List[dict], predictions: Dict[str, str], out_xlsx: Path):
    import pandas as pd

    table = []
    has_pred_col = "prediction" in src_headers
    cols = list(src_headers)
    if not has_pred_col:
        cols.append("prediction")

    for i, r in enumerate(src_rows, 1):
        sid = str(i)
        row = {k: r.get(k, "") for k in cols}
        row["prediction"] = predictions.get(sid, str(r.get("prediction", "")))
        table.append(row)
    df = pd.DataFrame(table, columns=cols)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_xlsx, index=False)


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
    # Handles dict / DataFrame outputs from dataset.evaluate
    b = benchmark.lower()
    if eval_results is None:
        return None

    # dict-like result (OCRBench and some others)
    if isinstance(eval_results, dict):
        if b == "ocrbench":
            for k in ["Final Score Norm", "Final Score", "Overall", "Avg ACC", "score", "acc"]:
                if k in eval_results:
                    return float(eval_results[k])
        for k in ["Overall", "Avg ACC", "score", "acc", "Final Score Norm", "Final Score"]:
            if k in eval_results:
                return float(eval_results[k])
        vals = _flatten_numeric_values(eval_results)
        return float(vals[0]) if vals else None

    # DataFrame-like result
    try:
        import pandas as pd
        if isinstance(eval_results, pd.DataFrame):
            df = eval_results.copy()
            # Most benchmarks have 'Overall' column on row 0
            if "Overall" in df.columns and len(df) > 0:
                try:
                    return float(df["Overall"].iloc[0])
                except Exception:
                    pass
            # AMBER often uses split='Overall' + Avg ACC
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
            # Fallback: first numeric cell
            for col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                s = s.dropna()
                if len(s) > 0:
                    return float(s.iloc[0])
    except Exception:
        pass

    return None


def fmt_float(x):
    if x is None:
        return "NA"
    return f"{float(x):.6f}"


def ensure_unsloth_stub():
    """Allow importing vlmeval.dataset in CPU-only jobs."""
    if "unsloth" in sys.modules:
        return
    m = types.ModuleType("unsloth")
    m.__spec__ = importlib.machinery.ModuleSpec("unsloth", loader=None)
    # trl may try to import these symbols; keep harmless placeholders.
    m.FastVisionModel = object
    m.FastLanguageModel = object
    sys.modules["unsloth"] = m


def import_build_dataset_lightweight():
    """
    Import vlmeval.dataset without executing vlmeval/__init__.py,
    which pulls in api/vlm/outlines and can break on CPU/login nodes.
    """
    if "vlmeval" not in sys.modules:
        pkg = types.ModuleType("vlmeval")
        pkg.__path__ = [str(REPO_ROOT / "vlmeval")]
        pkg.__package__ = "vlmeval"
        pkg.__spec__ = importlib.machinery.ModuleSpec("vlmeval", loader=None, is_package=True)
        sys.modules["vlmeval"] = pkg
    ds_mod = importlib.import_module("vlmeval.dataset")
    return ds_mod.build_dataset


def load_diag(path: Path) -> Dict[str, dict]:
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


def build_features(diag_obj, question, support_map):
    candidate_list = diag_obj.get("candidate_list") or []
    base_answer = diag_obj.get("base_answer", "")
    answer_space = diag_obj.get("answer_space", "unknown")
    n_views = int(diag_obj.get("n_views", 8) or 8)

    options = extract_options(question)
    base_norm = normalize_answer(base_answer)

    feats = {}
    for raw in candidate_list:
        n = normalize_answer(raw)
        if not n or n in feats:
            continue
        sup = support_map.get(n, 0.0)
        sup = float(max(0.0, min(1.0, sup)))
        view_count = int(round(sup * max(1, n_views)))
        feats[n] = {
            "raw": raw,
            "view_freq": sup,
            "view_count": view_count,
            "is_base": (n == base_norm),
            "is_numeric": is_numeric(n),
            "is_yes_no": is_yes_no(n),
            "is_mcq_letter": is_mcq_letter(raw),
            "is_alnum_short": is_alphanumeric_short(n),
            "length_words": answer_length(n),
            "from_options": any(normalize_answer(t) == n for _, t in options),
        }
    return feats, answer_space, base_norm


def rerank_with_weights(diag_obj, question, w):
    support_map = extract_support(diag_obj.get("scored_top"))
    feats, answer_space, base_norm = build_features(diag_obj, question, support_map)
    if not feats:
        return None

    base_len = feats.get(base_norm, {}).get("length_words", 0)
    scored = {}
    for n_ans, f in feats.items():
        fv = _format_validity(n_ans, feats, answer_space)
        lr = _length_risk(n_ans, feats, answer_space, base_len)
        s = (
            float(w["w_sup"]) * float(f["view_freq"]) +
            float(w["w_valid"]) * float(fv) +
            (float(w["w_base"]) if f["is_base"] else 0.0) -
            float(w["w_risk"]) * float(lr)
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


def main():
    ap = argparse.ArgumentParser(description="Group A score-module ablation rerank-only from regen cache.")
    ap.add_argument("--repo_root", default="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--regen_tag", default="regenA_20260503_161454")
    ap.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    ap.add_argument("--output_dir", default="paper_neurips2026_artifacts/ablations/groupA_groupB_from_regen")
    ap.add_argument("--official_eval", action="store_true", default=True, help="Use official dataset.evaluate scoring.")
    ap.add_argument("--official_eval_inputs_subdir", default="official_eval_inputs")
    args = ap.parse_args()

    repo = Path(args.repo_root)
    outdir = repo / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    benches = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    table_rows = []
    changed_rows = []

    for b in benches:
        dataset_tag = DATASET_TAG.get(b, "")
        if not dataset_tag:
            print(f"[SKIP] {b} missing DATASET_TAG mapping")
            continue
        cfg_stem = f"test_config_smolvlm2_v91_nocf_regen_{b}"
        xlsx = discover_xlsx(repo, args.n, cfg_stem, "V91NoCF_SmolVLM2_2B", dataset_tag)
        diag_path = repo / ".runtime_cache" / f"{cfg_stem}_n{args.n}_{args.regen_tag}" / "diagnostics" / "v91nocf_samples.jsonl"

        if xlsx is None or not diag_path.exists():
            print(f"[SKIP] {b} missing input: xlsx={xlsx} diag={diag_path.exists()}")
            continue

        src_headers, rows = load_source_rows(xlsx)
        diag = load_diag(diag_path)

        gt_map = {}
        q_map = {}
        for i, r in enumerate(rows, 1):
            sid = str(i)
            gt_map[sid] = str(r.get("answer", ""))
            q_map[sid] = str(r.get("question", ""))

        preds_by_setting = {k: {} for k in SETTINGS}

        for i, r in enumerate(rows, 1):
            sid = str(i)
            d = diag.get(sid)
            src_pred = str(r.get("prediction", ""))
            if d is None:
                # fallback: keep model prediction if diag missing
                raw_pred = src_pred
                for k in SETTINGS:
                    preds_by_setting[k][sid] = raw_pred
                continue
            for name, w in SETTINGS.items():
                if name == "full_asca":
                    # Keep exact source prediction for full setting, to remain bit-aligned with Table-1 run.
                    preds_by_setting[name][sid] = src_pred
                    continue
                p = rerank_with_weights(d, q_map[sid], w)
                if p is None:
                    p = src_pred
                # If normalized candidate equals source prediction, preserve original surface form
                # to avoid punctuation/casing artifacts in official evaluator.
                if normalize_answer(p) == normalize_answer(src_pred):
                    p = src_pred
                preds_by_setting[name][sid] = p

        full = preds_by_setting["full_asca"]

        # Compute official scores per setting via dataset.evaluate
        official_scores = {}
        official_paths = {}
        if args.official_eval:
            ensure_unsloth_stub()
            build_dataset = import_build_dataset_lightweight()
            dataset = build_dataset(dataset_tag)
            if dataset is None:
                print(f"[SKIP] {b} build_dataset({dataset_tag}) returned None")
                continue
            for name in ["full_asca", "without_consistency_module", "without_answer_space_module"]:
                cur = preds_by_setting[name]
                out_eval_xlsx = outdir / args.official_eval_inputs_subdir / b / name / f"{name}_{dataset_tag}.xlsx"
                write_prediction_xlsx(src_headers, rows, cur, out_eval_xlsx)
                eval_ret = dataset.evaluate(str(out_eval_xlsx))
                official_scores[name] = extract_official_score(eval_ret, b)
                official_paths[name] = str(out_eval_xlsx)

        for name in ["full_asca", "without_consistency_module", "without_answer_space_module"]:
            cur = preds_by_setting[name]
            n = len(gt_map)
            correct = 0
            changed = 0
            full_wins = 0
            ablation_wins = 0
            for i, r in enumerate(rows, 1):
                sid = str(i)
                fp = full[sid]
                cp = cur[sid]
                if normalize_answer(fp) != normalize_answer(cp):
                    changed += 1
                f_ok = is_correct(gt_map[sid], fp, r)
                c_ok = is_correct(gt_map[sid], cp, r)
                if c_ok:
                    correct += 1
                if f_ok and not c_ok:
                    full_wins += 1
                elif c_ok and not f_ok:
                    ablation_wins += 1

            proxy_score = correct / max(1, n)
            full_correct = sum(1 for i, r in enumerate(rows, 1) if is_correct(gt_map[str(i)], full[str(i)], r))
            full_proxy_score = full_correct / max(1, n)
            if args.official_eval:
                score = official_scores.get(name)
                full_score = official_scores.get("full_asca")
            else:
                score = proxy_score
                full_score = full_proxy_score
            delta = None if (score is None or full_score is None) else (score - full_score)
            net = ablation_wins - full_wins

            table_rows.append({
                "benchmark": b,
                "n": n,
                "setting": name,
                "w_sup": SETTINGS[name]["w_sup"],
                "w_valid": SETTINGS[name]["w_valid"],
                "w_base": SETTINGS[name]["w_base"],
                "w_risk": SETTINGS[name]["w_risk"],
                "correct_count_proxy": correct,
                "score": fmt_float(score),
                "full_score": fmt_float(full_score),
                "delta_vs_full": fmt_float(delta),
                "changed_count_vs_full": changed,
                "full_wins_proxy": full_wins,
                "ablation_wins_proxy": ablation_wins,
                "net_gain_proxy": net,
                "proxy_score": fmt_float(proxy_score),
                "full_proxy_score": fmt_float(full_proxy_score),
                "official_eval_used": int(args.official_eval),
                "official_eval_input": official_paths.get(name, ""),
                "source_xlsx": str(xlsx),
                "source_diag": str(diag_path),
            })

            if name != "full_asca":
                changed_rows.append({
                    "benchmark": b,
                    "setting": name,
                    "changed": changed,
                    "full_wins_proxy": full_wins,
                    "ablation_wins_proxy": ablation_wins,
                    "net_gain_proxy": net,
                })

    out_csv = outdir / "ablation_groupA_scores.csv"
    with out_csv.open("w", newline="") as f:
        if table_rows:
            w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
            w.writeheader()
            w.writerows(table_rows)
        else:
            f.write("benchmark,n,setting,w_sup,w_valid,w_base,w_risk,correct_count_proxy,score,full_score,delta_vs_full,changed_count_vs_full,full_wins_proxy,ablation_wins_proxy,net_gain_proxy,proxy_score,full_proxy_score,official_eval_used,official_eval_input,source_xlsx,source_diag\n")

    out_changed = outdir / "ablation_groupA_changed_cases.csv"
    with out_changed.open("w", newline="") as f:
        if changed_rows:
            w = csv.DictWriter(f, fieldnames=list(changed_rows[0].keys()))
            w.writeheader()
            w.writerows(changed_rows)
        else:
            f.write("benchmark,setting,changed,full_wins_proxy,ablation_wins_proxy,net_gain_proxy\n")

    print(f"[DONE] GroupA scores: {out_csv}")
    print(f"[DONE] GroupA changed: {out_changed}")


if __name__ == "__main__":
    main()
