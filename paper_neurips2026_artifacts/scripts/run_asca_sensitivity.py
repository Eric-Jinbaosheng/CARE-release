#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import math
import re
import string
import sys
import types
import importlib
import importlib.machinery
from collections import defaultdict
from pathlib import Path

from common import read_table_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BENCHMARKS = ["textvqa", "ocrvqa", "chartqa", "ocrbench", "gqa"]
DATASET_TAG = {
    "textvqa": "TextVQA_VAL",
    "ocrvqa": "OCRVQA_TEST",
    "chartqa": "ChartQA_TEST",
    "ocrbench": "OCRBench",
    "gqa": "GQA_TestDev_Balanced",
}

DEFAULT = dict(w_sup=2.0, w_valid=1.0, w_base=0.4, w_risk=0.5)
WSUP_POINTS = [1.0, 1.5, 2.0, 2.5, 3.0]
WVALID_POINTS = [0.25, 0.5, 1.0, 1.5, 2.0]

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
            vals = ast.literal_eval(g)
            if isinstance(vals, list):
                pn = norm_text(p)
                return pn in {norm_text(v) for v in vals}
        except Exception:
            pass

    return norm_text(g) == norm_text(p)


def setting_name(w):
    def f(x):
        return str(x).replace("-", "m").replace(".", "p")
    return f"wsup{f(w['w_sup'])}_wvalid{f(w['w_valid'])}_wbase{f(w['w_base'])}_wrisk{f(w['w_risk'])}"


def discover_xlsx(repo_root: Path, n: int, cfg: str, model: str, dataset_tag: str):
    d = repo_root / "benchmark_results" / f"n_samples_{n}" / cfg / model
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


def discover_diag(repo_root: Path, cfg_stem: str):
    p = repo_root / ".runtime_cache" / cfg_stem / "diagnostics" / "v91nocf_samples.jsonl"
    return p if p.exists() else None


def discover_diag_fuzzy(repo_root: Path, cfg_stem_prefix: str):
    # Match exact stem or stem with suffixes (e.g., *_regenA_YYYYMMDD_*).
    pats = list((repo_root / ".runtime_cache").glob(f"{cfg_stem_prefix}*/diagnostics/v91nocf_samples.jsonl"))
    if not pats:
        return None
    best = None
    best_n = -1
    for p in pats:
        n = 0
        try:
            with p.open() as f:
                for ln in f:
                    if ln.strip():
                        n += 1
        except Exception:
            n = -1
        if n > best_n:
            best_n = n
            best = p
    return best


def discover_best_default_xlsx(repo_root: Path, n: int, bench: str, dataset_tag: str):
    # Keep sensitivity aligned with the same default branch used by routed-CF comparisons.
    cfg_order = [
        f"test_config_smolvlm2_v91_nocf_regen_{bench}",
        f"test_config_smolvlm2_v91_nocf_{bench}",
    ]
    for cfg in cfg_order:
        p = discover_xlsx(repo_root, n, cfg, "V91NoCF_SmolVLM2_2B", dataset_tag)
        if p is not None:
            return p, cfg
    return None, ""


def discover_routed_xlsx(repo_root: Path, n: int, bench: str, dataset_tag: str):
    return discover_xlsx(repo_root, n, f"test_config_smolvlm2_v91_cf3_routed_{bench}", "V91CF3Routed_SmolVLM2_2B", dataset_tag)


def discover_best_main_diag(repo_root: Path, n: int, bench: str):
    # Priority: same-branch noCF diagnostics, then ablation diagnostics fallback.
    for stem in [f"test_config_smolvlm2_v91_nocf_regen_{bench}_n{n}", f"test_config_smolvlm2_v91_nocf_{bench}_n{n}"]:
        p = discover_diag(repo_root, stem) or discover_diag_fuzzy(repo_root, stem)
        if p is not None:
            return p, stem
    for v in ["no_format", "no_length_risk", "frequency_only", "majority_vote", "no_base_bias"]:
        stem = f"test_config_smolvlm2_v91_nocf_ablation_{v}_{bench}_n{n}"
        p = discover_diag(repo_root, stem) or discover_diag_fuzzy(repo_root, stem)
        if p is not None:
            return p, stem
    return None, ""


def discover_metric_file(repo_root: Path, n: int, cfg: str, model: str, dataset_tag: str):
    d = repo_root / "benchmark_results" / f"n_samples_{n}" / cfg / model
    if not d.exists():
        return None
    pats = [
        f"*{dataset_tag}*_acc.csv",
        f"*{dataset_tag}*_score.csv",
        f"*{dataset_tag}*_score.json",
        f"*{dataset_tag}*_score.xlsx",
    ]
    cands = []
    for pat in pats:
        cands.extend(d.glob(pat))
    if not cands:
        return None
    cands.sort()
    return cands[0]


def read_official_score(path: Path):
    if path is None or not path.exists():
        return None
    try:
        if path.suffix.lower() == ".csv":
            with path.open(newline="") as f:
                for row in csv.reader(f):
                    for cell in row:
                        v = safe_float(cell)
                        if v is not None:
                            return float(v)
        if path.suffix.lower() == ".json":
            obj = json.loads(path.read_text())
            if isinstance(obj, dict):
                for k in ["Overall", "Avg ACC", "score", "acc", "Final Score", "Final Score Norm"]:
                    if k in obj and safe_float(obj[k]) is not None:
                        return float(obj[k])
                for v in obj.values():
                    fv = safe_float(v)
                    if fv is not None:
                        return float(fv)
        if path.suffix.lower() == ".xlsx":
            rows = read_table_file(path)
            for r in rows:
                for v in r.values():
                    fv = safe_float(v)
                    if fv is not None:
                        return float(fv)
    except Exception:
        return None
    return None


def load_diag_map(path: Path):
    out = {}
    if path is None or not path.exists():
        return out
    with path.open() as f:
        for ln in f:
            if not ln.strip():
                continue
            try:
                o = json.loads(ln)
            except Exception:
                continue
            sid = str(o.get("sample_id", ""))
            if sid:
                out[sid] = o
    return out


def row_id(row, idx):
    for k in ["index", "id", "question_id", "image_id", "sample_id"]:
        if k in row and str(row[k]).strip() != "":
            return str(row[k]).strip()
    return str(idx)


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


def build_features(sample_diag, question, support_map):
    candidate_list = sample_diag.get("candidate_list") or []
    base_answer = sample_diag.get("base_answer", "")
    answer_space = sample_diag.get("answer_space", "unknown")
    n_views = int(sample_diag.get("n_views", 8) or 8)

    options = extract_options(question)
    base_norm = normalize_answer(base_answer)

    feats = {}
    for raw in candidate_list:
        n = normalize_answer(raw)
        if not n:
            continue
        if n in feats:
            feats[n]["all_raw"].append(raw)
            continue
        sup = support_map.get(n, 0.0)
        sup = float(max(0.0, min(1.0, sup)))
        view_count = int(round(sup * max(1, n_views)))
        feats[n] = {
            "raw": raw,
            "all_raw": [raw],
            "view_freq": sup,
            "view_count": view_count,
            "is_base": (n == base_norm),
            "is_numeric": is_numeric(n),
            "is_yes_no": is_yes_no(n),
            "is_mcq_letter": is_mcq_letter(raw),
            "is_alnum_short": is_alphanumeric_short(n),
            "length_words": answer_length(n),
            "from_options": any(normalize_answer(t) == n for _, t in options),
            "is_substring_of_base": (base_norm and n != base_norm and n in base_norm),
            "is_superstring_of_base": (base_norm and n != base_norm and base_norm in n),
        }

    return feats, answer_space, base_norm


def choose_output_raw(n_ans, feats, default_pred_raw=""):
    # Keep official-eval-compatible formatting when possible.
    if not n_ans or n_ans not in feats:
        return default_pred_raw or ""
    bucket = feats[n_ans].get("all_raw") or [feats[n_ans].get("raw", n_ans)]
    dnorm = normalize_answer(default_pred_raw)
    if dnorm and dnorm == n_ans and default_pred_raw:
        return default_pred_raw
    # Prefer shortest raw for deterministic formatting (often cleaner numeric/text span).
    bucket_sorted = sorted(bucket, key=lambda s: (len(str(s)), str(s)))
    return str(bucket_sorted[0])


def rerank_one(sample_diag, question, support_map, w, default_pred_raw=""):
    feats, answer_space, base_norm = build_features(sample_diag, question, support_map)
    if not feats:
        return None, {}, answer_space

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
            "length_risk": float(lr),
            "is_base": bool(f["is_base"]),
            "raw": f["raw"],
        }

    # deterministic tie-break: score, support, fmt, is_base, then shorter answer
    best = sorted(
        scored.items(),
        key=lambda kv: (
            kv[1]["score"], kv[1]["view_freq"], kv[1]["fmt_validity"], 1 if kv[1]["is_base"] else 0, -answer_length(kv[0])
        ),
        reverse=True,
    )[0][0]
    out_raw = choose_output_raw(best, feats, default_pred_raw=default_pred_raw)
    return out_raw, scored, answer_space


def quick_settings():
    out = []
    seen = set()

    def add(w, varied, val):
        name = setting_name(w)
        if name in seen:
            return
        seen.add(name)
        out.append({
            "setting_name": name,
            "varied_param": varied,
            "param_value": val,
            **w,
        })

    add(dict(DEFAULT), "default", "default")
    for x in WSUP_POINTS:
        w = dict(DEFAULT)
        w["w_sup"] = x
        add(w, "w_sup", x)
    for x in WVALID_POINTS:
        w = dict(DEFAULT)
        w["w_valid"] = x
        add(w, "w_valid", x)
    return out


def custom_settings(args):
    w = dict(
        w_sup=float(args.w_sup),
        w_valid=float(args.w_valid),
        w_base=float(args.w_base),
        w_risk=float(args.w_risk),
    )
    return [{
        "setting_name": setting_name(w),
        "varied_param": args.varied_param or "custom",
        "param_value": args.param_value if args.param_value is not None else "custom",
        **w,
    }]


def rank_map(values):
    # higher is better; average rank for ties
    pairs = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    ranks = {}
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][1] == pairs[i][1]:
            j += 1
        r = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[pairs[k][0]] = r
        i = j + 1
    return ranks


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


def write_prediction_xlsx(rows, predictions, out_xlsx: Path):
    import pandas as pd
    if not rows:
        return
    if isinstance(rows, dict):
        first = next(iter(rows.values()))
        row_items = list(rows.items())
    else:
        first = rows[0]
        row_items = [(str(i + 1), r) for i, r in enumerate(rows)]
    cols = list(first.keys())
    if "prediction" not in cols:
        cols.append("prediction")
    table = []
    for sid, r in row_items:
        row = {k: r.get(k, "") for k in cols}
        row["prediction"] = predictions.get(sid, str(r.get("prediction", "")))
        table.append(row)
    df = pd.DataFrame(table, columns=cols)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_xlsx, index=False)


def extract_official_score(eval_results):
    if eval_results is None:
        return None
    try:
        import pandas as pd
        if isinstance(eval_results, pd.DataFrame):
            df = eval_results
            if "Overall" in df.columns and len(df) > 0:
                return float(df["Overall"].iloc[0])
            if "split" in df.columns:
                for col in ["Avg ACC", "Overall", "score", "acc"]:
                    if col in df.columns:
                        sub = df[df["split"].astype(str).str.lower() == "overall"]
                        if len(sub) > 0:
                            return float(sub[col].iloc[0])
                        return float(df[col].iloc[0])
    except Exception:
        pass
    if isinstance(eval_results, dict):
        for k in ["Overall", "Avg ACC", "score", "acc", "Final Score Norm", "Final Score"]:
            if k in eval_results and safe_float(eval_results[k]) is not None:
                return float(eval_results[k])
    return None


def main():
    ap = argparse.ArgumentParser(description="ASCA sensitivity rerank-only analysis (no new generation).")
    ap.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS), help="comma-separated")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--candidate_file", default=None, help="optional direct diagnostics jsonl for single benchmark")
    ap.add_argument("--benchmark", default=None, help="single benchmark name (for single-run mode)")
    ap.add_argument("--w_sup", type=float, default=2.0)
    ap.add_argument("--w_valid", type=float, default=1.0)
    ap.add_argument("--w_base", type=float, default=0.4)
    ap.add_argument("--w_risk", type=float, default=0.5)
    ap.add_argument("--varied_param", default=None)
    ap.add_argument("--param_value", default=None)
    ap.add_argument("--sweep", choices=["quick", "single"], default="quick")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--official_eval", action="store_true", default=True)
    ap.add_argument("--require_official_eval", action="store_true", default=True)
    ap.add_argument(
        "--default-method",
        choices=["nocf", "routed"],
        default="routed",
        help="Reference default for delta_vs_default in Group A. routed uses v91_cf3_routed official score.",
    )
    ap.add_argument(
        "--full-support-only",
        action="store_true",
        help="Only rerank samples with full candidate support coverage; otherwise keep default prediction.",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    out_root = Path(args.output_dir) if args.output_dir else (repo_root / "paper_neurips2026_artifacts" / "sensitivity_asca_weights")
    out_root.mkdir(parents=True, exist_ok=True)

    build_dataset = None
    if args.official_eval:
        try:
            ensure_unsloth_stub()
            build_dataset = import_build_dataset_lightweight()
        except Exception as e:
            if args.require_official_eval:
                raise RuntimeError(f"official evaluator import failed: {e}") from e
            print(f"[WARN] official_eval disabled due to import failure: {e}")
            args.official_eval = False

    benches = [args.benchmark] if args.benchmark else [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    settings = quick_settings() if args.sweep == "quick" else custom_settings(args)
    default_name = setting_name(DEFAULT)

    # Load benchmark tables and diag sources.
    bench_data = {}
    missing = []
    for b in benches:
        ds = DATASET_TAG.get(b, "")
        default_xlsx, default_cfg = discover_best_default_xlsx(repo_root, args.n, b, ds)
        freq_xlsx = discover_xlsx(repo_root, args.n, f"test_config_smolvlm2_v91_nocf_ablation_frequency_only_{b}", "V91NoCFAbl_frequency_only_SmolVLM2_2B", ds)

        if default_xlsx is None:
            missing.append((b, "missing_default_xlsx"))
            continue

        rows = read_table_file(default_xlsx)
        default_pred = {}
        gt_map = {}
        q_map = {}
        for i, r in enumerate(rows, 1):
            sid = str(i)
            default_pred[sid] = str(r.get("prediction", ""))
            gt_map[sid] = str(r.get("answer", ""))
            q_map[sid] = str(r.get("question", ""))

        freq_pred = {}
        if freq_xlsx is not None:
            frows = read_table_file(freq_xlsx)
            for i, r in enumerate(frows, 1):
                sid = str(i)
                freq_pred[sid] = str(r.get("prediction", ""))

        # Main candidate source (no_format preferred).
        if args.candidate_file and args.benchmark == b:
            main_diag_path = Path(args.candidate_file)
            main_diag_source = "cli_candidate_file"
        else:
            main_diag_path, main_diag_source = discover_best_main_diag(repo_root, args.n, b)
        if main_diag_path is None:
            missing.append((b, "missing_candidate_diag"))
            continue

        aux_diags = {}
        for v in ["no_format", "no_length_risk", "frequency_only", "majority_vote", "no_base_bias"]:
            p = discover_diag(repo_root, f"test_config_smolvlm2_v91_nocf_ablation_{v}_{b}_n{args.n}")
            if p is not None:
                aux_diags[v] = load_diag_map(p)

        bench_data[b] = {
            "default_cfg": default_cfg,
            "default_xlsx": default_xlsx,
            "routed_xlsx": discover_routed_xlsx(repo_root, args.n, b, ds),
            "default_metric_file": discover_metric_file(repo_root, args.n, default_cfg, "V91NoCF_SmolVLM2_2B", ds),
            "default_rows": {str(i): r for i, r in enumerate(rows, 1)},
            "default_pred": default_pred,
            "gt_map": gt_map,
            "q_map": q_map,
            "freq_pred": freq_pred,
            "main_diag_path": main_diag_path,
            "main_diag_source": main_diag_source,
            "main_diag": load_diag_map(main_diag_path),
            "aux_diags": aux_diags,
        }

    raw_rows = []
    changed_rows = []

    for b in benches:
        if b not in bench_data:
            continue
        bd = bench_data[b]
        default_pred = bd["default_pred"]
        gt_map = bd["gt_map"]
        q_map = bd["q_map"]
        main_diag = bd["main_diag"]
        aux_diags = bd["aux_diags"]
        official_default_eval = None
        official_routed_eval = None
        if args.official_eval:
            ds_name = DATASET_TAG.get(b, "")
            ds0 = build_dataset(ds_name)
            ret0 = ds0.evaluate(str(bd["default_xlsx"]))
            official_default_eval = extract_official_score(ret0)
            if official_default_eval is None and args.require_official_eval:
                raise RuntimeError(f"official default eval failed for {b}: {bd['default_xlsx']}")
            if bd.get("routed_xlsx") is not None:
                ret_r = ds0.evaluate(str(bd["routed_xlsx"]))
                official_routed_eval = extract_official_score(ret_r)
            if args.default_method == "routed" and official_routed_eval is None and args.require_official_eval:
                raise RuntimeError(f"official routed default eval failed for {b}: {bd.get('routed_xlsx')}")

        for st in settings:
            st_name = st["setting_name"]
            pred_dir = out_root / st_name
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_path = pred_dir / f"{b}_predictions.csv"

            s_preds = {}
            miss_diag = 0
            miss_support_candidate = 0

            for sid in default_pred.keys():
                # Anchor the default setting to the exact official default predictions.
                if st_name == default_name:
                    s_preds[sid] = default_pred[sid]
                    continue
                # same rerank path for every setting, including default,
                # to avoid path-mismatch artifacts in sensitivity.
                diag = main_diag.get(sid)
                if diag is None:
                    miss_diag += 1
                    s_preds[sid] = default_pred[sid]
                    continue

                # merge support from all available variants' scored_top
                support_map = {}
                # 1) always trust same-sample scored_top first (same candidate universe)
                support_map.update(extract_support(diag.get("scored_top")))
                # 2) then merge aux ablation diagnostics if available
                for _, m in aux_diags.items():
                    d = m.get(sid)
                    if d is None:
                        continue
                    sm = extract_support(d.get("scored_top"))
                    for k, v in sm.items():
                        support_map[k] = v

                cand_list = diag.get("candidate_list") or []
                sample_miss = 0
                for c in cand_list:
                    n = normalize_answer(c)
                    if n and n not in support_map:
                        sample_miss += 1
                miss_support_candidate += sample_miss

                if args.full_support_only and sample_miss > 0:
                    s_preds[sid] = default_pred[sid]
                    continue

                pred, scored, _ = rerank_one(
                    diag,
                    q_map.get(sid, ""),
                    support_map,
                    st,
                    default_pred_raw=default_pred[sid],
                )
                if pred is None:
                    pred = default_pred[sid]
                s_preds[sid] = pred

            # metrics
            n_total = len(default_pred)
            changed = 0
            default_wins = 0
            setting_wins = 0
            correct = 0
            changed_vs_freq = None if not bd["freq_pred"] else 0

            for sid, dp in default_pred.items():
                sp = s_preds.get(sid, dp)
                if norm_text(sp) != norm_text(dp):
                    changed += 1

                row = bd["default_rows"][sid]
                gt = gt_map[sid]
                d_ok = is_correct(gt, dp, row)
                s_ok = is_correct(gt, sp, row)
                if d_ok and not s_ok:
                    default_wins += 1
                elif s_ok and not d_ok:
                    setting_wins += 1
                if s_ok:
                    correct += 1

                if changed_vs_freq is not None:
                    fp = bd["freq_pred"].get(sid)
                    if fp is not None and norm_text(fp) != norm_text(sp):
                        changed_vs_freq += 1

            score = correct / max(1, n_total)

            # default score in this benchmark (from prediction file, consistent with current correctness proxy)
            d_correct = 0
            for sid, dp in default_pred.items():
                if is_correct(gt_map[sid], dp, bd["default_rows"][sid]):
                    d_correct += 1
            d_score = d_correct / max(1, n_total)

            official_score = None
            if args.official_eval:
                ds_name = DATASET_TAG.get(b, "")
                ds = build_dataset(ds_name)
                eval_xlsx = pred_dir / f"{b}_official_eval_input.xlsx"
                write_prediction_xlsx(bd["default_rows"], s_preds, eval_xlsx)
                eval_ret = ds.evaluate(str(eval_xlsx))
                official_score = extract_official_score(eval_ret)
                if official_score is None and args.require_official_eval:
                    raise RuntimeError(f"official score extraction failed for {b} {st_name}")

            net = setting_wins - default_wins

            with pred_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["sample_id", "prediction", "default_prediction", "changed_vs_default"])
                w.writeheader()
                for sid in default_pred.keys():
                    sp = s_preds.get(sid, default_pred[sid])
                    w.writerow({
                        "sample_id": sid,
                        "prediction": sp,
                        "default_prediction": default_pred[sid],
                        "changed_vs_default": int(norm_text(sp) != norm_text(default_pred[sid])),
                    })

            official_default_nocf = official_default_eval if official_default_eval is not None else read_official_score(bd.get("default_metric_file"))
            official_default = official_default_nocf
            default_basis = "nocf_official"
            if args.default_method == "routed" and official_routed_eval is not None:
                official_default = official_routed_eval
                default_basis = "routed_official"
            if args.official_eval and official_default is None and args.require_official_eval:
                raise RuntimeError(f"missing official default score for {b} from {bd.get('default_metric_file')}")
            if official_score is not None and official_default is not None:
                delta = official_score - official_default
                metric_name = "official_metric"
                score_out = f"{official_score:.6f}"
            else:
                delta = score - d_score
                metric_name = "accuracy_proxy"
                score_out = f"{score:.6f}"
            raw_rows.append({
                "benchmark": b,
                "n": n_total,
                "varied_param": st["varied_param"],
                "param_value": st["param_value"],
                "setting_name": st_name,
                "w_sup": st["w_sup"],
                "w_valid": st["w_valid"],
                "w_base": st["w_base"],
                "w_risk": st["w_risk"],
                "metric": metric_name,
                "score": score_out,
                "delta_vs_default": f"{delta:.6f}",
                "official_default_score": ("NA" if official_default is None else f"{official_default:.6f}"),
                "official_nocf_default_score": ("NA" if official_default_nocf is None else f"{official_default_nocf:.6f}"),
                "official_routed_default_score": ("NA" if official_routed_eval is None else f"{official_routed_eval:.6f}"),
                "default_method": args.default_method,
                "default_basis": default_basis,
                "proxy_score": f"{score:.6f}",
                "proxy_default_score": f"{d_score:.6f}",
                "changed_vs_default": changed,
                "default_wins": default_wins,
                "setting_wins": setting_wins,
                "net_vs_default": net,
                "changed_vs_frequency_only": ("NA" if changed_vs_freq is None else changed_vs_freq),
                "setting_output_file": str(pred_path),
                "candidate_source": str(bd["main_diag_path"]),
                "candidate_source_policy": bd.get("main_diag_source", ""),
                "default_source_cfg": bd.get("default_cfg", ""),
                "missing_diag_samples": miss_diag,
                "missing_support_candidates": miss_support_candidate,
            })

            changed_rows.append({
                "benchmark": b,
                "setting_name": st_name,
                "changed": changed,
                "default_wins": default_wins,
                "setting_wins": setting_wins,
                "net": net,
            })

    # save raw
    raw_csv = out_root / "sensitivity_raw.csv"
    with raw_csv.open("w", newline="") as f:
        if raw_rows:
            w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            w.writeheader(); w.writerows(raw_rows)
        else:
            f.write(
                "benchmark,n,varied_param,param_value,setting_name,w_sup,w_valid,w_base,w_risk,metric,score,delta_vs_default,official_default_score,official_nocf_default_score,official_routed_default_score,default_method,default_basis,proxy_score,proxy_default_score,changed_vs_default,default_wins,setting_wins,net_vs_default,changed_vs_frequency_only,setting_output_file,candidate_source,candidate_source_policy,default_source_cfg,missing_diag_samples,missing_support_candidates\n"
            )

    # Rebased view: delta versus the sensitivity-default setting itself
    # (same pipeline / same run), to avoid cross-pipeline baseline mismatch.
    rebased_csv = out_root / "sensitivity_rebased_vs_setting_default.csv"
    rebased_rows = []
    if raw_rows:
        by_bench_default = {}
        for r in raw_rows:
            if r.get("setting_name") == default_name:
                by_bench_default[r["benchmark"]] = float(r["score"])
        for r in raw_rows:
            b = r["benchmark"]
            s = float(r["score"])
            d = by_bench_default.get(b)
            rr = dict(r)
            rr["delta_vs_setting_default"] = "" if d is None else f"{(s - d):.6f}"
            rebased_rows.append(rr)
    with rebased_csv.open("w", newline="") as f:
        if rebased_rows:
            flds = list(rebased_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=flds)
            w.writeheader()
            w.writerows(rebased_rows)
        else:
            f.write(
                "benchmark,n,varied_param,param_value,setting_name,w_sup,w_valid,w_base,w_risk,metric,score,delta_vs_default,official_default_score,official_nocf_default_score,official_routed_default_score,default_method,default_basis,proxy_score,proxy_default_score,changed_vs_default,default_wins,setting_wins,net_vs_default,changed_vs_frequency_only,setting_output_file,candidate_source,candidate_source_policy,default_source_cfg,missing_diag_samples,missing_support_candidates,delta_vs_setting_default\n"
            )

    changed_csv = out_root / "sensitivity_changed_cases.csv"
    with changed_csv.open("w", newline="") as f:
        if changed_rows:
            w = csv.DictWriter(f, fieldnames=list(changed_rows[0].keys()))
            w.writeheader(); w.writerows(changed_rows)
        else:
            f.write("benchmark,setting_name,changed,default_wins,setting_wins,net\n")

    # summary by benchmark
    by_bench = defaultdict(list)
    for r in raw_rows:
        by_bench[r["benchmark"]].append(r)

    summary_bench_rows = []
    for b, rows in by_bench.items():
        rows2 = sorted(rows, key=lambda x: float(x["score"]), reverse=True)
        drow = next((x for x in rows if x["setting_name"] == setting_name(DEFAULT)), None)
        dscore = float(drow["score"]) if drow else None
        best = rows2[0]
        bscore = float(best["score"])
        bdelta = float(best["delta_vs_default"])

        # isolated spike check among same varied param neighbors
        stable = "unknown"
        if best["varied_param"] in {"w_sup", "w_valid"}:
            vp = best["varied_param"]
            val = float(best["param_value"])
            points = WSUP_POINTS if vp == "w_sup" else WVALID_POINTS
            neighbors = [x for x in [val - (0.5 if vp == "w_sup" else 0.25), val + (0.5 if vp == "w_sup" else 0.25)] if x in points]
            n_deltas = []
            for nval in neighbors:
                rr = next((x for x in rows if x["varied_param"] == vp and abs(float(x["param_value"]) - nval) < 1e-9), None)
                if rr is not None:
                    n_deltas.append(float(rr["delta_vs_default"]))
            if not n_deltas:
                stable = "isolated"
            elif any(d >= bdelta - 0.001 for d in n_deltas):
                stable = "stable"
            else:
                stable = "isolated"
        elif best["setting_name"] == setting_name(DEFAULT):
            stable = "default"

        warn = ""
        if b in {"textvqa", "ocrbench"} and bdelta < -0.002:
            warn = "best_setting_hurts_key_benchmark"

        summary_bench_rows.append({
            "benchmark": b,
            "default_score": ("NA" if dscore is None else f"{dscore:.6f}"),
            "best_setting": best["setting_name"],
            "best_score": f"{bscore:.6f}",
            "best_delta": f"{bdelta:.6f}",
            "stability": stable,
            "warning": warn,
        })

    summary_bench_csv = out_root / "sensitivity_summary_by_benchmark.csv"
    with summary_bench_csv.open("w", newline="") as f:
        if summary_bench_rows:
            w = csv.DictWriter(f, fieldnames=list(summary_bench_rows[0].keys()))
            w.writeheader(); w.writerows(summary_bench_rows)
        else:
            f.write("benchmark,default_score,best_setting,best_score,best_delta,stability,warning\n")

    # summary by setting
    by_setting = defaultdict(list)
    for r in raw_rows:
        by_setting[r["setting_name"]].append(r)

    # per benchmark ranks
    bench_ranks = {}
    for b, rows in by_bench.items():
        scores = {r["setting_name"]: float(r["score"]) for r in rows}
        bench_ranks[b] = rank_map(scores)

    summary_setting_rows = []
    for sname, rows in by_setting.items():
        deltas = [float(r["delta_vs_default"]) for r in rows]
        improved = sum(1 for d in deltas if d > 0)
        hurt = sum(1 for d in deltas if d < 0)
        avg_delta = sum(deltas) / max(1, len(deltas))
        worst = min(deltas) if deltas else 0.0
        ranks = [bench_ranks[r["benchmark"]][sname] for r in rows if r["benchmark"] in bench_ranks and sname in bench_ranks[r["benchmark"]]]
        mean_rank = sum(ranks) / max(1, len(ranks))

        # candidate new default criteria
        textvqa_delta = next((float(r["delta_vs_default"]) for r in rows if r["benchmark"] == "textvqa"), 0.0)
        ocrbench_delta = next((float(r["delta_vs_default"]) for r in rows if r["benchmark"] == "ocrbench"), 0.0)
        non_iso = (improved >= 2) and (improved >= hurt)
        no_key_harm = (textvqa_delta >= -0.002) and (ocrbench_delta >= -0.002)
        not_spike = (max(deltas) < 0.03 or improved >= 2) if deltas else False
        net_nonneg = True
        for r in rows:
            if int(r["net_vs_default"]) < 0:
                net_nonneg = False
                break
        candidate_new_default = all([improved >= 2, no_key_harm, non_iso, not_spike, net_nonneg])

        summary_setting_rows.append({
            "setting_name": sname,
            "w_sup": rows[0]["w_sup"],
            "w_valid": rows[0]["w_valid"],
            "w_base": rows[0]["w_base"],
            "w_risk": rows[0]["w_risk"],
            "mean_rank": f"{mean_rank:.3f}",
            "n_benchmarks": len(rows),
            "improved_count": improved,
            "hurt_count": hurt,
            "average_delta": f"{avg_delta:.6f}",
            "worst_negative_delta": f"{worst:.6f}",
            "candidate_new_default": "yes" if candidate_new_default else "no",
            "note": "default" if sname == setting_name(DEFAULT) else "",
        })

    summary_setting_rows = sorted(summary_setting_rows, key=lambda x: (x["candidate_new_default"] != "yes", float(x["mean_rank"])))
    summary_setting_csv = out_root / "sensitivity_summary_by_setting.csv"
    with summary_setting_csv.open("w", newline="") as f:
        if summary_setting_rows:
            w = csv.DictWriter(f, fieldnames=list(summary_setting_rows[0].keys()))
            w.writeheader(); w.writerows(summary_setting_rows)
        else:
            f.write("setting_name,w_sup,w_valid,w_base,w_risk,mean_rank,n_benchmarks,improved_count,hurt_count,average_delta,worst_negative_delta,candidate_new_default,note\n")

    # report
    report = []
    report.append("# ASCA Sensitivity Report")
    report.append("")
    report.append("## 默认参数")
    report.append(f"- w_sup={DEFAULT['w_sup']}, w_valid={DEFAULT['w_valid']}, w_base={DEFAULT['w_base']}, w_risk={DEFAULT['w_risk']}")
    report.append("")
    report.append("## 运行范围")
    report.append(f"- benchmarks: {', '.join(benches)}")
    report.append(f"- n: {args.n}")
    report.append(f"- settings: {len(settings)} (sweep={args.sweep})")
    report.append("- 说明：仅复用已有 diagnostics / predictions 做 rerank+eval，没有重新生成 VLM 输出。")
    report.append("")
    if missing:
        report.append("## 缺失输入")
        for b, why in missing:
            report.append(f"- {b}: {why}")
        report.append("")

    report.append("## 候选新默认参数建议")
    cands = [r for r in summary_setting_rows if r["candidate_new_default"] == "yes"]
    if cands:
        for c in cands[:5]:
            report.append(f"- {c['setting_name']} (avg_delta={c['average_delta']}, improved={c['improved_count']}, hurt={c['hurt_count']})")
    else:
        report.append("- 当前无 setting 满足 candidate_new_default 全部条件，建议保持默认参数。")
    report.append("")

    report.append("## 文件输出")
    report.append(f"- {raw_csv}")
    report.append(f"- {summary_bench_csv}")
    report.append(f"- {summary_setting_csv}")
    report.append(f"- {changed_csv}")
    report.append(f"- {rebased_csv}")
    report.append(f"- {out_root / 'sensitivity_report.md'}")
    report.append("")

    report.append("## 口径提醒")
    if args.official_eval:
        report.append("- score/delta_vs_default 使用官方 evaluator（与主表同口径）。")
        report.append(f"- default_method={args.default_method}（delta 参考分数来源：{args.default_method}）。")
    else:
        report.append("- score/delta_vs_default 使用 sample-level accuracy proxy（非主表口径）。")
    report.append("- changed/default_wins/setting_wins/net_vs_default 始终按 sample-level 口径统计。")

    (out_root / "sensitivity_report.md").write_text("\n".join(report) + "\n")

    print(f"[DONE] raw={raw_csv}")
    print(f"[DONE] by_benchmark={summary_bench_csv}")
    print(f"[DONE] by_setting={summary_setting_csv}")
    print(f"[DONE] changed={changed_csv}")
    print(f"[DONE] report={out_root / 'sensitivity_report.md'}")


if __name__ == "__main__":
    main()
