#!/usr/bin/env python3
import argparse
import ast
import csv
import glob
import json
import re
import string
import sys
import types
import importlib
import importlib.machinery
from pathlib import Path

from common import read_table_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_MCQ_LETTER_RE = re.compile(r"^\(?([A-Ea-e])\)?(?:[\.:\)\s]|$)")
_YESNO_SET = {"yes", "no", "true", "false"}
_YES_NO = {"yes", "no", "true", "false", "yeah", "nope"}
_NUMERIC_RE = re.compile(
    r"^[\$£€¥]?[+-]?\d+(?:[\.,]\d+)*(?:%|°|usd|gb|kg|mg|cm|mm|km|kmh|hp|ml|l|kw|kwh)?\.?$",
    flags=re.IGNORECASE,
)
_DATE_RE = re.compile(r"\b(19|20)\d{2}\b|\b\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}\b")

_OCR_INTENT_PATTERNS = [
    "what does the sign", "what does it say", "what is written",
    "what word is", "read the text", "read the label", "read the sign",
    "what is the label", "what is the title", "what is the brand",
    "what brand", "what company", "what product", "what book",
    "what author", "what number is", "what number on", "what time is",
    "license plate", "what does the text", "what is the name",
]

BENCH_CFG = {
    "textvqa": {
        "dataset_tag": "TextVQA_VAL",
        "nocf_cfg": "test_config_smolvlm2_v91_nocf_regen_textvqa",
        "nocf_model": "V91NoCF_SmolVLM2_2B",
        "routed_cfg": "test_config_smolvlm2_v91_cf3_routed_textvqa",
        "routed_model": "V91CF3Routed_SmolVLM2_2B",
    },
    "chartqa": {
        "dataset_tag": "ChartQA_TEST",
        "nocf_cfg": "test_config_smolvlm2_v91_nocf_regen_chartqa",
        "nocf_model": "V91NoCF_SmolVLM2_2B",
        "routed_cfg": "test_config_smolvlm2_v91_cf3_routed_chartqa",
        "routed_model": "V91CF3Routed_SmolVLM2_2B",
    },
    "ocrvqa": {
        "dataset_tag": "OCRVQA_TEST",
        "nocf_cfg": "test_config_smolvlm2_v91_nocf_regen_ocrvqa",
        "nocf_model": "V91NoCF_SmolVLM2_2B",
        "routed_cfg": "test_config_smolvlm2_v91_cf3_routed_ocrvqa",
        "routed_model": "V91CF3Routed_SmolVLM2_2B",
    },
    "ocrbench": {
        "dataset_tag": "OCRBench",
        "nocf_cfg": "test_config_smolvlm2_v91_nocf_regen_ocrbench",
        "nocf_model": "V91NoCF_SmolVLM2_2B",
        "routed_cfg": "test_config_smolvlm2_v91_cf3_routed_ocrbench",
        "routed_model": "V91CF3Routed_SmolVLM2_2B",
    },
}


def norm_text(s):
    return str(s or "").strip().lower().translate(_PUNCT_TABLE).strip()


def normalize(s):
    return norm_text(s)


def answer_length(s):
    return len(normalize(s).split())


def is_alphanumeric_short(s):
    n = normalize(s)
    if not n or len(n) > 32:
        return False
    return bool(re.match(r"^[a-z0-9 \-_/\.]+$", n))


def is_numeric(s):
    n = normalize(s)
    if not n:
        return False
    if _NUMERIC_RE.match(n):
        return True
    return bool(_DATE_RE.search(n))


def has_visual_text_intent(question):
    q = (question or "").lower()
    if any(p in q for p in _OCR_INTENT_PATTERNS):
        return True
    extra = (
        "text", "written", "write", "word", "letters", "spelling",
        "sign", "label", "logo", "brand", "name", "title",
        "license plate", "plate", "phone number", "address",
        "serial", "model", "code", "barcode", "price tag",
    )
    return any(k in q for k in extra)


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


def load_rows(xlsx_path: Path):
    return read_table_file(xlsx_path)


def load_diag_map(path: Path):
    out = {}
    with path.open() as f:
        for ln in f:
            if not ln.strip():
                continue
            try:
                o = json.loads(ln)
            except Exception:
                continue
            sid = str(o.get("sample_id", "")).strip()
            if sid:
                out[sid] = o
    return out


def discover_cf_diag(repo_root: Path, bench: str):
    cands = sorted(glob.glob(str(repo_root / f".runtime_cache/test_config_smolvlm2_v91_cf3_routed_{bench}*/diagnostics/v91cf3_samples.jsonl")))
    if not cands:
        return None
    best = None
    best_n = -1
    for p in cands:
        n = 0
        with open(p) as f:
            for ln in f:
                if ln.strip():
                    n += 1
        if n > best_n:
            best_n = n
            best = p
    return Path(best)


def extract_general_and_support(diag):
    gen = {}
    sup = {}
    st = diag.get("scored_top") or []
    for e in st:
        if not isinstance(e, list) or len(e) < 5:
            continue
        n = normalize(e[0])
        if not n:
            continue
        try:
            gen[n] = float(e[2])
        except Exception:
            pass
        try:
            sup[n] = float(e[4])
        except Exception:
            pass
    return gen, sup


def format_validity(candidate_norm: str, answer_space: str):
    n = normalize(candidate_norm)
    lw = answer_length(n)
    if answer_space == "yes_no":
        return 1.0 if n in _YES_NO else 0.0
    if answer_space == "multiple_choice":
        return 1.0 if bool(re.match(r"^[a-h]$", n)) else 0.2
    if answer_space == "numeric":
        return 1.0 if is_numeric(n) else 0.4
    if answer_space == "ocr_text_short":
        return 1.0 if is_alphanumeric_short(n) else 0.3
    if answer_space == "open_entity":
        return 1.0 if lw <= 5 else 0.6
    if answer_space == "chart_or_diagram":
        return 1.0 if (is_numeric(n) or lw <= 6) else 0.5
    if answer_space == "caption_like":
        return 1.0 if lw >= 4 else 0.3
    return 1.0


def routed_space_ok(diag, question: str):
    answer_space = str(diag.get("answer_space", "unknown"))
    candidate_map = diag.get("cf_score") or {}
    margin = float(diag.get("margin", 0.0) or 0.0)
    entropy = float(diag.get("entropy", 0.0) or 0.0)

    if answer_space in ("chart_or_diagram", "caption_like", "multiple_choice", "yes_no", "numeric"):
        return False, "answer_space_block"

    if len(candidate_map) < 2:
        return False, "no_valid_alternative"

    text_intent = has_visual_text_intent(question)
    uncertain = (entropy >= 0.70) or (margin <= 0.50)
    min_textlike = 2
    max_words = 4
    max_chars = 24

    textlike = 0
    for n_ans in candidate_map.keys():
        n = normalize(n_ans)
        if not n:
            continue
        if answer_length(n) > max_words:
            continue
        if len(n) > max_chars:
            continue
        if not is_alphanumeric_short(n):
            continue
        if n in _YES_NO:
            continue
        if not re.search(r"[a-z0-9]", n):
            continue
        textlike += 1

    if answer_space == "ocr_text_short":
        return True, "routed_ocr_text_short"
    if answer_space == "open_entity":
        if text_intent:
            return True, "routed_open_entity_text_intent"
        if textlike >= min_textlike and uncertain:
            return True, "routed_open_entity_uncertain_textlike"
        return False, "open_entity_not_text_grounded"
    if answer_space == "unknown":
        if text_intent and textlike >= min_textlike and uncertain:
            return True, "routed_unknown_text_intent"
        return False, "unknown_not_routed"
    return False, "answer_space_block"


def pick_best_alt(diag):
    cf = diag.get("cf_score") or {}
    if not isinstance(cf, dict) or len(cf) < 2:
        return None
    a0 = normalize(diag.get("no_cf_winner", ""))
    if a0 not in cf:
        return None
    alts = [(k, float(v)) for k, v in cf.items() if k != a0]
    if not alts:
        return None
    a1, s1 = sorted(alts, key=lambda kv: kv[1], reverse=True)[0]
    return a0, a1, float(cf[a0]), s1


def write_prediction_xlsx(rows, predictions, out_xlsx: Path):
    import pandas as pd
    cols = list(rows[0].keys()) if rows else []
    if "prediction" not in cols:
        cols.append("prediction")
    table = []
    for i, r in enumerate(rows, 1):
        sid = str(i)
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
            if k in eval_results:
                return float(eval_results[k])
    return None


def parse_float_grid(s):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="CF sensitivity by tau_cf and tau_gap under fixed routed gates.")
    ap.add_argument("--repo_root", default=str(REPO_ROOT))
    ap.add_argument("--benchmarks", default="textvqa,chartqa")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--tau_cf_grid", default="0.0,0.05,0.10,0.15,0.20")
    ap.add_argument("--tau_gap_grid", default="0.10,0.15,0.20,0.30")
    ap.add_argument("--official_eval", action="store_true", default=True)
    ap.add_argument("--require_official_eval", action="store_true", default=True)
    ap.add_argument("--output_dir", default="paper_neurips2026_artifacts/sensitivity_cf_gate/tau_cf_gap_grid")
    args = ap.parse_args()

    repo = Path(args.repo_root)
    outdir = repo / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.official_eval:
        try:
            ensure_unsloth_stub()
            build_dataset = import_build_dataset_lightweight()
        except Exception as e:
            if args.require_official_eval:
                raise RuntimeError(f"official evaluator import failed: {e}") from e
            print(f"[WARN] official eval disabled: {e}")
            args.official_eval = False
            build_dataset = None
    else:
        build_dataset = None

    tau_cf_vals = parse_float_grid(args.tau_cf_grid)
    tau_gap_vals = parse_float_grid(args.tau_gap_grid)

    summary = []

    for b in [x.strip() for x in args.benchmarks.split(",") if x.strip()]:
        if b not in BENCH_CFG:
            print(f"[WARN] unknown benchmark: {b}")
            continue
        cfg = BENCH_CFG[b]

        nocf_xlsx = discover_xlsx(repo, args.n, cfg["nocf_cfg"], cfg["nocf_model"], cfg["dataset_tag"])
        if nocf_xlsx is None:
            print(f"[WARN] {b}: missing nocf xlsx")
            continue
        rows = load_rows(nocf_xlsx)
        if not rows:
            print(f"[WARN] {b}: empty rows")
            continue

        diag_path = discover_cf_diag(repo, b)
        if diag_path is None:
            print(f"[WARN] {b}: missing cf diag")
            continue
        diag_map = load_diag_map(diag_path)

        asca_pred = {str(i): str(r.get("prediction", "")) for i, r in enumerate(rows, 1)}
        gt = {str(i): str(r.get("answer", "")) for i, r in enumerate(rows, 1)}

        nocf_official = None
        if args.official_eval:
            ds0 = build_dataset(cfg["dataset_tag"])
            try:
                nocf_official = extract_official_score(ds0.evaluate(str(nocf_xlsx)))
            except Exception:
                nocf_official = None

        for tau_cf in tau_cf_vals:
            for tau_gap in tau_gap_vals:
                preds = {}
                eligible = switched = cwin = closs = routed = 0
                block_counts = {}

                for i, r in enumerate(rows, 1):
                    sid = str(i)
                    base = asca_pred[sid]
                    d = diag_map.get(sid)
                    if d is None:
                        preds[sid] = base
                        block_counts["missing_diag"] = block_counts.get("missing_diag", 0) + 1
                        continue

                    q = str(r.get("question", "") or r.get("Question", "") or "")
                    space_ok, space_reason = routed_space_ok(d, q)
                    if not space_ok:
                        preds[sid] = base
                        block_counts[space_reason] = block_counts.get(space_reason, 0) + 1
                        continue

                    pick = pick_best_alt(d)
                    if pick is None:
                        preds[sid] = base
                        block_counts["no_cf_scores"] = block_counts.get("no_cf_scores", 0) + 1
                        continue

                    a0, a1, cf0, cf1 = pick
                    eligible += 1
                    delta_cf = cf1 - cf0

                    gen_map, sup_map = extract_general_and_support(d)
                    if a0 not in gen_map or a1 not in gen_map:
                        preds[sid] = base
                        block_counts["missing_general_score"] = block_counts.get("missing_general_score", 0) + 1
                        continue
                    gap = float(gen_map[a0]) - float(gen_map[a1])

                    # fixed non-tau gates
                    n_views = int(d.get("n_views", 8) or 8)
                    sup = float(sup_map.get(a1, 0.0))
                    view_count = int(round(sup * max(1, n_views)))
                    fmt_ok = format_validity(a1, str(d.get("answer_space", "unknown"))) >= 0.8
                    mask_q = str(d.get("mask_quality", "low"))
                    ctrl_q = str(d.get("control_quality", "low"))

                    if view_count < 2:
                        preds[sid] = base
                        block_counts["low_view_support"] = block_counts.get("low_view_support", 0) + 1
                        continue
                    if not fmt_ok:
                        preds[sid] = base
                        block_counts["format_veto"] = block_counts.get("format_veto", 0) + 1
                        continue
                    if mask_q not in ("medium", "high"):
                        preds[sid] = base
                        block_counts["low_mask_quality"] = block_counts.get("low_mask_quality", 0) + 1
                        continue
                    if ctrl_q not in ("matched_grid", "matched_periphery"):
                        preds[sid] = base
                        block_counts["low_control_quality"] = block_counts.get("low_control_quality", 0) + 1
                        continue

                    # switch rule from paper
                    # CF(a1)-CF(a0) > tau_cf and S_ASCA(a0)-S_ASCA(a1) <= tau_gap
                    if not (delta_cf > tau_cf):
                        preds[sid] = base
                        block_counts["low_cf_margin"] = block_counts.get("low_cf_margin", 0) + 1
                        continue
                    if not (gap <= tau_gap):
                        preds[sid] = base
                        block_counts["general_score_gap_too_large"] = block_counts.get("general_score_gap_too_large", 0) + 1
                        continue

                    # pass all gates -> switch
                    cand_raw = None
                    for c in (d.get("candidate_list") or []):
                        if normalize(c) == a1:
                            cand_raw = str(c)
                            break
                    if not cand_raw:
                        cand_raw = a1
                    preds[sid] = cand_raw
                    routed += 1

                    if normalize(cand_raw) != normalize(base):
                        switched += 1
                        b_ok = is_correct(gt[sid], base, r)
                        n_ok = is_correct(gt[sid], cand_raw, r)
                        if n_ok and not b_ok:
                            cwin += 1
                        elif b_ok and not n_ok:
                            closs += 1

                correct = sum(1 for i, r in enumerate(rows, 1) if is_correct(gt[str(i)], preds.get(str(i), asca_pred[str(i)]), r))
                asca_correct = sum(1 for i, r in enumerate(rows, 1) if is_correct(gt[str(i)], asca_pred[str(i)], r))
                proxy_score = correct / max(1, len(rows))
                proxy_asca = asca_correct / max(1, len(rows))
                proxy_delta = proxy_score - proxy_asca

                off_score = None
                off_delta = None
                if args.official_eval:
                    ds = build_dataset(cfg["dataset_tag"])
                    eval_xlsx = outdir / "official_eval_inputs" / b / f"taucf_{tau_cf:.2f}_taugap_{tau_gap:.2f}_{cfg['dataset_tag']}.xlsx"
                    write_prediction_xlsx(rows, preds, eval_xlsx)
                    eval_ret = ds.evaluate(str(eval_xlsx))
                    off_score = extract_official_score(eval_ret)
                    if (off_score is not None) and (nocf_official is not None):
                        off_delta = float(off_score) - float(nocf_official)

                row_out = {
                    "benchmark": b,
                    "n": len(rows),
                    "tau_cf": f"{tau_cf:.3f}",
                    "tau_gap": f"{tau_gap:.3f}",
                    "score": "NA" if off_score is None else f"{off_score:.6f}",
                    "delta_vs_nocf": "NA" if off_delta is None else f"{off_delta:.6f}",
                    "proxy_score": f"{proxy_score:.6f}",
                    "proxy_delta_vs_nocf": f"{proxy_delta:.6f}",
                    "eligible_count": eligible,
                    "routed_count": routed,
                    "switch_count": switched,
                    "correct_switches": cwin,
                    "incorrect_switches": closs,
                    "net_switch_gain": cwin - closs,
                    "switch_coverage": f"{(switched/max(1,len(rows))):.6f}",
                    "block_counts": json.dumps(block_counts, ensure_ascii=False),
                    "source_nocf_xlsx": str(nocf_xlsx),
                    "source_cf_diag": str(diag_path),
                }
                summary.append(row_out)

    out_csv = outdir / "cf_tau_sensitivity_summary.csv"
    fields = [
        "benchmark", "n", "tau_cf", "tau_gap", "score", "delta_vs_nocf",
        "proxy_score", "proxy_delta_vs_nocf",
        "eligible_count", "routed_count", "switch_count", "correct_switches", "incorrect_switches",
        "net_switch_gain", "switch_coverage", "block_counts", "source_nocf_xlsx", "source_cf_diag"
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary:
            w.writerow(r)

    print(f"[DONE] {out_csv}")


if __name__ == "__main__":
    main()
