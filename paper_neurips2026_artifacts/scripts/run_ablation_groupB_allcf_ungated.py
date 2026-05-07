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

from common import read_table_file

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_MCQ_LETTER_RE = re.compile(r"^\(?([A-Ea-e])\)?(?:[\.:\)\s]|$)")
_YESNO_SET = {"yes", "no", "true", "false"}
_YES_NO = {"yes", "no", "true", "false", "yeah", "nope"}

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def norm_text(s):
    return str(s or "").strip().lower().translate(_PUNCT_TABLE).strip()


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


def write_prediction_xlsx(src_headers, src_rows, predictions, out_xlsx: Path):
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
    b = benchmark.lower()
    if eval_results is None:
        return None
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


def fmt_float(x):
    if x is None:
        return "NA"
    return f"{float(x):.6f}"


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


def load_jsonl_map(path: Path):
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


def normalize(s):
    return norm_text(s)


def main():
    ap = argparse.ArgumentParser(description="Group B ungated CF ablation from existing CF diagnostics.")
    ap.add_argument("--repo_root", default="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--benchmark", default="textvqa")
    ap.add_argument("--dataset_tag", default="TextVQA_VAL")
    ap.add_argument("--full_cfg_stem", default="test_config_smolvlm2_v91_nocf_regen_textvqa")
    ap.add_argument("--full_model", default="V91NoCF_SmolVLM2_2B")
    ap.add_argument("--cf_diag", default=".runtime_cache/test_config_smolvlm2_v91_cf3_routed_textvqa_n1000/diagnostics/v91cf3_samples.jsonl")
    ap.add_argument("--output_dir", default="paper_neurips2026_artifacts/ablations/groupA_groupB_from_regen")
    ap.add_argument("--official_eval", action="store_true", default=True)
    ap.add_argument("--official_eval_inputs_subdir", default="official_eval_inputs_groupB")
    args = ap.parse_args()

    repo = Path(args.repo_root)
    outdir = repo / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    xlsx = discover_xlsx(repo, args.n, args.full_cfg_stem, args.full_model, args.dataset_tag)
    if xlsx is None:
        raise SystemExit(f"Missing full ASCA xlsx for {args.benchmark}: {args.full_cfg_stem}")

    cf_diag_path = repo / args.cf_diag
    if not cf_diag_path.exists():
        raise SystemExit(f"Missing CF diagnostics: {cf_diag_path}")

    src_headers, rows = load_source_rows(xlsx)
    cf_diag = load_jsonl_map(cf_diag_path)

    asca_only = {}
    gt = {}
    for i, r in enumerate(rows, 1):
        sid = str(i)
        asca_only[sid] = str(r.get("prediction", ""))
        gt[sid] = str(r.get("answer", ""))

    eligible_count = 0
    routed_count = 0
    switch_count = 0
    correct_switches = 0
    incorrect_switches = 0

    all_cf_pred = {}

    for i, r in enumerate(rows, 1):
        sid = str(i)
        base_pred = asca_only[sid]
        all_cf_pred[sid] = base_pred

        d = cf_diag.get(sid)
        if d is None:
            continue

        cand_list = d.get("candidate_list") or []
        logp_rel = d.get("logp_rel") or {}
        logp_ctrl = d.get("logp_ctrl") or {}

        if len(cand_list) < 2:
            continue
        if not isinstance(logp_rel, dict) or not isinstance(logp_ctrl, dict):
            continue

        # eligible: at least one candidate has both ctrl/rel logp
        score_map = {}
        raw_by_norm = {}
        for c in cand_list:
            n = normalize(c)
            if not n:
                continue
            raw_by_norm.setdefault(n, str(c))
            if n in logp_rel and n in logp_ctrl:
                try:
                    score_map[n] = float(logp_ctrl[n]) - float(logp_rel[n])
                except Exception:
                    pass

        if not score_map:
            continue

        eligible_count += 1
        routed_count += 1  # ungated: all eligible are treated as routed

        best_n = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)[0][0]
        new_pred = raw_by_norm.get(best_n, base_pred)
        all_cf_pred[sid] = new_pred

        if normalize(new_pred) != normalize(base_pred):
            switch_count += 1
            b_ok = is_correct(gt[sid], base_pred, r)
            n_ok = is_correct(gt[sid], new_pred, r)
            if n_ok and not b_ok:
                correct_switches += 1
            elif b_ok and not n_ok:
                incorrect_switches += 1

    n_total = len(rows)
    asca_correct = sum(1 for i, r in enumerate(rows, 1) if is_correct(gt[str(i)], asca_only[str(i)], r))
    cf_correct = sum(1 for i, r in enumerate(rows, 1) if is_correct(gt[str(i)], all_cf_pred[str(i)], r))
    asca_proxy_score = asca_correct / max(1, n_total)
    cf_proxy_score = cf_correct / max(1, n_total)

    if args.official_eval:
        ensure_unsloth_stub()
        build_dataset = import_build_dataset_lightweight()
        dataset = build_dataset(args.dataset_tag)

        asca_eval_xlsx = outdir / args.official_eval_inputs_subdir / args.benchmark / "asca_only" / f"asca_only_{args.dataset_tag}.xlsx"
        allcf_eval_xlsx = outdir / args.official_eval_inputs_subdir / args.benchmark / "all_cf_ungated" / f"all_cf_ungated_{args.dataset_tag}.xlsx"
        write_prediction_xlsx(src_headers, rows, asca_only, asca_eval_xlsx)
        write_prediction_xlsx(src_headers, rows, all_cf_pred, allcf_eval_xlsx)
        asca_ret = dataset.evaluate(str(asca_eval_xlsx))
        cf_ret = dataset.evaluate(str(allcf_eval_xlsx))
        asca_score = extract_official_score(asca_ret, args.benchmark)
        cf_score = extract_official_score(cf_ret, args.benchmark)
        asca_eval_input = str(asca_eval_xlsx)
        allcf_eval_input = str(allcf_eval_xlsx)
    else:
        asca_score = asca_proxy_score
        cf_score = cf_proxy_score
        asca_eval_input = ""
        allcf_eval_input = ""

    delta = None if (cf_score is None or asca_score is None) else (cf_score - asca_score)
    net_switch_gain = correct_switches - incorrect_switches

    out_csv = outdir / "ablation_groupB_allcf_ungated.csv"
    with out_csv.open("w", newline="") as f:
        fields = [
            "benchmark", "n", "variant", "final_score", "asca_only_score", "delta_vs_asca_only",
            "correct_count_proxy", "asca_only_correct_count_proxy",
            "eligible_count", "routed_count", "switch_count",
            "correct_switches", "incorrect_switches", "net_switch_gain",
            "proxy_final_score", "proxy_asca_only_score",
            "official_eval_used", "official_eval_input_asca", "official_eval_input_allcf",
            "source_xlsx", "source_cf_diag"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({
            "benchmark": args.benchmark,
            "n": n_total,
            "variant": "all_cf_ungated",
            "final_score": fmt_float(cf_score),
            "asca_only_score": fmt_float(asca_score),
            "delta_vs_asca_only": fmt_float(delta),
            "correct_count_proxy": cf_correct,
            "asca_only_correct_count_proxy": asca_correct,
            "eligible_count": eligible_count,
            "routed_count": routed_count,
            "switch_count": switch_count,
            "correct_switches": correct_switches,
            "incorrect_switches": incorrect_switches,
            "net_switch_gain": net_switch_gain,
            "proxy_final_score": fmt_float(cf_proxy_score),
            "proxy_asca_only_score": fmt_float(asca_proxy_score),
            "official_eval_used": int(args.official_eval),
            "official_eval_input_asca": asca_eval_input,
            "official_eval_input_allcf": allcf_eval_input,
            "source_xlsx": str(xlsx),
            "source_cf_diag": str(cf_diag_path),
        })

    pred_csv = outdir / "ablation_groupB_allcf_ungated_predictions.csv"
    with pred_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "asca_only_prediction", "all_cf_prediction", "switched"])
        w.writeheader()
        for i in range(1, n_total + 1):
            sid = str(i)
            b = asca_only[sid]
            c = all_cf_pred[sid]
            w.writerow({
                "sample_id": sid,
                "asca_only_prediction": b,
                "all_cf_prediction": c,
                "switched": int(normalize(b) != normalize(c)),
            })

    print(f"[DONE] GroupB ungated cf: {out_csv}")
    print(f"[DONE] GroupB predictions: {pred_csv}")


if __name__ == "__main__":
    main()
