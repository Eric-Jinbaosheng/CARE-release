#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import string
from pathlib import Path

from common import read_table_file


DEFAULT = dict(w_sup=2.0, w_valid=1.0, w_base=0.4, w_risk=0.5)
WSUP_POINTS = [1.0, 1.5, 2.0, 2.5, 3.0]
WVALID_POINTS = [0.25, 0.5, 1.0, 1.5, 2.0]
PUNCT = str.maketrans("", "", string.punctuation)
NUMERIC_RE = re.compile(
    r"^[\$£€¥]?[+-]?\d+(?:[\.,]\d+)*(?:%|°|usd|gb|kg|mg|cm|mm|km|kmh|hp|ml|l|kw|kwh)?\.?$",
    flags=re.IGNORECASE,
)
YES_NO = {"yes", "no", "true", "false", "yeah", "nope"}


def norm_text(s):
    return str(s or "").strip().lower().translate(PUNCT).strip()


def normalize_answer(s):
    return norm_text(s)


def answer_length(s):
    return len(normalize_answer(s).split())


def is_numeric(s):
    s = normalize_answer(s)
    if not s:
        return False
    return bool(NUMERIC_RE.match(s))


def is_yes_no(s):
    return normalize_answer(s) in YES_NO


def is_alnum_short(s):
    n = normalize_answer(s)
    if not n or len(n) > 32:
        return False
    return bool(re.match(r"^[a-z0-9 \-_/\.]+$", n))


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


def is_correct(gt, pred):
    g = str(gt or "")
    p = str(pred or "")
    if g.startswith("[") and g.endswith("]"):
        try:
            import ast

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


def quick_settings():
    out = []
    seen = set()

    def add(w, varied, val):
        name = setting_name(w)
        if name in seen:
            return
        seen.add(name)
        out.append({"setting_name": name, "varied_param": varied, "param_value": val, **w})

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


def load_diag_map(diag_path: Path):
    out = {}
    with diag_path.open() as f:
        for ln in f:
            if not ln.strip():
                continue
            obj = json.loads(ln)
            sid = str(obj.get("sample_id", ""))
            if sid:
                out[sid] = obj
    return out


def make_features(diag):
    candidate_list = diag.get("candidate_list") or []
    scored_top = diag.get("scored_top") or []
    answer_space = str(diag.get("answer_space", "unknown"))
    base = str(diag.get("base_answer", ""))
    base_norm = normalize_answer(base)
    support = {}
    for item in scored_top:
        if isinstance(item, list) and len(item) >= 5:
            n = normalize_answer(item[0])
            if n:
                try:
                    support[n] = float(item[4])
                except Exception:
                    pass
    feats = {}
    for raw in candidate_list:
        n = normalize_answer(raw)
        if not n or n in feats:
            continue
        feats[n] = {
            "raw": raw,
            "view_freq": float(support.get(n, 0.0)),
            "is_base": n == base_norm,
            "is_numeric": is_numeric(n),
            "is_yes_no": is_yes_no(n),
            "is_alnum_short": is_alnum_short(n),
            "length_words": answer_length(n),
        }
    return feats, answer_space, base_norm


def rerank(diag, w):
    feats, answer_space, base_norm = make_features(diag)
    if not feats:
        return str(diag.get("final_answer") or diag.get("base_answer") or "")
    base_len = feats.get(base_norm, {}).get("length_words", 0)
    best = None
    best_key = None
    for n_ans, f in feats.items():
        fv = format_validity(n_ans, feats, answer_space)
        lr = length_risk(n_ans, feats, answer_space, base_len)
        s = (
            float(w["w_sup"]) * float(f["view_freq"])
            + float(w["w_valid"]) * float(fv)
            + (float(w["w_base"]) if f["is_base"] else 0.0)
            - float(w["w_risk"]) * float(lr)
        )
        key = (s, f["view_freq"], fv, 1 if f["is_base"] else 0, -answer_length(n_ans))
        if best_key is None or key > best_key:
            best_key = key
            best = f["raw"]
    return best


def main():
    ap = argparse.ArgumentParser(description="Second-backbone ASCA rerank-only sensitivity (uses diagnostics jsonl).")
    ap.add_argument("--repo_root", default="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean")
    ap.add_argument("--backbone", default="ovis2_2b", choices=["ovis2_2b", "internvl2_5_2b"])
    ap.add_argument("--benchmarks", default="textvqa,ocrvqa,chartqa,gqa")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    model_name = "ASCA_Ovis2_2B" if args.backbone == "ovis2_2b" else "ASCA_InternVL2_5_2B"
    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    settings = quick_settings()

    raw_rows = []
    changed_rows = []
    missing_rows = []

    for bench in benches:
        result_dir = repo_root / "benchmark_results" / f"n_samples_{args.n}" / f"test_config_{args.backbone}_asca_{bench}" / model_name
        cands = sorted(result_dir.glob("**/*.xlsx"))
        default_xlsx = cands[-1] if cands else None
        # prefer task output xlsx not aux/openai/score
        if default_xlsx is not None and (
            "_score.xlsx" in default_xlsx.name
            or "openai_result" in default_xlsx.name
            or "auxmatch" in default_xlsx.name
        ):
            cands = sorted(
                p
                for p in result_dir.glob("**/*.xlsx")
                if "_score.xlsx" not in p.name and "openai_result" not in p.name and "auxmatch" not in p.name
            )
            default_xlsx = cands[-1] if cands else None
        diag_path = repo_root / ".runtime_cache" / f"test_config_{args.backbone}_asca_{bench}_n{args.n}" / "diagnostics" / "asca_samples.jsonl"

        if default_xlsx is None or not default_xlsx.exists() or not diag_path.exists():
            missing_rows.append(
                {
                    "benchmark": bench,
                    "default_xlsx": str(default_xlsx) if default_xlsx else "NA",
                    "diag_path": str(diag_path),
                    "reason": "missing_default_xlsx_or_diag",
                }
            )
            continue

        rows = read_table_file(default_xlsx)
        diag_map = load_diag_map(diag_path)
        default_pred = {}
        gt_map = {}
        for i, r in enumerate(rows, 1):
            sid = str(i)
            default_pred[sid] = str(r.get("prediction", ""))
            gt_map[sid] = str(r.get("answer", ""))

        # default score
        d_correct = 0
        for sid, dp in default_pred.items():
            if is_correct(gt_map[sid], dp):
                d_correct += 1
        d_score = d_correct / max(1, len(default_pred))

        for st in settings:
            st_name = st["setting_name"]
            pred_dir = out_root / bench / st_name
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_path = pred_dir / f"{bench}_predictions.csv"

            s_preds = {}
            changed = 0
            default_wins = 0
            setting_wins = 0
            correct = 0
            missing_diag = 0

            for sid, dp in default_pred.items():
                diag = diag_map.get(sid)
                if diag is None:
                    missing_diag += 1
                    sp = dp
                else:
                    sp = rerank(diag, st)
                s_preds[sid] = sp
                if norm_text(sp) != norm_text(dp):
                    changed += 1
                d_ok = is_correct(gt_map[sid], dp)
                s_ok = is_correct(gt_map[sid], sp)
                if d_ok and not s_ok:
                    default_wins += 1
                elif s_ok and not d_ok:
                    setting_wins += 1
                if s_ok:
                    correct += 1

            score = correct / max(1, len(default_pred))
            delta = score - d_score
            net = setting_wins - default_wins

            with pred_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["sample_id", "prediction", "default_prediction", "changed_vs_default"])
                w.writeheader()
                for sid in default_pred.keys():
                    w.writerow(
                        {
                            "sample_id": sid,
                            "prediction": s_preds[sid],
                            "default_prediction": default_pred[sid],
                            "changed_vs_default": int(norm_text(s_preds[sid]) != norm_text(default_pred[sid])),
                        }
                    )

            raw_rows.append(
                {
                    "backbone": args.backbone,
                    "benchmark": bench,
                    "n": len(default_pred),
                    "setting_name": st_name,
                    "varied_param": st["varied_param"],
                    "param_value": st["param_value"],
                    "w_sup": st["w_sup"],
                    "w_valid": st["w_valid"],
                    "w_base": st["w_base"],
                    "w_risk": st["w_risk"],
                    "metric": "accuracy_proxy",
                    "score": f"{score:.6f}",
                    "default_score": f"{d_score:.6f}",
                    "delta_vs_default": f"{delta:.6f}",
                    "changed_vs_default": changed,
                    "default_wins": default_wins,
                    "setting_wins": setting_wins,
                    "net_vs_default": net,
                    "missing_diag_samples": missing_diag,
                    "diag_path": str(diag_path),
                    "default_xlsx": str(default_xlsx),
                    "setting_output_file": str(pred_path),
                }
            )
            changed_rows.append(
                {
                    "backbone": args.backbone,
                    "benchmark": bench,
                    "setting_name": st_name,
                    "changed": changed,
                    "default_wins": default_wins,
                    "setting_wins": setting_wins,
                    "net": net,
                }
            )

    raw_csv = out_root / "sensitivity_raw.csv"
    with raw_csv.open("w", newline="") as f:
        fn = [
            "backbone",
            "benchmark",
            "n",
            "setting_name",
            "varied_param",
            "param_value",
            "w_sup",
            "w_valid",
            "w_base",
            "w_risk",
            "metric",
            "score",
            "default_score",
            "delta_vs_default",
            "changed_vs_default",
            "default_wins",
            "setting_wins",
            "net_vs_default",
            "missing_diag_samples",
            "diag_path",
            "default_xlsx",
            "setting_output_file",
        ]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(raw_rows)

    changed_csv = out_root / "sensitivity_changed_cases.csv"
    with changed_csv.open("w", newline="") as f:
        fn = ["backbone", "benchmark", "setting_name", "changed", "default_wins", "setting_wins", "net"]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(changed_rows)

    miss_csv = out_root / "sensitivity_missing_inputs.csv"
    with miss_csv.open("w", newline="") as f:
        fn = ["benchmark", "default_xlsx", "diag_path", "reason"]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(missing_rows)

    # simple by-setting aggregate
    by_setting = {}
    for r in raw_rows:
        k = r["setting_name"]
        by_setting.setdefault(k, {"sum": 0.0, "n": 0, "improved": 0, "hurt": 0, "worst": 0.0})
        d = float(r["delta_vs_default"])
        by_setting[k]["sum"] += d
        by_setting[k]["n"] += 1
        if d > 0:
            by_setting[k]["improved"] += 1
        elif d < 0:
            by_setting[k]["hurt"] += 1
        by_setting[k]["worst"] = min(by_setting[k]["worst"], d)

    agg_csv = out_root / "sensitivity_summary_by_setting.csv"
    with agg_csv.open("w", newline="") as f:
        fn = ["setting_name", "avg_delta", "benchmarks_covered", "n_improved", "n_hurt", "worst_delta"]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for k in sorted(by_setting):
            v = by_setting[k]
            w.writerow(
                {
                    "setting_name": k,
                    "avg_delta": f"{(v['sum']/max(1,v['n'])):.6f}",
                    "benchmarks_covered": v["n"],
                    "n_improved": v["improved"],
                    "n_hurt": v["hurt"],
                    "worst_delta": f"{v['worst']:.6f}",
                }
            )

    print(f"[DONE] wrote: {raw_csv}")
    print(f"[DONE] wrote: {agg_csv}")
    print(f"[DONE] wrote: {changed_csv}")
    print(f"[DONE] wrote: {miss_csv}")


if __name__ == "__main__":
    main()
