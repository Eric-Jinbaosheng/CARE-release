#!/usr/bin/env python3
import argparse
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

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_MCQ_LETTER_RE = re.compile(r"^\(?([A-Ea-e])\)?(?:[\.:\)\s]|$)")
_YESNO_SET = {"yes", "no", "true", "false"}

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BENCH_CFG = {
    "textvqa": {
        "dataset_tag": "TextVQA_VAL",
        "full_cfg": "test_config_smolvlm2_v91_nocf_textvqa",
        "full_model": "V91NoCF_SmolVLM2_2B",
    },
    "chartqa": {
        "dataset_tag": "ChartQA_TEST",
        "full_cfg": "test_config_smolvlm2_v91_nocf_regen_chartqa",
        "full_model": "V91NoCF_SmolVLM2_2B",
    },
    "ocrvqa": {
        "dataset_tag": "OCRVQA_TEST",
        "full_cfg": "test_config_smolvlm2_v91_nocf_regen_ocrvqa",
        "full_model": "V91NoCF_SmolVLM2_2B",
    },
}


def norm_text(s):
    return str(s or "").strip().lower().translate(_PUNCT_TABLE).strip()


def normalize(s):
    return norm_text(s)


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


def load_source_rows(xlsx_path: Path):
    rows = read_table_file(xlsx_path)
    return rows


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
    # pick file with max number of lines
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


def build_norm2raw(diag):
    out = {}
    for c in (diag.get("candidate_list") or []):
        n = normalize(c)
        if n and n not in out:
            out[n] = str(c)
    return out


def support_map_from_scored_top(diag):
    sm = {}
    st = diag.get("scored_top") or []
    for e in st:
        if not isinstance(e, list) or len(e) < 2:
            continue
        n = normalize(e[0])
        if not n:
            continue
        sup = None
        if len(e) >= 5:
            try:
                sup = float(e[4])
            except Exception:
                sup = None
        if sup is None:
            try:
                sup = float(e[1])
            except Exception:
                sup = None
        if sup is not None:
            sm[n] = sup
    return sm


def pick_best_alt(diag):
    cf = diag.get("cf_score") or {}
    if not isinstance(cf, dict) or len(cf) < 2:
        return None
    nocf = normalize(diag.get("no_cf_winner", ""))
    if nocf not in cf:
        return None
    alts = [(k, float(v)) for k, v in cf.items() if k != nocf]
    if not alts:
        return None
    best_n, best_s = sorted(alts, key=lambda kv: kv[1], reverse=True)[0]
    delta = best_s - float(cf.get(nocf, 0.0))
    return {
        "nocf": nocf,
        "best": best_n,
        "best_score": best_s,
        "delta_cf": delta,
    }


def predict_gate(diag, asca_pred, mode):
    # mode: strict/default/loose
    # default: mirror existing routed pass
    if mode == "default":
        # Prefer routed run's recorded final answer when available.
        # This keeps default-gate sensitivity bit-consistent with the routed
        # experiment instead of re-deriving the gate from partial diagnostics.
        fa = str(diag.get("final_answer", "") or "").strip()
        if fa:
            return fa, normalize(fa) != normalize(asca_pred), "default_final_answer"
        if bool(diag.get("cf_used", False)) and str(diag.get("block_reason", "")) == "cf3_verifier_pass":
            cand = normalize(diag.get("cf_verifier_candidate", ""))
            n2r = build_norm2raw(diag)
            if cand and cand in n2r:
                return n2r[cand], True, "default_pass"
        return asca_pred, False, "default_block"

    alt = pick_best_alt(diag)
    if alt is None:
        return asca_pred, False, "no_alt"

    n2r = build_norm2raw(diag)
    alt_raw = n2r.get(alt["best"], asca_pred)
    gap = float(diag.get("cf_verifier_general_gap", 999.0) or 999.0)
    margin = float(diag.get("cf_margin", 0.0) or 0.0)
    mask_q = str(diag.get("mask_quality", "low"))
    ctrl_q = str(diag.get("control_quality", "low"))
    sup_map = support_map_from_scored_top(diag)
    sup = float(sup_map.get(alt["best"], 0.0))

    if mode == "strict":
        # stricter than default
        if alt["delta_cf"] > 0.20 and margin > 0.25 and gap <= 0.15 and sup >= 0.25 and mask_q in {"high", "medium"} and ctrl_q in {"matched_grid", "matched_periphery"}:
            return alt_raw, normalize(alt_raw) != normalize(asca_pred), "strict_pass"
        return asca_pred, False, "strict_block"

    if mode == "loose":
        # looser than default (still avoid totally random switches)
        if alt["delta_cf"] > 0.0 and gap <= 0.50 and sup >= 0.10:
            return alt_raw, normalize(alt_raw) != normalize(asca_pred), "loose_pass"
        return asca_pred, False, "loose_block"

    return asca_pred, False, "unknown_mode"


def write_prediction_xlsx(rows, predictions, out_xlsx: Path):
    import pandas as pd
    if not rows:
        return
    cols = list(rows[0].keys())
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


def main():
    ap = argparse.ArgumentParser(description="CF gate sensitivity (strict/default/loose) from cached routed diagnostics.")
    ap.add_argument("--repo_root", default=str(REPO_ROOT))
    ap.add_argument("--benchmarks", default="chartqa,textvqa")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--output_dir", default="paper_neurips2026_artifacts/sensitivity_cf_gate")
    ap.add_argument("--official_eval", action="store_true", default=True)
    ap.add_argument(
        "--diag_override",
        default="",
        help="Optional comma-separated overrides: bench=/abs/path/v91cf3_samples.jsonl, e.g. textvqa=/path/a.jsonl,chartqa=/path/b.jsonl",
    )
    args = ap.parse_args()

    repo = Path(args.repo_root)
    outdir = repo / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.official_eval:
        ensure_unsloth_stub()
        build_dataset = import_build_dataset_lightweight()
    else:
        build_dataset = None

    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    modes = ["strict", "default", "loose"]
    diag_override = {}
    if args.diag_override:
        for item in str(args.diag_override).split(","):
            item = item.strip()
            if not item or "=" not in item:
                continue
            b, p = item.split("=", 1)
            b = b.strip().lower()
            p = p.strip()
            if b and p:
                diag_override[b] = Path(p)

    summary_rows = []

    for b in benches:
        if b not in BENCH_CFG:
            print(f"[WARN] skip unknown benchmark: {b}")
            continue
        cfg = BENCH_CFG[b]
        xlsx = discover_xlsx(repo, args.n, cfg["full_cfg"], cfg["full_model"], cfg["dataset_tag"])
        if xlsx is None:
            print(f"[WARN] {b}: missing full xlsx")
            continue
        diag_path = diag_override.get(b, None)
        if diag_path is not None and not diag_path.exists():
            print(f"[WARN] {b}: override diag missing: {diag_path}")
            diag_path = None
        if diag_path is None:
            diag_path = discover_cf_diag(repo, b)
        if diag_path is None:
            print(f"[WARN] {b}: missing routed cf diagnostics")
            continue

        rows = load_source_rows(xlsx)
        diag_map = load_diag_map(diag_path)
        if not rows:
            print(f"[WARN] {b}: empty source rows")
            continue

        asca_pred = {str(i): str(r.get("prediction", "")) for i, r in enumerate(rows, 1)}
        gt = {str(i): str(r.get("answer", "")) for i, r in enumerate(rows, 1)}

        for mode in modes:
            pred = {}
            eligible = 0
            routed = 0
            switched = 0
            cwin = 0
            closs = 0

            for i, r in enumerate(rows, 1):
                sid = str(i)
                base = asca_pred[sid]
                d = diag_map.get(sid)
                if d is None:
                    pred[sid] = base
                    continue

                if pick_best_alt(d) is not None:
                    eligible += 1

                outp, changed, reason = predict_gate(d, base, mode)
                pred[sid] = outp
                if reason.endswith("pass"):
                    routed += 1
                if changed:
                    switched += 1
                    b_ok = is_correct(gt[sid], base, r)
                    n_ok = is_correct(gt[sid], outp, r)
                    if n_ok and not b_ok:
                        cwin += 1
                    elif b_ok and not n_ok:
                        closs += 1

            correct = sum(1 for i, r in enumerate(rows, 1) if is_correct(gt[str(i)], pred[str(i)], r))
            asca_correct = sum(1 for i, r in enumerate(rows, 1) if is_correct(gt[str(i)], asca_pred[str(i)], r))
            proxy_score = correct / max(1, len(rows))
            proxy_asca = asca_correct / max(1, len(rows))

            if args.official_eval:
                ds = build_dataset(cfg["dataset_tag"])
                eval_xlsx = outdir / "official_eval_inputs" / b / mode / f"{mode}_{cfg['dataset_tag']}.xlsx"
                write_prediction_xlsx(rows, pred, eval_xlsx)
                eval_ret = ds.evaluate(str(eval_xlsx))
                off_score = extract_official_score(eval_ret)
            else:
                off_score = proxy_score

            summary_rows.append({
                "benchmark": b,
                "n": len(rows),
                "gate_setting": mode,
                "score": "NA" if off_score is None else f"{off_score:.6f}",
                "delta_vs_asca_only": f"{(proxy_score - proxy_asca):.6f}",
                "switch_coverage": f"{(switched / max(1, len(rows))):.6f}",
                "eligible_count": eligible,
                "routed_count": routed,
                "switch_count": switched,
                "correct_switches": cwin,
                "incorrect_switches": closs,
                "net_switch_gain": cwin - closs,
                "proxy_score": f"{proxy_score:.6f}",
                "proxy_asca_only": f"{proxy_asca:.6f}",
                "source_xlsx": str(xlsx),
                "source_cf_diag": str(diag_path),
            })

            # per-mode predictions dump
            pred_csv = outdir / f"{b}_{mode}_predictions.csv"
            with pred_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["sample_id", "prediction", "asca_only_prediction", "changed"])
                w.writeheader()
                for i in range(1, len(rows) + 1):
                    sid = str(i)
                    w.writerow({
                        "sample_id": sid,
                        "prediction": pred[sid],
                        "asca_only_prediction": asca_pred[sid],
                        "changed": int(normalize(pred[sid]) != normalize(asca_pred[sid])),
                    })

    out_csv = outdir / "cf_gate_sensitivity_summary.csv"
    with out_csv.open("w", newline="") as f:
        fields = [
            "benchmark", "n", "gate_setting", "score", "delta_vs_asca_only", "switch_coverage",
            "eligible_count", "routed_count", "switch_count", "correct_switches", "incorrect_switches",
            "net_switch_gain", "proxy_score", "proxy_asca_only", "source_xlsx", "source_cf_diag"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(f"[DONE] {out_csv}")


if __name__ == "__main__":
    main()
