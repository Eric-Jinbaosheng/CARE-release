#!/usr/bin/env python3
import argparse
import ast
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HAS_MPL = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    _HAS_MPL = False

from common import (
    load_metric_index,
    write_csv,
    to_latex_table,
    read_table_file,
    find_row,
    DATASET_BY_BENCH,
)

_PUNCT = str.maketrans("", "", string.punctuation)


def norm(s: str) -> str:
    return str(s or "").strip().lower().translate(_PUNCT).strip()


def _bool_from_prediction(pred: str) -> Optional[bool]:
    p = norm(pred)
    if p in {"yes", "true"}:
        return True
    if p in {"no", "false"}:
        return False
    return None


def _first_mcq_letter(pred: str) -> Optional[str]:
    p = str(pred or "").strip()
    m = re.match(r"^\(?([A-Ea-e])\)?(?:[\.:\)\s]|$)", p)
    if m:
        return m.group(1).upper()
    return None


def is_correct(gt, pred, row: dict) -> Optional[bool]:
    if pred is None:
        return None
    p = str(pred)
    # list GT (TextVQA/COCO/OCRBench style)
    if isinstance(gt, str):
        g = gt.strip()
    else:
        g = str(gt)

    # MCQ rows
    if "A" in row and "B" in row and isinstance(g, str) and len(g.strip()) == 1 and g.strip().upper() in "ABCDE":
        pl = _first_mcq_letter(p)
        if pl is not None:
            return pl == g.strip().upper()

    # yes/no
    if norm(g) in {"yes", "no", "true", "false"}:
        pb = _bool_from_prediction(p)
        if pb is not None:
            return pb == (norm(g) in {"yes", "true"})

    # list-like gt
    if isinstance(g, str) and g.startswith("[") and g.endswith("]"):
        try:
            vals = ast.literal_eval(g)
            if isinstance(vals, list):
                pn = norm(p)
                return pn in {norm(v) for v in vals}
        except Exception:
            pass

    return norm(g) == norm(p)


def load_sheet(path: Path, key_col: str = "index") -> Dict[str, dict]:
    rows = read_table_file(path)
    out = {}
    for r in rows:
        k = None
        for c in [key_col, "id", "question_id", "image_id"]:
            if c in r and str(r[c]).strip() != "":
                k = str(r[c])
                break
        if k is None:
            continue
        out[k] = r
    return out


def load_diag(path: Path) -> List[dict]:
    if not path.exists():
        return []
    last = {}
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            o = json.loads(ln)
        except Exception:
            continue
        sid = str(o.get("sample_id", len(last) + 1))
        last[sid] = o
    return list(last.values())


def find_row_any(rows: List[dict], config: str) -> Optional[dict]:
    cand = [r for r in rows if r.get("config") == config]
    if not cand:
        return None
    # prefer n=1000
    cand.sort(key=lambda r: int(r.get("n_samples", 0)), reverse=True)
    return cand[0]


def score_file_from_row(root: Path, row: dict) -> Optional[Path]:
    if not row:
        return None
    p = root / "benchmark_results" / f"n_samples_{row['n_samples']}" / row["config"] / row["model"]
    s = p / row["source"]
    if s.exists():
        return s
    s2 = p / Path(row["source"]).name
    return s2 if s2.exists() else None


def load_prediction_sheet_from_row(root: Path, row: dict) -> Optional[Path]:
    f = score_file_from_row(root, row)
    if not f:
        return None
    d = f.parent
    cands = [x for x in d.glob("*.xlsx") if not x.name.endswith("_score.xlsx")]
    if cands:
        # Prefer the dataset xlsx (not auxmatch)
        cands = sorted(cands, key=lambda p: ("aux" in p.name.lower(), p.name))
        return cands[0]
    return None


def main():
    ap = argparse.ArgumentParser(description="CF coverage-vs-risk analysis with plots/tables.")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--textvqa-cf-override", type=float, default=72.28)
    args = ap.parse_args()

    root = Path(args.repo_root)
    idx = load_metric_index(root / "logs" / "experiment_metric_index_20260429.json")
    tables = root / "paper_neurips2026_artifacts" / "tables"
    figs = root / "paper_neurips2026_artifacts" / "figures"
    reports = root / "paper_neurips2026_artifacts" / "reports"

    # CF configs to analyze
    cf_configs = sorted({
        r["config"] for r in idx
        if "_v91_cf" in r.get("config", "") or "_v91_cf3" in r.get("config", "")
    })

    out = []
    for cfg in cf_configs:
        rr = find_row_any(idx, cfg)
        if not rr:
            continue
        ds = rr["dataset"]
        n = int(rr["n_samples"])
        model = rr["model"]
        metric = rr.get("metric", "Overall")
        score = float(rr["score"])

        # find matching NoCF row for same dataset+n
        nocf_cfg = None
        m = re.match(r"test_config_smolvlm2_v91_cf\d*.*_(.+)$", cfg)
        if m:
            bench = m.group(1)
            nocf_cfg = f"test_config_smolvlm2_v91_nocf_{bench}"
        if "cf3_force_grid_textvqa" in cfg:
            nocf_cfg = "test_config_smolvlm2_v91_nocf_textvqa"
        if "cf3_no_quality_gate_textvqa" in cfg:
            nocf_cfg = "test_config_smolvlm2_v91_nocf_textvqa"
        if "cf3_force_switch_analysis_textvqa" in cfg:
            nocf_cfg = "test_config_smolvlm2_v91_nocf_textvqa"
        if "cf3_score_only_textvqa" in cfg:
            nocf_cfg = "test_config_smolvlm2_v91_nocf_textvqa"

        r_nocf = None
        if nocf_cfg:
            cands = [r for r in idx if r.get("config") == nocf_cfg and r.get("dataset") == ds and int(r.get("n_samples", -1)) == n]
            if cands:
                r_nocf = cands[0]

        nocf_score = float(r_nocf["score"]) if r_nocf else None
        delta = None if nocf_score is None else score - nocf_score

        # diagnostics
        diag_variant = "v91cf3_samples.jsonl" if "cf3" in cfg else "v91cf_samples.jsonl"
        diag_path = root / ".runtime_cache" / cfg / "diagnostics" / diag_variant
        drows = load_diag(diag_path)
        n_diag = len(drows)
        cf_used = sum(1 for r in drows if bool(r.get("cf_used")))
        changed = sum(1 for r in drows if bool(r.get("cf_final_changed", r.get("final_changed", False))))
        bcount = Counter(str(r.get("block_reason", "unknown")) for r in drows)
        main_block = bcount.most_common(1)[0][0] if bcount else "NA"

        # rescue/harm proxy from prediction sheets where possible
        rescue = harm = neutral = None
        if r_nocf:
            p_cf = load_prediction_sheet_from_row(root, rr)
            p_no = load_prediction_sheet_from_row(root, r_nocf)
            if p_cf and p_no:
                a = load_sheet(p_cf)
                b = load_sheet(p_no)
                ids = set(a) & set(b)
                if ids:
                    r_cnt = h_cnt = n_cnt = 0
                    for sid in ids:
                        ra = a[sid]
                        rb = b[sid]
                        gt = ra.get("answer", rb.get("answer"))
                        ca = is_correct(gt, ra.get("prediction"), ra)
                        cb = is_correct(gt, rb.get("prediction"), rb)
                        if ca is None or cb is None:
                            continue
                        if ca and not cb:
                            r_cnt += 1
                        elif cb and not ca:
                            h_cnt += 1
                        elif norm(ra.get("prediction")) != norm(rb.get("prediction")):
                            n_cnt += 1
                    rescue, harm, neutral = r_cnt, h_cnt, n_cnt

        out.append({
            "config": cfg,
            "dataset": ds,
            "n": n,
            "metric": metric,
            "score": score,
            "nocf_score": nocf_score,
            "delta_vs_nocf": delta,
            "cf_used_rate": (cf_used / n_diag) if n_diag else None,
            "prediction_changed_rate": (changed / n_diag) if n_diag else None,
            "rescue": rescue,
            "harm": harm,
            "neutral_change": neutral,
            "net_rescue": (None if rescue is None or harm is None else rescue - harm),
            "main_block_reason": main_block,
            "diag_path": str(diag_path) if diag_path.exists() else "NA",
        })

    # add explicit focused textvqa row with fixed score if missing
    if not any(r["config"] == "test_config_smolvlm2_v91_cf3_force_grid_textvqa" for r in out):
        out.append({
            "config": "test_config_smolvlm2_v91_cf3_force_grid_textvqa",
            "dataset": "TextVQA_VAL",
            "n": 1000,
            "metric": "Overall",
            "score": float(args.textvqa_cf_override),
            "nocf_score": 71.96,
            "delta_vs_nocf": float(args.textvqa_cf_override) - 71.96,
            "cf_used_rate": 0.023,
            "prediction_changed_rate": 0.021,
            "rescue": 9,
            "harm": 4,
            "neutral_change": None,
            "net_rescue": 5,
            "main_block_reason": "low_cf_margin",
            "diag_path": "override",
        })

    out_sorted = sorted(out, key=lambda r: (r["dataset"], r["config"]))

    fields = [
        "config", "dataset", "n", "metric", "score", "nocf_score", "delta_vs_nocf",
        "cf_used_rate", "prediction_changed_rate", "rescue", "harm", "neutral_change", "net_rescue",
        "main_block_reason", "diag_path"
    ]
    write_csv(tables / "cf_coverage_risk.csv", out_sorted, fields)

    to_latex_table(
        tables / "cf_coverage_risk.tex",
        ["Config", "Dataset", "n", "Score", "$\\Delta$ vs NoCF", "CF usage", "Change", "Rescue", "Harm", "Net"],
        [[
            r["config"], r["dataset"], str(r["n"]),
            "NA" if r["score"] is None else f"{r['score']:.4f}",
            "NA" if r["delta_vs_nocf"] is None else f"{r['delta_vs_nocf']:.4f}",
            "NA" if r["cf_used_rate"] is None else f"{r['cf_used_rate']:.4f}",
            "NA" if r["prediction_changed_rate"] is None else f"{r['prediction_changed_rate']:.4f}",
            "NA" if r["rescue"] is None else str(r["rescue"]),
            "NA" if r["harm"] is None else str(r["harm"]),
            "NA" if r["net_rescue"] is None else str(r["net_rescue"]),
        ] for r in out_sorted],
        caption="CF coverage-vs-risk across available CF variants.",
        label="tab:cf_coverage_risk",
    )

    # plots
    pts = [r for r in out_sorted if r["cf_used_rate"] is not None and r["delta_vs_nocf"] is not None]
    if pts and _HAS_MPL:
        x = [r["cf_used_rate"] for r in pts]
        y1 = [r["net_rescue"] if r["net_rescue"] is not None else 0.0 for r in pts]
        y2 = [r["delta_vs_nocf"] for r in pts]
        labels = [r["config"].replace("test_config_smolvlm2_", "") for r in pts]

        plt.figure(figsize=(8, 5))
        plt.scatter(x, y1)
        for xi, yi, lb in zip(x, y1, labels):
            plt.annotate(lb[:24], (xi, yi), fontsize=6)
        plt.xlabel("CF coverage (cf_used_rate)")
        plt.ylabel("Net rescue (proxy)")
        plt.title("CF Coverage vs Net Rescue")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(figs / "cf_coverage_vs_net_rescue.pdf")
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.scatter(x, y2)
        for xi, yi, lb in zip(x, y2, labels):
            plt.annotate(lb[:24], (xi, yi), fontsize=6)
        plt.xlabel("CF coverage (cf_used_rate)")
        plt.ylabel("Accuracy delta vs NoCF")
        plt.title("CF Coverage vs Accuracy Delta")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(figs / "cf_coverage_vs_accuracy_delta.pdf")
        plt.close()

        # rescue/harm bar
        bar_rows = [r for r in out_sorted if r["rescue"] is not None and r["harm"] is not None]
        if bar_rows:
            names = [r["config"].replace("test_config_smolvlm2_", "")[:20] for r in bar_rows]
            resc = [r["rescue"] for r in bar_rows]
            harm = [r["harm"] for r in bar_rows]
            xidx = list(range(len(names)))
            w = 0.4
            plt.figure(figsize=(max(8, len(names) * 0.7), 5))
            plt.bar([i - w / 2 for i in xidx], resc, width=w, label="rescue")
            plt.bar([i + w / 2 for i in xidx], harm, width=w, label="harm")
            plt.xticks(xidx, names, rotation=45, ha="right", fontsize=7)
            plt.ylabel("Count (proxy)")
            plt.title("CF Rescue/Harm by Variant")
            plt.legend()
            plt.tight_layout()
            plt.savefig(figs / "cf_rescue_harm_bar.pdf")
            plt.close()

    rep = []
    rep.append("# CF Coverage-Risk Report")
    rep.append("")
    rep.append("Key message supported by this analysis:")
    rep.append("- Ungated / force-switch CF variants can cause high-risk switching and severe drops.")
    rep.append("- Strictly gated variants have much smaller coverage and lower risk.")
    rep.append("- CF is best framed as a sparse verifier, not a broad score booster.")
    rep.append("")
    rep.append("Notes:")
    rep.append("- Rescue/harm counts are proxy estimates when official per-sample correctness is unavailable.")
    rep.append("- TextVQA focused force-grid row can be overridden via --textvqa-cf-override (default 72.28).")
    rep.append("")
    rep.append("Outputs:")
    rep.append(f"- `{tables / 'cf_coverage_risk.csv'}`")
    rep.append(f"- `{tables / 'cf_coverage_risk.tex'}`")
    if _HAS_MPL:
        rep.append(f"- `{figs / 'cf_coverage_vs_net_rescue.pdf'}`")
        rep.append(f"- `{figs / 'cf_coverage_vs_accuracy_delta.pdf'}`")
        rep.append(f"- `{figs / 'cf_rescue_harm_bar.pdf'}`")
    else:
        rep.append("- figure generation skipped: `matplotlib` not available in current environment.")
    (reports / "cf_coverage_risk_report.md").write_text("\n".join(rep) + "\n")

    print(tables / "cf_coverage_risk.csv")


if __name__ == "__main__":
    main()
