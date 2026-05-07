#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common import write_csv, to_latex_table

_TIME_FMT = "%a %b %d %H:%M:%S %Z %Y"


def _parse_job_runtime(log_path: Path) -> Tuple[Optional[float], str]:
    if not log_path.exists():
        return None, "missing_log"
    txt = log_path.read_text(errors="ignore")
    m1 = re.search(r"Starting\s+(.+)", txt)
    m2 = re.search(r"Finished\s+(.+?):\s+", txt)
    if not (m1 and m2):
        return None, "no_start_finish"
    try:
        t1 = datetime.strptime(m1.group(1).strip(), _TIME_FMT)
        t2 = datetime.strptime(m2.group(1).strip(), _TIME_FMT)
    except Exception:
        return None, "parse_time_failed"
    sec = (t2 - t1).total_seconds()
    if sec < 0:
        return None, "negative_runtime"
    return sec, "ok"


def _find_latest_log_for_config(logs_dir: Path, config_stem: str) -> Optional[Path]:
    cands = sorted(logs_dir.glob(f"*{config_stem}*.out"), key=lambda p: p.stat().st_mtime)
    if cands:
        return cands[-1]
    return None


def _load_diag(path: Path) -> List[dict]:
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


def _diag_stats(rows: List[dict]) -> Dict[str, Optional[float]]:
    if not rows:
        return {
            "n": 0,
            "cf_used_rate": None,
            "pred_change_rate": None,
            "avg_logprob_calls": None,
        }
    n = len(rows)
    cf_used = sum(1 for r in rows if r.get("cf_used"))
    changed = sum(1 for r in rows if r.get("cf_final_changed", r.get("final_changed", False)))
    lp_calls = []
    for r in rows:
        lp_calls.append(
            len(r.get("logp_pos", {}) or {})
            + len(r.get("logp_rel", {}) or {})
            + len(r.get("logp_ctrl", {}) or {})
        )
    return {
        "n": n,
        "cf_used_rate": cf_used / n,
        "pred_change_rate": changed / n,
        "avg_logprob_calls": sum(lp_calls) / n,
    }


def main():
    ap = argparse.ArgumentParser(description="Compute/latency accounting from logs + diagnostics.")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()

    root = Path(args.repo_root)
    logs_dir = root / "logs"
    out_tables = root / "paper_neurips2026_artifacts" / "tables"
    out_reports = root / "paper_neurips2026_artifacts" / "reports"

    methods = [
        {
            "method": "Base_SmolVLM2",
            "config": "test_config_smolvlm2_base_textvqa",
            "diag": None,
            "gen_per_sample": 1.0,
            "notes": "single-view base inference",
        },
        {
            "method": "TTAug_deterministic",
            "config": "test_config_smolvlm2_paper_ttaug_classical_textvqa",
            "diag": None,
            "gen_per_sample": 8.0,
            "notes": "8 deterministic views",
        },
        {
            "method": "V91_NoCF",
            "config": "test_config_smolvlm2_v91_nocf_textvqa",
            "diag": ".runtime_cache/test_config_smolvlm2_v91_nocf_textvqa/diagnostics/v91nocf_samples.jsonl",
            "gen_per_sample": 8.0,
            "notes": "same 8-view generation, rerank-only",
        },
        {
            "method": "V91_CF3_Routed",
            "config": "test_config_smolvlm2_v91_cf3_routed_textvqa",
            "diag": ".runtime_cache/test_config_smolvlm2_v91_cf3_routed_textvqa/diagnostics/v91cf3_samples.jsonl",
            "gen_per_sample": 8.0,
            "notes": "optional sparse CF verifier",
        },
        {
            "method": "V91_CF3_ForceGrid",
            "config": "test_config_smolvlm2_v91_cf3_force_grid_textvqa",
            "diag": ".runtime_cache/test_config_smolvlm2_v91_cf3_force_grid_textvqa/diagnostics/v91cf3_samples.jsonl",
            "gen_per_sample": 8.0,
            "notes": "focused CF stress-test",
        },
        {
            "method": "CF3_NoQualityGate",
            "config": "test_config_smolvlm2_v91_cf3_no_quality_gate_textvqa",
            "diag": ".runtime_cache/test_config_smolvlm2_v91_cf3_no_quality_gate_textvqa/diagnostics/v91cf3_samples.jsonl",
            "gen_per_sample": 8.0,
            "notes": "negative ablation",
        },
        {
            "method": "CF3_ForceSwitch",
            "config": "test_config_smolvlm2_v91_cf3_force_switch_analysis_textvqa",
            "diag": ".runtime_cache/test_config_smolvlm2_v91_cf3_force_switch_analysis_textvqa/diagnostics/v91cf3_samples.jsonl",
            "gen_per_sample": 8.0,
            "notes": "negative ablation",
        },
    ]

    rows = []
    base_ref = None
    for m in methods:
        log = _find_latest_log_for_config(logs_dir, m["config"])
        runtime_sec, runtime_status = _parse_job_runtime(log) if log else (None, "missing_log")
        diag_rows = _load_diag(root / m["diag"]) if m["diag"] else []
        d = _diag_stats(diag_rows)

        rel_cost = None
        if runtime_sec is not None:
            if base_ref is None and m["method"] == "Base_SmolVLM2":
                base_ref = runtime_sec
            if base_ref:
                rel_cost = runtime_sec / base_ref

        rows.append({
            "Method": m["method"],
            "Generations/sample": f"{m['gen_per_sample']:.2f}",
            "Logprob calls/sample": "NA" if d["avg_logprob_calls"] is None else f"{d['avg_logprob_calls']:.2f}",
            "CF usage": "NA" if d["cf_used_rate"] is None else f"{d['cf_used_rate']:.4f}",
            "Prediction change rate": "NA" if d["pred_change_rate"] is None else f"{d['pred_change_rate']:.4f}",
            "Runtime_sec": "NA" if runtime_sec is None else f"{runtime_sec:.2f}",
            "Runtime_status": runtime_status,
            "Relative cost vs base": "NA" if rel_cost is None else f"{rel_cost:.2f}",
            "Log source": str(log) if log else "NA",
            "Notes": m["notes"],
        })

    fields = [
        "Method", "Generations/sample", "Logprob calls/sample", "CF usage",
        "Prediction change rate", "Runtime_sec", "Runtime_status", "Relative cost vs base", "Log source", "Notes"
    ]
    write_csv(out_tables / "compute_cost_table.csv", rows, fields)

    to_latex_table(
        out_tables / "compute_cost_table.tex",
        ["Method", "Generations/sample", "Logprob calls/sample", "CF usage", "Prediction change", "Relative cost", "Notes"],
        [[r["Method"], r["Generations/sample"], r["Logprob calls/sample"], r["CF usage"], r["Prediction change rate"], r["Relative cost vs base"], r["Notes"]] for r in rows],
        caption="Compute and latency accounting (measured where logs allow; otherwise deterministic call-count estimates).",
        label="tab:compute_cost",
    )

    rep = []
    rep.append("# Compute Cost Report")
    rep.append("")
    rep.append("This report combines measured wall-clock runtime from start/finish log lines with deterministic call-count estimates.")
    rep.append("")
    rep.append("- AS-TTA (V91-NoCF) uses the same 8 deterministic views as TTAug; overhead is in aggregation/rerank logic.")
    rep.append("- Optional CF verifier adds candidate log-likelihood evaluations only on routed subsets (see CF usage and logprob calls/sample).")
    rep.append("")
    rep.append("## Output")
    rep.append(f"- `{out_tables / 'compute_cost_table.csv'}`")
    rep.append(f"- `{out_tables / 'compute_cost_table.tex'}`")
    (out_reports / "compute_cost_report.md").write_text("\n".join(rep) + "\n")

    print(out_tables / "compute_cost_table.csv")


if __name__ == "__main__":
    main()
