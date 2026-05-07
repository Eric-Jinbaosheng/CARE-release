#!/usr/bin/env python3
import argparse
import json
import csv
from pathlib import Path


def status_for(path: Path):
    return "DONE" if path.exists() else "MISSING"


def _has_metric_row(index_rows, *, n_samples, config, dataset):
    for r in index_rows:
        if int(r.get("n_samples", -1)) == int(n_samples) and r.get("config") == config and r.get("dataset") == dataset:
            return True
    return False


def _ablation_has_values(csv_path: Path):
    if not csv_path.exists():
        return False
    with open(csv_path, newline="") as f:
        rr = csv.DictReader(f)
        for r in rr:
            if (r.get("AblationScore") or "").strip().upper() not in {"", "NA"}:
                return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Build final NeurIPS gap report from artifact files.")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()
    root = Path(args.repo_root)
    art = root / "paper_neurips2026_artifacts"

    idx_path = root / "logs" / "experiment_metric_index_20260429.json"
    index_rows = json.loads(idx_path.read_text()) if idx_path.exists() else []

    lines = []
    lines.append("# Final NeurIPS Gap Report")
    lines.append("")
    lines.append("| Item | Status | Evidence file | Action needed | Paper location |")
    lines.append("|---|---|---|---|---|")

    # main table
    p_main = art / "tables" / "main_results_n1000.tex"
    st_main = status_for(p_main)
    lines.append(f"| main 9-benchmark n=1000 table | {st_main} | `{p_main}` | {'None' if st_main=='DONE' else 'Regenerate table'} | Main results section |")

    # routed TextVQA n=1000
    routed_exists = _has_metric_row(
        index_rows,
        n_samples=1000,
        config="test_config_smolvlm2_v91_cf3_routed_textvqa",
        dataset="TextVQA_VAL",
    )
    p_routed = art / "reports" / "textvqa_routed_decision.md"
    if routed_exists:
        st_routed = "DONE"
        action_routed = "None"
    else:
        st_routed = "PARTIAL" if p_routed.exists() else "MISSING"
        action_routed = "Run routed TextVQA n=1000 or keep focused-force-grid caveat"
    lines.append(f"| TextVQA routed n=1000 | {st_routed} | `{p_routed}` | {action_routed} | CF section caveat |")

    # focused force-grid check
    p_fg = art / "tables" / "cf_textvqa_focused.tex"
    st_fg = status_for(p_fg)
    lines.append(f"| TextVQA force-grid focused check | {st_fg} | `{p_fg}` | {'None' if st_fg=='DONE' else 'Generate focused CF table'} | CF focused subsection |")

    # bad CF ablation
    p_bad = art / "tables" / "cf_bad_ablation.tex"
    st_bad = status_for(p_bad)
    lines.append(f"| no_quality_gate / force_switch negative ablation | {st_bad} | `{p_bad}` | {'None' if st_bad=='DONE' else 'Generate CF bad-ablation table'} | CF negative ablations |")

    # NoCF ablation component status depends on actual ablation values
    p_abl = art / "tables" / "nocf_ablation_n200.tex"
    p_abl_csv = art / "tables" / "nocf_ablation_n200.csv"
    if _ablation_has_values(p_abl_csv):
        st_abl = "DONE"
        action_abl = "None"
    else:
        st_abl = "PARTIAL" if p_abl.exists() else "MISSING"
        action_abl = "Run ablation jobs; current table is mostly NA"
    lines.append(f"| NoCF component ablations | {st_abl} | `{p_abl}` | {action_abl} | NoCF ablations |")

    # CI and compute
    p_ci = art / "tables" / "bootstrap_ci_main.tex"
    st_ci = status_for(p_ci)
    lines.append(f"| bootstrap CI | {st_ci} | `{p_ci}` | {'None' if st_ci=='DONE' else 'Run bootstrap script'} | Stats/robustness |")

    p_cost = art / "tables" / "compute_cost_table.tex"
    st_cost = status_for(p_cost)
    lines.append(f"| compute/latency table | {st_cost} | `{p_cost}` | {'None' if st_cost=='DONE' else 'Run compute-cost script'} | Compute section |")

    # qualitative
    p_qual = art / "reports" / "qualitative_examples.md"
    st_qual = status_for(p_qual)
    lines.append(f"| qualitative examples | {st_qual} | `{p_qual}` | {'None' if st_qual=='DONE' else 'Run qualitative extraction'} | Qualitative appendix |")

    # answer-space rules
    p_rules = art / "reports" / "answer_space_rules.md"
    st_rules = status_for(p_rules)
    lines.append(f"| answer-space rules | {st_rules} | `{p_rules}` | {'None' if st_rules=='DONE' else 'Export rules from code'} | Method appendix |")

    # second backbone: partial unless actual result rows exist for generated configs
    sb_rows = [r for r in index_rows if str(r.get("config", "")).startswith("test_config_ovis2_1b_ttaug_det_")]
    p_sb = art / "reports" / "second_backbone_feasibility.md"
    if sb_rows:
        st_sb = "DONE"
        act_sb = "None"
    else:
        st_sb = "PARTIAL" if p_sb.exists() else "MISSING"
        act_sb = "Run second-backbone n=200 jobs"
    lines.append(f"| second backbone | {st_sb} | `{p_sb}` | {act_sb} | Generalization appendix |")

    # reproducibility and supplement
    p_rep = art / "reproducibility" / "anonymize_for_neurips.sh"
    st_rep = status_for(p_rep)
    lines.append(f"| anonymous reproducibility package | {st_rep} | `{p_rep}` | {'None' if st_rep=='DONE' else 'Create anonymization script/package'} | Reproducibility |")

    p_chk = art / "supplement" / "limitations_checklist_notes.md"
    st_chk = status_for(p_chk)
    lines.append(f"| checklist notes | {st_chk} | `{p_chk}` | {'None' if st_chk=='DONE' else 'Write checklist notes'} | Checklist |")

    p_lim = art / "supplement" / "paper_patch_sections.tex"
    st_lim = status_for(p_lim)
    lines.append(f"| limitations section | {st_lim} | `{p_lim}` | {'None' if st_lim=='DONE' else 'Write limitations patch'} | Limitations |")

    # Mark ablation n1000 as partial if only config exists but table maybe mostly NA
    n1000_csv = art / "tables" / "nocf_ablation_n1000.csv"
    if n1000_csv.exists():
        lines.append("")
        lines.append("Note: n=1000 ablation table may contain NA rows until GPU jobs finish.")

    out = art / "reports" / "final_neurips_gap_report.md"
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
