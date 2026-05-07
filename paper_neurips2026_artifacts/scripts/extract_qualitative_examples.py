#!/usr/bin/env python3
import argparse
import ast
import json
import re
import shutil
import string
from pathlib import Path
from typing import Dict, List, Optional

from common import read_table_file

_PUNCT = str.maketrans("", "", string.punctuation)


def norm(s: str) -> str:
    return str(s or "").strip().lower().translate(_PUNCT).strip()


def is_correct(gt, pred):
    if pred is None:
        return False
    g = str(gt or "")
    p = norm(pred)
    if g.startswith("[") and g.endswith("]"):
        try:
            vals = ast.literal_eval(g)
            if isinstance(vals, list):
                return p in {norm(v) for v in vals}
        except Exception:
            pass
    return p == norm(g)


def load_xlsx(path: Path) -> Dict[str, dict]:
    rows = read_table_file(path)
    out = {}
    for r in rows:
        sid = str(r.get("index", r.get("id", r.get("question_id", r.get("image_id", "")))))
        if sid:
            out[sid] = r
    return out


def load_diag(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        sid = str(obj.get("sample_id", ""))
        if sid:
            out[sid] = obj
    return out


def pick_examples(base, nocf, cf, ungated, diag_nocf, diag_cf, diag_ungated):
    ex = {
        "nocf_rescue": [],
        "nocf_harm": [],
        "cf_rescue": [],
        "cf_harm": [],
        "ungated_catastrophic": [],
        "format_rescue": [],
        "length_risk_rescue": [],
    }

    common = sorted(set(base) & set(nocf) & set(cf) & set(ungated))
    for sid in common:
        b = base[sid]
        n = nocf[sid]
        c = cf[sid]
        u = ungated[sid]
        gt = n.get("answer", b.get("answer", ""))
        cb = is_correct(gt, b.get("prediction"))
        cn = is_correct(gt, n.get("prediction"))
        cc = is_correct(gt, c.get("prediction"))
        cu = is_correct(gt, u.get("prediction"))

        if cn and not cb:
            ex["nocf_rescue"].append((sid, b, n, c, u))
        if cb and not cn:
            ex["nocf_harm"].append((sid, b, n, c, u))
        if cc and not cn:
            ex["cf_rescue"].append((sid, b, n, c, u))
        if cn and not cc:
            ex["cf_harm"].append((sid, b, n, c, u))
        if cn and not cu:
            ex["ungated_catastrophic"].append((sid, b, n, c, u))

        dn = diag_nocf.get(sid, {})
        dc = diag_cf.get(sid, {})
        du = diag_ungated.get(sid, {})

        # heuristics for format/length rescue from NoCF diagnostics
        if cn and not cb:
            top = dn.get("scored_top", [])
            if len(top) >= 2:
                # scored_top items: [cand, score, general, cf, view_freq]
                # try if chosen has higher fmt than runner-up by reading candidate_map in debug if present
                if dn.get("answer_space") in {"yes_no", "multiple_choice", "numeric", "ocr_text_short"}:
                    ex["format_rescue"].append((sid, b, n, c, u))
                if dn.get("answer_space") in {"open_entity", "ocr_text_short"}:
                    ex["length_risk_rescue"].append((sid, b, n, c, u))

    # keep small curated sets
    for k in ex:
        ex[k] = ex[k][:10]
    return ex


def main():
    ap = argparse.ArgumentParser(description="Extract qualitative examples for NoCF/CF behavior.")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()

    root = Path(args.repo_root)
    out_reports = root / "paper_neurips2026_artifacts" / "reports"
    out_tables = root / "paper_neurips2026_artifacts" / "tables"
    out_fig = root / "paper_neurips2026_artifacts" / "figures" / "example_assets"
    out_fig.mkdir(parents=True, exist_ok=True)

    # Use TextVQA where all needed variants are available
    base_x = root / "benchmark_results/n_samples_1000/test_config_smolvlm2_paper_ttaug_classical_textvqa/TTAugClassical_SmolVLM2_2B/TTAugClassical_SmolVLM2_2B_TextVQA_VAL.xlsx"
    nocf_x = root / "benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_textvqa/V91NoCF_SmolVLM2_2B/V91NoCF_SmolVLM2_2B_TextVQA_VAL.xlsx"
    cf_x = root / "benchmark_results/n_samples_1000/test_config_smolvlm2_v91_cf3_force_grid_textvqa/V91CF3ForceGrid_SmolVLM2_2B/V91CF3ForceGrid_SmolVLM2_2B_TextVQA_VAL.xlsx"
    ungated_x = root / "benchmark_results/n_samples_200/test_config_smolvlm2_v91_cf3_no_quality_gate_textvqa/V91CF3NoQualityGate_SmolVLM2_2B/T20260428_G8433322c/V91CF3NoQualityGate_SmolVLM2_2B_TextVQA_VAL.xlsx"

    base = load_xlsx(base_x)
    nocf = load_xlsx(nocf_x)
    cf = load_xlsx(cf_x)
    ungated = load_xlsx(ungated_x)

    diag_nocf = load_diag(root / ".runtime_cache/test_config_smolvlm2_v91_nocf_textvqa/diagnostics/v91nocf_samples.jsonl")
    diag_cf = load_diag(root / ".runtime_cache/test_config_smolvlm2_v91_cf3_force_grid_textvqa/diagnostics/v91cf3_samples.jsonl")
    diag_ungated = load_diag(root / ".runtime_cache/test_config_smolvlm2_v91_cf3_no_quality_gate_textvqa/diagnostics/v91cf3_samples.jsonl")

    ex = pick_examples(base, nocf, cf, ungated, diag_nocf, diag_cf, diag_ungated)

    md = []
    md.append("# Qualitative Examples")
    md.append("")
    md.append("All examples are from TextVQA. Correctness is proxy (normalized string/list match).")

    compact_rows = []

    for k, items in ex.items():
        md.append("")
        md.append(f"## {k}")
        md.append("")
        if not items:
            md.append("- none found")
            continue
        for sid, b, n, c, u in items:
            q = n.get("question", "")
            gt = n.get("answer", "")
            bp = b.get("prediction", "")
            np = n.get("prediction", "")
            cp = c.get("prediction", "")
            up = u.get("prediction", "")
            img = n.get("image_path", b.get("image_path", ""))
            md.append(f"- sample `{sid}`")
            md.append(f"  - question: {q}")
            md.append(f"  - gt: {gt}")
            md.append(f"  - baseline: {bp}")
            md.append(f"  - nocf: {np}")
            md.append(f"  - cf_force_grid: {cp}")
            md.append(f"  - cf_no_quality_gate: {up}")
            md.append(f"  - image_path: {img}")

            compact_rows.append({
                "Type": k,
                "SampleID": sid,
                "Question": q[:120],
                "GT": str(gt)[:80],
                "Baseline": str(bp)[:40],
                "NoCF": str(np)[:40],
                "CF": str(cp)[:40],
            })

            # copy/symlink image only if path exists
            try:
                p = Path(str(img))
                if p.exists():
                    target = out_fig / f"{sid}{p.suffix}"
                    if not target.exists():
                        target.symlink_to(p)
            except Exception:
                pass

    (out_reports / "qualitative_examples.md").write_text("\n".join(md) + "\n")

    # compact tex/csv
    import csv
    fields = ["Type", "SampleID", "Question", "GT", "Baseline", "NoCF", "CF"]
    with open(out_tables / "qualitative_examples_compact.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in compact_rows:
            w.writerow(r)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{llp{4.2cm}p{2.1cm}p{2.1cm}p{2.1cm}p{2.1cm}}")
    lines.append("\\toprule")
    lines.append("Type & ID & Question & GT & Baseline & NoCF & CF \\\\")
    lines.append("\\midrule")
    for r in compact_rows[:18]:
        def esc(x):
            return str(x).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
        lines.append(
            f"{esc(r['Type'])} & {esc(r['SampleID'])} & {esc(r['Question'])} & {esc(r['GT'])} & {esc(r['Baseline'])} & {esc(r['NoCF'])} & {esc(r['CF'])} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Qualitative examples (text-only) for NoCF and CF behaviors.}")
    lines.append("\\label{tab:qual_examples}")
    lines.append("\\end{table}")
    (out_tables / "qualitative_examples_compact.tex").write_text("\n".join(lines) + "\n")

    print(out_reports / "qualitative_examples.md")


if __name__ == "__main__":
    main()
