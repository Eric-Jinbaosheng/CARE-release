#!/usr/bin/env python3
import argparse
import inspect
import json
import re
from collections import Counter
from pathlib import Path


def _extract_list_literal(src: str, var_name: str):
    m = re.search(rf"{re.escape(var_name)}\s*=\s*\[(.*?)\]\n", src, flags=re.S)
    if not m:
        return None
    body = m.group(1)
    vals = re.findall(r"'([^']+)'|\"([^\"]+)\"", body)
    out = []
    for a, b in vals:
        out.append(a or b)
    return out


def _extract_regex(src: str, var_name: str):
    m = re.search(rf"{re.escape(var_name)}\s*=\s*re\.compile\((.*?)\)\n", src, flags=re.S)
    if not m:
        return None
    return m.group(1).strip()


def _load_diag_counts(root: Path):
    out = {}
    for b in ["ocrbench", "gqa", "textvqa", "chartqa", "ocrvqa", "ai2d", "mme_rw", "coco", "amber"]:
        cfg = f"test_config_smolvlm2_v91_nocf_{b}"
        p = root / ".runtime_cache" / cfg / "diagnostics" / "v91nocf_samples.jsonl"
        c = Counter()
        if p.exists():
            last = {}
            for ln in p.read_text().splitlines():
                if not ln.strip():
                    continue
                try:
                    o = json.loads(ln)
                except Exception:
                    continue
                sid = str(o.get("sample_id", len(last) + 1))
                last[sid] = o
            for o in last.values():
                c[str(o.get("answer_space", "unknown"))] += 1
        out[b] = dict(c)
    return out


def main():
    ap = argparse.ArgumentParser(description="Export answer-space and normalization rules from V91 code.")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()

    root = Path(args.repo_root)
    code_path = root / "vlmeval" / "vlm" / "tta" / "tta_v91_aggregator.py"
    src = code_path.read_text()

    rules = {
        "source_file": str(code_path),
        "normalization": {
            "function": "normalize_answer",
            "behavior": "strip -> lowercase -> remove punctuation -> strip",
            "inferred_from": "normalize_answer()",
        },
        "regex_patterns": {
            "numeric": _extract_regex(src, "_NUMERIC_RE"),
            "date": _extract_regex(src, "_DATE_RE"),
            "mcq_letter": _extract_regex(src, "_MCQ_LETTER_RE"),
            "option_parse": _extract_regex(src, "_OPTION_RE"),
        },
        "keyword_patterns": {
            "ocr_intent": _extract_list_literal(src, "_OCR_INTENT_PATTERNS"),
            "chart_intent": _extract_list_literal(src, "_CHART_INTENT_PATTERNS"),
            "caption_intent": _extract_list_literal(src, "_CAPTION_INTENT_PATTERNS"),
        },
        "answer_spaces": [
            "yes_no", "multiple_choice", "numeric", "ocr_text_short",
            "open_entity", "chart_or_diagram", "caption_like", "unknown",
        ],
        "format_validity": "implemented in _format_validity()",
        "length_risk": "implemented in _length_risk()",
        "general_score": "score = 2.0*view_freq + 1.0*fmt_validity + 0.4*is_base - 0.5*length_risk (default weights)",
        "inferred_from": "score_candidates_general()",
    }

    # add benchmark-wise answer-space observed distributions
    rules["observed_answer_space_distribution_n1000_v91_nocf"] = _load_diag_counts(root)

    out_json = root / "paper_neurips2026_artifacts" / "reproducibility" / "answer_space_rules.json"
    out_json.write_text(json.dumps(rules, indent=2))

    md = []
    md.append("# Answer-Space Rules")
    md.append("")
    md.append(f"Source: `{code_path}`")
    md.append("")
    md.append("## 1) Normalization")
    md.append("- `normalize_answer`: lowercase + punctuation removal + strip.")
    md.append("")
    md.append("## 2) Classifier spaces")
    md.append("- yes_no")
    md.append("- multiple_choice")
    md.append("- numeric")
    md.append("- ocr_text_short")
    md.append("- open_entity")
    md.append("- chart_or_diagram")
    md.append("- caption_like")
    md.append("- unknown")
    md.append("")
    md.append("## 3) Trigger heuristics")
    md.append("- OCR intent keywords and alphanumeric-short candidates route to `ocr_text_short`.")
    md.append("- chart keywords route to `chart_or_diagram`.")
    md.append("- caption-like prompts or long candidates route to `caption_like`.")
    md.append("- yes/no and MCQ regex/option parsing are handled explicitly.")
    md.append("- numeric/date regex used for `numeric`.")
    md.append("")
    md.append("## 4) Candidate validity and risks")
    md.append("- `fmt_validity(a)` depends on answer-space (`_format_validity`).")
    md.append("- `length_risk(a)` penalizes suspicious long candidates in short-answer spaces (`_length_risk`).")
    md.append("")
    md.append("## 5) General scoring formula")
    md.append("- `score_general(a)=2.0*view_freq(a)+1.0*fmt_validity(a)+0.4*1[a==base]-0.5*length_risk(a)`")
    md.append("")
    md.append("## 6) Observed answer-space distribution (n=1000, V91-NoCF diagnostics)")
    md.append("")
    for b, dist in rules["observed_answer_space_distribution_n1000_v91_nocf"].items():
        md.append(f"- {b}: {dist}")

    (root / "paper_neurips2026_artifacts" / "reports" / "answer_space_rules.md").write_text("\n".join(md) + "\n")

    # compact tex
    tex = []
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\begin{tabular}{ll}")
    tex.append("\\toprule")
    tex.append("Component & Rule summary \\\\")
    tex.append("\\midrule")
    tex.append("Normalization & lowercase + punctuation removal + strip \\\\")
    tex.append("Answer spaces & yes/no, MCQ, numeric, ocr\\_text\\_short, open\\_entity, chart, caption, unknown \\\\")
    tex.append("Format validity & space-dependent validity score (\\_format\\_validity) \\\\")
    tex.append("Length risk & penalize suspicious long outputs in short-answer spaces (\\_length\\_risk) \\\\")
    tex.append("General score & $2.0f+1.0v+0.4b-0.5r$ \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\caption{Answer-space classifier and normalization rules used by V91-NoCF.}")
    tex.append("\\label{tab:answer_space_rules}")
    tex.append("\\end{table}")
    (root / "paper_neurips2026_artifacts" / "tables" / "answer_space_rules.tex").write_text("\n".join(tex) + "\n")

    print(out_json)


if __name__ == "__main__":
    main()
