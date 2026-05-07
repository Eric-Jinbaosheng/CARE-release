#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

from common import read_table_file

PUNCT = str.maketrans('', '', string.punctuation)

BENCHES = ["ocrbench", "gqa", "textvqa", "chartqa", "ocrvqa", "ai2d", "mme_rw", "coco", "amber"]

DS = {
    "ocrbench": "OCRBench",
    "gqa": "GQA_TestDev_Balanced",
    "textvqa": "TextVQA_VAL",
    "chartqa": "ChartQA_TEST",
    "ocrvqa": "OCRVQA_TEST",
    "ai2d": "AI2D_TEST",
    "mme_rw": "MME-RealWorld-Lite",
    "coco": "COCO_VAL",
    "amber": "AMBER",
}


def norm(s):
    return str(s or "").strip().lower().translate(PUNCT).strip()


def first_letter(pred):
    p = str(pred or "").strip()
    m = re.match(r"^\(?([A-Ea-e])\)?(?:[\.:\)\s]|$)", p)
    return m.group(1).upper() if m else None


def bool_from_pred(pred):
    p = norm(pred)
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

    # MCQ-like row
    if "A" in row and "B" in row and len(g.strip()) == 1 and g.strip().upper() in "ABCDE":
        pl = first_letter(p)
        if pl is not None:
            return pl == g.strip().upper()

    # yes/no style
    if norm(g) in {"yes", "no", "true", "false"}:
        bp = bool_from_pred(p)
        if bp is not None:
            return bp == (norm(g) in {"yes", "true"})

    # list GT style
    if g.startswith("[") and g.endswith("]"):
        try:
            vals = ast.literal_eval(g)
            if isinstance(vals, list):
                pn = norm(p)
                return pn in {norm(v) for v in vals}
        except Exception:
            pass

    return norm(g) == norm(p)


def row_id(row, i):
    for k in ["index", "id", "question_id", "image_id"]:
        if k in row and str(row[k]).strip() != "":
            return str(row[k])
    return str(i)


def load_diag(path):
    if not path.exists():
        return {}
    out = {}
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            o = json.loads(ln)
        except Exception:
            continue
        sid = int(o.get("sample_id", len(out) + 1))
        out[sid] = o
    return out


def find_run_xlsx(root, n_samples, config, model):
    d = root / "benchmark_results" / f"n_samples_{n_samples}" / config / model
    if not d.exists():
        return None
    cands = [p for p in d.glob("*.xlsx") if not p.name.endswith("_score.xlsx") and "openai_result" not in p.name]
    if not cands:
        return None
    cands.sort()
    return cands[0]


def main():
    ap = argparse.ArgumentParser(description="Oracle/selection-gap analysis for TTAug vs AS-TTA(NoCF).")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--n-samples", type=int, default=1000)
    args = ap.parse_args()

    root = Path(args.repo_root)
    out_tables = root / "paper_neurips2026_artifacts" / "tables"
    out_reports = root / "paper_neurips2026_artifacts" / "reports"
    out_tables.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    changed_rows = []
    space_rows = []

    for b in BENCHES:
        ds = DS[b]
        cfg_t = f"test_config_smolvlm2_paper_ttaug_classical_{b}"
        cfg_n = f"test_config_smolvlm2_v91_nocf_{b}"
        x_t = find_run_xlsx(root, args.n_samples, cfg_t, "TTAugClassical_SmolVLM2_2B")
        x_n = find_run_xlsx(root, args.n_samples, cfg_n, "V91NoCF_SmolVLM2_2B")
        diag = load_diag(root / ".runtime_cache" / cfg_n / "diagnostics" / "v91nocf_samples.jsonl")

        if x_t is None or x_n is None:
            summary_rows.append({
                "benchmark": b,
                "dataset": ds,
                "n": 0,
                "candidate_oracle_acc": "NA",
                "ttaug_acc": "NA",
                "as_tta_acc": "NA",
                "selection_gap": "NA",
                "changed_tta_right_as_wrong": "NA",
                "changed_tta_wrong_as_right": "NA",
                "note": "missing_xlsx",
            })
            continue

        t_rows = read_table_file(x_t)
        n_rows = read_table_file(x_n)
        n = min(len(t_rows), len(n_rows))
        if n == 0:
            continue

        oracle_hit = 0
        t_hit = 0
        n_hit = 0
        ch_r2w = 0
        ch_w2r = 0
        sp = defaultdict(lambda: {"n": 0, "oracle": 0, "tta": 0, "astta": 0, "r2w": 0, "w2r": 0})

        for i in range(1, n + 1):
            tr = t_rows[i - 1]
            nr = n_rows[i - 1]
            gt = nr.get("answer", tr.get("answer", ""))
            tpred = tr.get("prediction", "")
            npred = nr.get("prediction", "")
            t_ok = is_correct(gt, tpred, tr)
            n_ok = is_correct(gt, npred, nr)

            d = diag.get(i, {})
            space = str(d.get("answer_space", "unknown"))
            cands = d.get("candidate_list", []) or []
            o_ok = any(is_correct(gt, c, nr) for c in cands)

            sid = row_id(nr, i)

            oracle_hit += 1 if o_ok else 0
            t_hit += 1 if t_ok else 0
            n_hit += 1 if n_ok else 0
            if t_ok and not n_ok:
                ch_r2w += 1
                changed_rows.append({"benchmark": b, "sample_id": sid, "type": "tta_correct_as_wrong", "gt": str(gt), "tta": str(tpred), "as_tta": str(npred)})
            elif (not t_ok) and n_ok:
                ch_w2r += 1
                changed_rows.append({"benchmark": b, "sample_id": sid, "type": "tta_wrong_as_right", "gt": str(gt), "tta": str(tpred), "as_tta": str(npred)})

            sp[space]["n"] += 1
            sp[space]["oracle"] += 1 if o_ok else 0
            sp[space]["tta"] += 1 if t_ok else 0
            sp[space]["astta"] += 1 if n_ok else 0
            sp[space]["r2w"] += 1 if (t_ok and not n_ok) else 0
            sp[space]["w2r"] += 1 if ((not t_ok) and n_ok) else 0

        cand_acc = oracle_hit / n
        t_acc = t_hit / n
        a_acc = n_hit / n
        gap = cand_acc - a_acc

        summary_rows.append({
            "benchmark": b,
            "dataset": ds,
            "n": n,
            "candidate_oracle_acc": f"{cand_acc:.6f}",
            "ttaug_acc": f"{t_acc:.6f}",
            "as_tta_acc": f"{a_acc:.6f}",
            "selection_gap": f"{gap:.6f}",
            "changed_tta_right_as_wrong": ch_r2w,
            "changed_tta_wrong_as_right": ch_w2r,
            "note": "ok",
        })

        for k, v in sorted(sp.items(), key=lambda x: x[0]):
            m = v["n"]
            space_rows.append({
                "benchmark": b,
                "answer_space": k,
                "n": m,
                "oracle_acc": f"{(v['oracle']/m):.6f}",
                "ttaug_acc": f"{(v['tta']/m):.6f}",
                "as_tta_acc": f"{(v['astta']/m):.6f}",
                "selection_gap": f"{((v['oracle']-v['astta'])/m):.6f}",
                "changed_tta_right_as_wrong": v["r2w"],
                "changed_tta_wrong_as_right": v["w2r"],
            })

    with open(out_tables / "oracle_gap_summary_n1000.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    with open(out_tables / "oracle_gap_changed_examples_n1000.csv", "w", newline="") as f:
        if changed_rows:
            w = csv.DictWriter(f, fieldnames=list(changed_rows[0].keys()))
            w.writeheader(); w.writerows(changed_rows)
        else:
            f.write("benchmark,sample_id,type,gt,tta,as_tta\n")

    with open(out_tables / "oracle_gap_answer_space_breakdown_n1000.csv", "w", newline="") as f:
        if space_rows:
            w = csv.DictWriter(f, fieldnames=list(space_rows[0].keys()))
            w.writeheader(); w.writerows(space_rows)
        else:
            f.write("benchmark,answer_space,n,oracle_acc,ttaug_acc,as_tta_acc,selection_gap,changed_tta_right_as_wrong,changed_tta_wrong_as_right\n")

    md = []
    md.append("# Oracle / Selection Gap Analysis (n=1000)")
    md.append("")
    md.append("Definitions:")
    md.append("- candidate oracle accuracy: whether any AS-TTA candidate_list answer matches GT")
    md.append("- selection gap = oracle_acc - as_tta_acc")
    md.append("- changed examples compare TTAug selected answer vs AS-TTA selected answer")
    md.append("")
    md.append("Output files:")
    md.append(f"- {out_tables / 'oracle_gap_summary_n1000.csv'}")
    md.append(f"- {out_tables / 'oracle_gap_changed_examples_n1000.csv'}")
    md.append(f"- {out_tables / 'oracle_gap_answer_space_breakdown_n1000.csv'}")
    (out_reports / "oracle_gap_report.md").write_text("\n".join(md) + "\n")

    print(out_tables / "oracle_gap_summary_n1000.csv")


if __name__ == "__main__":
    main()
