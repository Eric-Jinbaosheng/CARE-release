#!/usr/bin/env python3
"""Analyze CF vs NoCF divergence from diagnostics JSONL only (no pandas)."""
import argparse
import csv
import json
from pathlib import Path


def load_jsonl(path):
    recs = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = int(r.get("sample_id"))
            recs[sid] = r
    return recs


def norm(s):
    if s is None:
        return ""
    return str(s).strip().lower()


def infer_reason(cf, nocf):
    final_cf = norm(cf.get("final_answer"))
    final_nocf = norm(nocf.get("final_answer"))
    rel = norm(cf.get("rel_norm", cf.get("rel_answer")))
    ctrl = norm(cf.get("ctrl_norm", cf.get("ctrl_answer")))
    if final_cf == final_nocf:
        return "same_final"
    if final_cf == ctrl and final_cf != rel:
        return "ctrl_match_promoted"
    if final_cf == rel and final_cf != ctrl:
        return "rel_match_promoted"
    if final_cf not in (rel, ctrl):
        return "continuous_or_general_rerank"
    return "mixed"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", default="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache")
    p.add_argument("--cf-config", default="test_config_smolvlm2_v91_cf_textvqa")
    p.add_argument("--nocf-config", default="test_config_smolvlm2_v91_nocf_textvqa")
    p.add_argument("--out-csv", default="")
    args = p.parse_args()

    root = Path(args.cache_root)
    cf_path = root / args.cf_config / "diagnostics" / "v91cf_samples.jsonl"
    nocf_path = root / args.nocf_config / "diagnostics" / "v91nocf_samples.jsonl"

    cf = load_jsonl(cf_path)
    nocf = load_jsonl(nocf_path)
    common = sorted(set(cf) & set(nocf))
    diffs = []
    for sid in common:
        rc = cf[sid]
        rn = nocf[sid]
        if norm(rc.get("final_answer")) == norm(rn.get("final_answer")):
            continue
        row = {
            "sample_id": sid,
            "answer_space": rc.get("answer_space", ""),
            "base_answer": rc.get("base_answer", ""),
            "nocf_final": rn.get("final_answer", ""),
            "cf_final": rc.get("final_answer", ""),
            "cf_used": bool(rc.get("cf_used", False)),
            "beta": rc.get("beta", 0.0),
            "margin": rc.get("margin", 0.0),
            "entropy": rc.get("entropy", 0.0),
            "rel_answer": rc.get("rel_answer", ""),
            "ctrl_answer": rc.get("ctrl_answer", ""),
            "reason": infer_reason(rc, rn),
            "block_reason": rc.get("block_reason", ""),
            "mask_type": rc.get("mask_type", ""),
            "mask_quality": rc.get("mask_quality", ""),
            "control_quality": rc.get("control_quality", ""),
        }
        diffs.append(row)

    print(f"common={len(common)} diff={len(diffs)}")
    by_space = {}
    for r in diffs:
        by_space[r["answer_space"]] = by_space.get(r["answer_space"], 0) + 1
    print("by_answer_space=", json.dumps(by_space, ensure_ascii=False, sort_keys=True))
    by_reason = {}
    for r in diffs:
        by_reason[r["reason"]] = by_reason.get(r["reason"], 0) + 1
    print("by_reason=", json.dumps(by_reason, ensure_ascii=False, sort_keys=True))

    for r in diffs[:40]:
        print(
            f"sid={r['sample_id']:4d} space={r['answer_space']:14s} cf_used={str(r['cf_used']):5s} "
            f"beta={float(r['beta']):.2f} nocf={r['nocf_final']!r} -> cf={r['cf_final']!r} "
            f"reason={r['reason']} mask={r['mask_type']}/{r['mask_quality']}"
        )

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(diffs[0].keys()) if diffs else [])
            if diffs:
                w.writeheader()
                w.writerows(diffs)
        print(f"wrote={out}")


if __name__ == "__main__":
    main()
