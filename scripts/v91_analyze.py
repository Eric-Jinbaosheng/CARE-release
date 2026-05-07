#!/usr/bin/env python3
"""Analyze v91 diagnostic JSONL files: answer_space distribution, candidate
pool, change rate, format veto, beta/CF usage, rescue/harm vs baseline.

Usage:
  python scripts/v91_analyze.py --baseline_xlsx <path> --eccts_xlsx <path> \
      --diag_jsonl <path> [--bench <name>]
"""
import argparse
import json
import os
import re
import string
import sys
from collections import Counter, defaultdict

import pandas as pd

_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s):
    if s is None:
        return ""
    return str(s).strip().lower().translate(_PUNCT).strip()


def is_correct(pred, gt):
    p = normalize(pred)
    if not gt or pd.isna(gt):
        return False
    if isinstance(gt, list):
        return any(normalize(g) and (normalize(g) in p or p in normalize(g)) for g in gt)
    g = normalize(gt)
    return bool(g) and (g in p or p in g)


def find_xlsx(d):
    for f in sorted(os.listdir(d)):
        if f.endswith(".xlsx") and not f.startswith("~"):
            return os.path.join(d, f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True)
    ap.add_argument("--v91_dir", required=True)
    ap.add_argument("--diag_jsonl", default=None)
    ap.add_argument("--bench", default=None)
    args = ap.parse_args()

    bf = find_xlsx(args.baseline_dir)
    ef = find_xlsx(args.v91_dir)
    if not bf or not ef:
        print(f"missing xlsx: base={bf} v91={ef}", file=sys.stderr); sys.exit(1)

    df_b = pd.read_excel(bf)
    df_e = pd.read_excel(ef)
    n = min(len(df_b), len(df_e))

    rescue = harm = same_correct = same_wrong = swap_wrong = 0
    answer_space_counts = Counter()
    cf_used = 0
    fmt_veto = 0
    fallback = 0
    final_changed = 0
    candidate_pool_sizes = []
    margin_list = []
    entropy_list = []
    grounding_pos = 0
    beta_sum = 0.0
    beta_pos_count = 0

    diag_records = {}
    if args.diag_jsonl and os.path.exists(args.diag_jsonl):
        with open(args.diag_jsonl) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                diag_records[r.get("sample_id")] = r

    for i in range(n):
        rb = df_b.iloc[i]; re_ = df_e.iloc[i]
        bp, ep = rb["prediction"], re_["prediction"]
        gt = rb.get("answer", re_.get("answer"))
        same = normalize(bp) == normalize(ep)
        b_ok = is_correct(bp, gt); e_ok = is_correct(ep, gt)
        if same:
            (same_correct if b_ok else same_wrong).__class__  # placeholder
            if b_ok: same_correct += 1
            else: same_wrong += 1
        else:
            if b_ok and not e_ok: harm += 1
            elif not b_ok and e_ok: rescue += 1
            elif b_ok and e_ok: same_correct += 1
            else: swap_wrong += 1

        # Diagnostic features (sample_id is 1-indexed in our writer)
        rec = diag_records.get(i + 1)
        if rec:
            answer_space_counts[rec.get("answer_space", "unknown")] += 1
            if rec.get("cf_used"): cf_used += 1
            if rec.get("format_veto"): fmt_veto += 1
            if rec.get("fallback_to_ttaug"): fallback += 1
            if rec.get("final_changed"): final_changed += 1
            if rec.get("n_candidates") is not None:
                candidate_pool_sizes.append(rec["n_candidates"])
            if rec.get("margin") is not None:
                margin_list.append(rec["margin"])
            if rec.get("entropy") is not None:
                entropy_list.append(rec["entropy"])
            if rec.get("max_grounding", 0) > 0:
                grounding_pos += 1
            if rec.get("beta", 0) > 0:
                beta_sum += rec["beta"]
                beta_pos_count += 1

    print(f"=== {os.path.basename(args.v91_dir)} (bench={args.bench or '?'}) ===")
    print(f"  n_samples         : {n}")
    print(f"  rescue            : {rescue}")
    print(f"  harm              : {harm}")
    print(f"  net (rescue-harm) : {rescue - harm:+d}")
    print(f"  swap_wrong        : {swap_wrong}")
    print(f"  same_correct      : {same_correct}")
    print(f"  same_wrong        : {same_wrong}")
    chg_rate = (rescue + harm + swap_wrong) / max(1, n)
    print(f"  change_rate       : {chg_rate*100:.1f}%")
    if diag_records:
        print(f"  candidate_pool    : "
              f"mean={sum(candidate_pool_sizes)/max(1,len(candidate_pool_sizes)):.2f} "
              f"median={sorted(candidate_pool_sizes)[len(candidate_pool_sizes)//2] if candidate_pool_sizes else 0}")
        print(f"  final_change_rate : {final_changed/max(1,n)*100:.1f}%")
        print(f"  fallback_rate     : {fallback/max(1,n)*100:.1f}%")
        print(f"  format_veto_rate  : {fmt_veto/max(1,n)*100:.1f}%")
        print(f"  cf_used_rate      : {cf_used/max(1,n)*100:.1f}%")
        print(f"  beta_mean(>0)     : {beta_sum/max(1,beta_pos_count):.3f} ({beta_pos_count}/{n})")
        print(f"  grounding_pos_rate: {grounding_pos/max(1,n)*100:.1f}%")
        print(f"  margin_median     : {sorted(margin_list)[len(margin_list)//2] if margin_list else 0:.3f}")
        print(f"  entropy_median    : {sorted(entropy_list)[len(entropy_list)//2] if entropy_list else 0:.3f}")
        print(f"  answer_space      : {dict(answer_space_counts.most_common())}")


if __name__ == "__main__":
    main()
