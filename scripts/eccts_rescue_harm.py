#!/usr/bin/env python3
"""Compute rescue/harm/net for an ECCTS-enabled run vs the deterministic
paper-TTAug baseline.

For each sample:
  - base_pred  : prediction from deterministic paper-TTAug baseline
  - eccts_pred : prediction from ECCTS-enabled run
  - gt         : ground truth answer
  Classify:
    rescue : eccts changed prediction AND eccts is correct AND base was wrong
    harm   : eccts changed prediction AND eccts is wrong AND base was correct
    same_correct/same_wrong : no change
    swap_wrong : changed but both wrong
Usage:
  python scripts/eccts_rescue_harm.py <baseline_dir> <eccts_dir> [--bench ocrbench|gqa|textvqa|...]

Both dirs are paths to test_config_*/<MODEL_NAME>/ folders containing
*_<DATASET>.xlsx prediction files. The script auto-discovers the xlsx.
"""
import os
import sys
import re
import json
import argparse
import pandas as pd

def find_xlsx(d):
    for f in sorted(os.listdir(d)):
        if f.endswith(".xlsx") and not f.startswith("~"):
            return os.path.join(d, f)
    return None


def normalize(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    return re.sub(r"[^\w\s]", "", s).strip()


def is_correct(pred, gt):
    p = normalize(pred)
    if not gt or pd.isna(gt):
        return False
    if isinstance(gt, list):
        return any(normalize(g) and normalize(g) in p or p in normalize(g) for g in gt)
    g = normalize(gt)
    if not g:
        return False
    return g in p or p in g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_dir")
    ap.add_argument("eccts_dir")
    ap.add_argument("--bench", default=None)
    args = ap.parse_args()

    bf = find_xlsx(args.baseline_dir)
    ef = find_xlsx(args.eccts_dir)
    if not bf or not ef:
        print(f"Missing xlsx: base={bf} eccts={ef}", file=sys.stderr)
        sys.exit(1)
    df_b = pd.read_excel(bf)
    df_e = pd.read_excel(ef)

    if len(df_b) != len(df_e):
        print(f"WARN: shape mismatch {len(df_b)} vs {len(df_e)}", file=sys.stderr)
    n = min(len(df_b), len(df_e))

    rescue = 0
    harm = 0
    same_correct = 0
    same_wrong = 0
    swap_wrong = 0
    swap_no_gt = 0

    bench = args.bench or os.path.basename(args.eccts_dir).split("_")[0]

    for i in range(n):
        rb = df_b.iloc[i]
        re_ = df_e.iloc[i]
        base_pred = rb["prediction"]
        eccts_pred = re_["prediction"]
        gt = rb.get("answer", re_.get("answer"))

        same_pred = normalize(base_pred) == normalize(eccts_pred)
        b_correct = is_correct(base_pred, gt)
        e_correct = is_correct(eccts_pred, gt)

        if same_pred:
            if b_correct:
                same_correct += 1
            else:
                same_wrong += 1
        else:
            if b_correct and not e_correct:
                harm += 1
            elif not b_correct and e_correct:
                rescue += 1
            elif b_correct and e_correct:
                same_correct += 1  # both correct (different surface forms)
            else:
                swap_wrong += 1

    total_changed = rescue + harm + swap_wrong
    net = rescue - harm

    print(f"=== {os.path.basename(args.eccts_dir)} ===")
    print(f"  n_samples            : {n}")
    print(f"  rescue               : {rescue}")
    print(f"  harm                 : {harm}")
    print(f"  net (rescue - harm)  : {net:+d}")
    print(f"  swap_wrong           : {swap_wrong}")
    print(f"  same_correct         : {same_correct}")
    print(f"  same_wrong           : {same_wrong}")
    print(f"  total changed        : {total_changed}")
    print(f"  change_rate          : {total_changed/n*100:.1f}%")
    if total_changed > 0:
        print(f"  rescue / changed     : {rescue/total_changed*100:.1f}%")
        print(f"  harm   / changed     : {harm/total_changed*100:.1f}%")


if __name__ == "__main__":
    main()
