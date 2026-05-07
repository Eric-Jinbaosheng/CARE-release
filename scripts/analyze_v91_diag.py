#!/usr/bin/env python3
import argparse
import json
from collections import Counter


def split_runs(samples):
    runs = []
    cur = []
    prev = -1
    for d in samples:
        sid = int(d.get("sample_id", 0) or 0)
        if cur and sid <= prev:
            runs.append(cur)
            cur = []
        cur.append(d)
        prev = sid
    if cur:
        runs.append(cur)
    return runs


def pct(a, b):
    return 0.0 if b <= 0 else 100.0 * float(a) / float(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to v91*.jsonl diagnostics")
    ap.add_argument("--last-run", action="store_true", help="Analyze only the last run in appended jsonl")
    args = ap.parse_args()

    rows = []
    with open(args.jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        print("No rows")
        return

    if args.last_run:
        runs = split_runs(rows)
        rows = runs[-1]

    n = len(rows)
    c_block = Counter(d.get("block_reason", "unknown") for d in rows)
    c_mask = Counter(d.get("mask_type", "unknown") for d in rows)
    c_mq = Counter(d.get("mask_quality", "unknown") for d in rows)
    c_cq = Counter(d.get("control_quality", "unknown") for d in rows)
    c_mode = Counter(d.get("cf3_mode", "unknown") for d in rows)

    cf_used = sum(1 for d in rows if bool(d.get("cf_used", False)))
    cf_changed = sum(1 for d in rows if bool(d.get("cf_final_changed", False)))
    final_changed = sum(1 for d in rows if bool(d.get("final_changed", False)))

    print(f"samples={n}")
    print(f"cf_used={cf_used} ({pct(cf_used,n):.2f}%)")
    print(f"cf_final_changed={cf_changed} ({pct(cf_changed,n):.2f}%)")
    print(f"final_changed={final_changed} ({pct(final_changed,n):.2f}%)")
    print("\nblock_reason_counts")
    for k, v in c_block.most_common():
        print(f"  {k}: {v}")
    print("\nmask_type_counts")
    for k, v in c_mask.most_common():
        print(f"  {k}: {v}")
    print("\nmask_quality_counts")
    for k, v in c_mq.most_common():
        print(f"  {k}: {v}")
    print("\ncontrol_quality_counts")
    for k, v in c_cq.most_common():
        print(f"  {k}: {v}")
    print("\ncf3_mode_counts")
    for k, v in c_mode.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
