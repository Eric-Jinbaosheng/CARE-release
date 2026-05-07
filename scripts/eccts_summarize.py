#!/usr/bin/env python3
"""Aggregate per-sample ECCTS diagnostic JSONL into per-benchmark stats.

Usage:
  python scripts/eccts_summarize.py <DIAG_DIR_OR_JSONL> [<more files>...]

Reads each JSONL diagnostic file produced by ECCTSAdapter_SmolVLM2 and prints:
  applied_rate, unc_pass_rate, qual_pass_rate, grounding_positive_rate,
  final_change_rate, block_reason_counts.

If multiple files are given, prints a comparison table.
"""

import json
import os
import sys
from collections import Counter, defaultdict
from glob import glob


def summarize_one(path):
    n = 0
    applied = 0
    unc = 0
    qual = 0
    g_pos = 0
    changed = 0
    block_reasons = Counter()
    variant = None
    benchmark = None

    if not os.path.exists(path):
        return None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n += 1
            if variant is None:
                variant = rec.get("variant")
            if benchmark is None:
                benchmark = rec.get("benchmark")
            if rec.get("applied"):
                applied += 1
            if rec.get("unc_pass"):
                unc += 1
            if rec.get("qual_pass"):
                qual += 1
            if (rec.get("max_grounding") or 0) > 0:
                g_pos += 1
            if rec.get("final_changed"):
                changed += 1
            br = rec.get("block_reason")
            if br:
                # collapse parameterized parts
                key = br.split("(")[0]
                block_reasons[key] += 1
    if n == 0:
        return None
    return {
        "path": path,
        "variant": variant,
        "benchmark": benchmark,
        "n": n,
        "applied_rate": applied / n,
        "unc_pass_rate": unc / n,
        "qual_pass_rate": qual / n,
        "grounding_positive_rate": g_pos / n,
        "final_change_rate": changed / n,
        "block_reasons": dict(block_reasons.most_common()),
    }


def expand_paths(args):
    out = []
    for a in args:
        if os.path.isdir(a):
            out.extend(sorted(glob(os.path.join(a, "**/*.jsonl"), recursive=True)))
            out.extend(sorted(glob(os.path.join(a, "**/eccts_*_samples.jsonl"), recursive=True)))
        else:
            out.append(a)
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    paths = expand_paths(sys.argv[1:])
    if not paths:
        print("No JSONL diagnostic files found.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for p in paths:
        s = summarize_one(p)
        if s:
            rows.append(s)

    if not rows:
        print("No samples found in any file.", file=sys.stderr)
        sys.exit(1)

    # Print table
    headers = [
        "benchmark", "variant", "n", "applied", "unc_pass",
        "qual_pass", "g_pos", "changed",
    ]
    widths = [16, 6, 6, 9, 9, 10, 8, 9]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        cells = [
            (r["benchmark"] or "?")[:widths[0]],
            (r["variant"] or "?")[:widths[1]],
            str(r["n"]),
            f"{r['applied_rate']:.2%}",
            f"{r['unc_pass_rate']:.2%}",
            f"{r['qual_pass_rate']:.2%}",
            f"{r['grounding_positive_rate']:.2%}",
            f"{r['final_change_rate']:.2%}",
        ]
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))

    # Block-reason summary
    print("\nBlock reasons (top 5 per row):")
    for r in rows:
        if r["block_reasons"]:
            top5 = list(r["block_reasons"].items())[:5]
            print(f"  {r['benchmark']}/{r['variant']}: {top5}")


if __name__ == "__main__":
    main()
