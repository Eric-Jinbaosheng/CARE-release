#!/usr/bin/env python3
"""Analyze P3 TextVQA n=1000: rescue/harm/swap + CF diagnostic."""
import json, os

base = '<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache'
bench = 'textvqa'
nocf_jsonl = f'{base}/test_config_smolvlm2_v91_nocf_{bench}/diagnostics/v91nocf_samples.jsonl'
cf_jsonl = f'{base}/test_config_smolvlm2_v91_cf_{bench}/diagnostics/v91cf_samples.jsonl'

nocf_recs = {}
with open(nocf_jsonl) as f:
    for line in f:
        r = json.loads(line)
        nocf_recs[r['sample_id']] = r

cf_recs = {}
with open(cf_jsonl) as f:
    for line in f:
        r = json.loads(line)
        cf_recs[r['sample_id']] = r

n = len(cf_recs)

# --- NoCF stats ---
def norm(s):
    import string
    p = str.maketrans('', '', string.punctuation)
    return str(s).strip().lower().translate(p).strip()

rescue = harm = swap_wrong = same_correct = same_wrong = 0
cf_used = cf_final_chg = cf_only_chg = 0
beta_vals = []

for sid in range(1, n+1):
    nr = nocf_recs[sid]
    cr = cf_recs[sid]

    # NoCF final_changed vs baseline (use prediction diff + answer_space heuristic)
    # We need the baseline predictions to compute rescue/harm. Use the xlsx approach.
    # Instead, use the diagnostic: nocf_final_changed vs cf_final_changed
    if cr.get('cf_used'): cf_used += 1
    if cr.get('final_changed'): cf_final_chg += 1
    if nr.get('beta', 0) > 0: beta_vals.append(nr['beta'])

    # CF-only changes (cf changed but nocf didn't)
    if cr.get('final_changed') and not nr.get('final_changed'):
        cf_only_chg += 1

print(f"=== TextVQA n={n} (P3) ===")
print(f"  nocf_final_changed : {sum(1 for r in nocf_recs.values() if r.get('final_changed'))}")
print(f"  cf_final_changed   : {cf_final_chg}")
print(f"  cf_only_changed    : {cf_only_chg}")
print(f"  cf_used            : {cf_used} ({cf_used/n*100:.1f}%)")
print(f"  beta_vals(>0)      : {len(beta_vals)} samples, mean={sum(beta_vals)/max(1,len(beta_vals)):.3f}")

# Answer space distribution
from collections import Counter
ans_dist = Counter(r.get('answer_space', 'unknown') for r in cf_recs.values())
print(f"  answer_space       : {dict(ans_dist.most_common())}")
