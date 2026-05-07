#!/usr/bin/env python3
"""Analyze the 13 P3 TextVQA n=1000 cases where v91_cf differs from v91_no_cf."""
import json, os, sys, string
sys.path.insert(0, '<ANON_ROOT>/peking/smolvlm2_paper/ets_clean')
import pandas as pd
import numpy as np

_PUNCT = str.maketrans('', '', string.punctuation)

def normalize(s):
    if s is None: return ""
    return str(s).strip().lower().translate(_PUNCT).strip()

def is_correct(pred, gt):
    p = normalize(pred)
    if not gt or pd.isna(gt): return False
    if isinstance(gt, list):
        return any(normalize(g) and (normalize(g) in p or p in normalize(g)) for g in gt)
    g = normalize(gt)
    return bool(g) and (g in p or p in g)

base = '<ANON_ROOT>/peking/smolvlm2_paper/ets_clean'
cache = f'{base}/.runtime_cache'
result = f'{base}/benchmark_results/n_samples_1000'

# Load P3 diagnostic jsonl
nocf_jsonl = f'{cache}/test_config_smolvlm2_v91_nocf_textvqa/diagnostics/v91nocf_samples.jsonl'
cf_jsonl   = f'{cache}/test_config_smolvlm2_v91_cf_textvqa/diagnostics/v91cf_samples.jsonl'

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

# Load xlsx predictions
def load_xlsx(stem, subdir):
    dirs = {
        'base': f'{result}/test_config_smolvlm2_base_textvqa/Base_SmolVLM2_2B',
        'nocf': f'{result}/test_config_smolvlm2_v91_nocf_textvqa/V91NoCF_SmolVLM2_2B',
        'cf':   f'{result}/test_config_smolvlm2_v91_cf_textvqa/V91CF_SmolVLM2_2B',
    }
    path = None
    for root, _, files in os.walk(dirs[subdir]):
        for fn in files:
            if fn.endswith('.xlsx') and not fn.startswith('~'):
                path = os.path.join(root, fn)
                break
        if path:
            break
    if not path:
        return None
    df = pd.read_excel(path)
    df['idx'] = df.index
    return df.set_index('idx')

df_base = load_xlsx('base', 'base')
df_nocf = load_xlsx('nocf', 'nocf')
df_cf   = load_xlsx('cf',   'cf')

# Find divergence cases: cf changed but nocf didn't, or differently
# Compare predictions
divergence_cases = []
for i in range(1, 1001):
    if i not in nocf_recs or i not in cf_recs:
        continue
    r_n = nocf_recs[i]
    r_c = cf_recs[i]
    if not r_n.get('final_changed') and r_c.get('final_changed'):
        # CF changed something NoCF didn't
        div = {
            'sample_id': i,
            'answer_space': r_n.get('answer_space', 'unknown'),
            'cf_used': r_c.get('cf_used', False),
            'beta': r_c.get('beta', 0),
            'n_candidates': r_n.get('n_candidates', 0),
            'cf_margin': r_c.get('margin', 0),
            'cf_entropy': r_c.get('entropy', 0),
            'nocf_changed': r_n.get('final_changed', False),
            'cf_changed': r_c.get('final_changed', False),
            'rel_answer': r_c.get('rel_answer', ''),
            'ctrl_answer': r_c.get('ctrl_answer', ''),
            'cf_score': r_c.get('cf_score', None),
        }
        # Get predictions
        if df_base is not None and i-1 in df_base.index:
            div['gt'] = df_base.loc[i-1, 'answer']
            div['base_pred'] = df_base.loc[i-1, 'prediction']
            div['nocf_pred'] = df_nocf.loc[i-1, 'prediction'] if df_nocf is not None else ''
            div['cf_pred']   = df_cf.loc[i-1, 'prediction']   if df_cf   is not None else ''
        divergence_cases.append(div)

print(f"Total divergence cases (cf changed, nocf didn't): {len(divergence_cases)}\n")

# Sort by answer_space
from collections import Counter
ans_counts = Counter(d['answer_space'] for d in divergence_cases)
print(f"By answer_space: {dict(ans_counts)}\n")

# For each case, show details
helped = harmed = neutral = 0
for div in sorted(divergence_cases, key=lambda x: x['answer_space']):
    sid = div['sample_id']
    gt = div.get('gt', '')
    bp = div.get('base_pred', '')
    np_ = div.get('nocf_pred', '')
    cp = div.get('cf_pred', '')

    base_ok = is_correct(bp, gt)
    nocf_ok = is_correct(np_, gt)
    cf_ok = is_correct(cp, gt)

    if cf_ok and not nocf_ok:
        verdict = "RESCUE ✅"
        helped += 1
    elif not cf_ok and nocf_ok:
        verdict = "HARM   ❌"
        harmed += 1
    else:
        verdict = "NEUTRAL"
        neutral += 1

    print(f"=== sample_id={sid} [{div['answer_space']}] ===")
    print(f"  GT:          {gt}")
    print(f"  base pred:   {bp[:80]}")
    print(f"  nocf pred:   {np_[:80]}")
    print(f"  cf pred:     {cp[:80]}")
    print(f"  verdict:     {verdict}")
    print(f"  cf_used={div['cf_used']} beta={div['beta']:.3f} n_cand={div['n_candidates']}")
    print(f"  margin={div['cf_margin']:.3f} entropy={div['cf_entropy']:.3f}")
    print(f"  rel_answer:  {str(div['rel_answer'])[:60]}")
    print(f"  ctrl_answer: {str(div['ctrl_answer'])[:60]}")
    print(f"  cf_score:   {str(div.get('cf_score', 'N/A'))[:80]}")
    print()

print("="*60)
print(f"SUMMARY: {helped} RESCUE  {harmed} HARM  {neutral} NEUTRAL")
print(f"Net: {helped - harmed:+d}")
print(f"Rescue:Harm = {helped/max(1,harmed):.2f}:1")
