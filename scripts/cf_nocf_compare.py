#!/usr/bin/env python3
import json, os

base = '<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache'
benchmarks = ['textvqa', 'ocrbench', 'gqa', 'chartqa', 'ocrvqa', 'ai2d', 'mme_rw', 'coco']

print(f"{'Benchmark':12s} {'n':4s} {'cf_used':>8} {'cf_chg':>7} {'nocf_chg':>8} {'agree+dis':>10} {'betamean':>9}")
print('-' * 62)

total_cf_used = total_cf_chg = total_nocf_chg = total = 0

for bm in benchmarks:
    nocf_jsonl = f'{base}/test_config_smolvlm2_v91_nocf_{bm}/diagnostics/v91nocf_samples.jsonl'
    cf_jsonl = f'{base}/test_config_smolvlm2_v91_cf_{bm}/diagnostics/v91cf_samples.jsonl'

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
    cf_used = sum(1 for r in cf_recs.values() if r.get('cf_used'))
    cf_final_chg = sum(1 for r in cf_recs.values() if r.get('final_changed'))
    nocf_final_chg = sum(1 for r in nocf_recs.values() if r.get('final_changed'))
    beta_vals = [r['beta'] for r in cf_recs.values() if r.get('beta', 0) > 0]
    beta_mean = sum(beta_vals)/max(1, len(beta_vals))

    nocf_changed_sids = {sid for sid, r in nocf_recs.items() if r.get('final_changed')}
    cf_also_changed = sum(1 for sid in nocf_changed_sids if cf_recs.get(sid, {}).get('final_changed'))
    cf_only_changed = sum(1 for sid, r in cf_recs.items() if r.get('final_changed') and not nocf_recs.get(sid, {}).get('final_changed'))

    print(f'{bm:12s} n={n:4d} cf_used={cf_used:4d}({cf_used/n*100:5.1f}%) cf_chg={cf_final_chg:3d} nocf_chg={nocf_final_chg:3d} agree={cf_also_changed}+{cf_only_changed} betamean={beta_mean:.3f}')

    total_cf_used += cf_used
    total_cf_chg += cf_final_chg
    total_nocf_chg += nocf_final_chg
    total += n

print('-' * 62)
print(f'{"TOTAL":12s} n={total:4d} cf_used={total_cf_used:4d}({total_cf_used/total*100:5.1f}%) cf_chg={total_cf_chg:3d} nocf_chg={total_nocf_chg:3d}')
