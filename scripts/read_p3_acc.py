#!/usr/bin/env python3
import pandas as pd, os

base = '<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/benchmark_results/n_samples_1000'
configs = {
    'base':     'test_config_smolvlm2_base_textvqa',
    'nocf':     'test_config_smolvlm2_v91_nocf_textvqa',
    'cf':       'test_config_smolvlm2_v91_cf_textvqa',
}

for name, cfg in configs.items():
    d = f'{base}/{cfg}'
    if not os.path.isdir(d):
        print(f'{name}: MISSING {d}')
        continue
    files = [f for f in os.listdir(d) if f.endswith('.xlsx')]
    if not files:
        print(f'{name}: no xlsx in {d}')
        continue
    path = os.path.join(d, files[0])
    df = pd.read_excel(path)
    n = len(df)
    acc = None
    if 'accuracy' in df.columns:
        acc = df['accuracy'].mean()
    elif 'correct' in df.columns:
        acc = df['correct'].mean()
    print(f'{name}: n={n}', end='')
    if acc is not None:
        print(f' accuracy={acc:.4f}')
    else:
        print(f' cols={list(df.columns)}')
