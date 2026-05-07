# Full vs Ablation Changed-Case Table

Columns: Benchmark | Ablation | Changed | Full wins | Ablation wins | Net

## n=200

| Benchmark | Ablation | Changed | Full wins | Ablation wins | Net | overlap | note |
|---|---:|---:|---:|---:|---:|---:|---|
| textvqa | frequency_only | 6 | 0 | 0 | 0 | 200 | ok |
| textvqa | majority_vote | 0 | 0 | 0 | 0 | 200 | ok |
| textvqa | no_format | 0 | 0 | 0 | 0 | 200 | ok |
| textvqa | no_base_bias | 6 | 0 | 0 | 0 | 200 | ok |
| textvqa | no_length_risk | 0 | 0 | 0 | 0 | 200 | ok |
| ocrvqa | frequency_only | 9 | 5 | 0 | 5 | 200 | ok |
| ocrvqa | majority_vote | 0 | 0 | 0 | 0 | 200 | ok |
| ocrvqa | no_format | 0 | 0 | 0 | 0 | 200 | ok |
| ocrvqa | no_base_bias | 9 | 5 | 0 | 5 | 200 | ok |
| ocrvqa | no_length_risk | 0 | 0 | 0 | 0 | 200 | ok |
| gqa | frequency_only | 33 | 5 | 11 | -6 | 200 | ok |
| gqa | majority_vote | 0 | 0 | 0 | 0 | 200 | ok |
| gqa | no_format | 7 | 0 | 2 | -2 | 200 | ok |
| gqa | no_base_bias | 32 | 5 | 9 | -4 | 200 | ok |
| gqa | no_length_risk | 0 | 0 | 0 | 0 | 200 | ok |
| chartqa | frequency_only | NA | NA | NA | NA | NA | missing_ablation |
| chartqa | majority_vote | NA | NA | NA | NA | NA | missing_ablation |
| chartqa | no_format | NA | NA | NA | NA | NA | missing_ablation |
| chartqa | no_base_bias | 4 | 1 | 0 | 1 | 200 | ok |
| chartqa | no_length_risk | 0 | 0 | 0 | 0 | 200 | ok |
| ocrbench | frequency_only | 7 | 1 | 1 | 0 | 200 | ok |
| ocrbench | majority_vote | 0 | 0 | 0 | 0 | 200 | ok |
| ocrbench | no_format | 0 | 0 | 0 | 0 | 200 | ok |
| ocrbench | no_base_bias | NA | NA | NA | NA | NA | missing_ablation |
| ocrbench | no_length_risk | NA | NA | NA | NA | NA | missing_ablation |

## n=1000

| Benchmark | Ablation | Changed | Full wins | Ablation wins | Net | overlap | note |
|---|---:|---:|---:|---:|---:|---:|---|
| textvqa | frequency_only | 44 | 0 | 0 | 0 | 1000 | ok |
| textvqa | majority_vote | 0 | 0 | 0 | 0 | 1000 | ok |
| textvqa | no_format | 2 | 0 | 0 | 0 | 1000 | ok |
| textvqa | no_base_bias | NA | NA | NA | NA | NA | missing_ablation |
| textvqa | no_length_risk | 0 | 0 | 0 | 0 | 1000 | ok |
| ocrvqa | frequency_only | 38 | 14 | 3 | 11 | 1000 | ok |
| ocrvqa | majority_vote | 0 | 0 | 0 | 0 | 1000 | ok |
| ocrvqa | no_format | 1 | 0 | 0 | 0 | 1000 | ok |
| ocrvqa | no_base_bias | NA | NA | NA | NA | NA | missing_ablation |
| ocrvqa | no_length_risk | 0 | 0 | 0 | 0 | 1000 | ok |
| gqa | frequency_only | NA | NA | NA | NA | NA | missing_ablation |
| gqa | majority_vote | NA | NA | NA | NA | NA | missing_ablation |
| gqa | no_format | NA | NA | NA | NA | NA | missing_ablation |
| gqa | no_base_bias | NA | NA | NA | NA | NA | missing_ablation |
| gqa | no_length_risk | NA | NA | NA | NA | NA | missing_ablation |
| chartqa | frequency_only | 37 | 10 | 6 | 4 | 1000 | ok |
| chartqa | majority_vote | NA | NA | NA | NA | NA | missing_ablation |
| chartqa | no_format | 0 | 0 | 0 | 0 | 1000 | ok |
| chartqa | no_base_bias | NA | NA | NA | NA | NA | missing_ablation |
| chartqa | no_length_risk | 0 | 0 | 0 | 0 | 1000 | ok |
| ocrbench | frequency_only | 43 | 5 | 6 | -1 | 1000 | ok |
| ocrbench | majority_vote | 0 | 0 | 0 | 0 | 1000 | ok |
| ocrbench | no_format | 1 | 1 | 0 | 1 | 1000 | ok |
| ocrbench | no_base_bias | NA | NA | NA | NA | NA | missing_ablation |
| ocrbench | no_length_risk | 0 | 0 | 0 | 0 | 1000 | ok |
