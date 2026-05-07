# Sensitivity Summary (TextVQA + ChartQA)

## Group 1: CARE `w_sup` Sensitivity (n=1000, cached rerank-only)

| Benchmark | w_sup | Score(0-1) | Delta vs default | Changed | Default wins | Setting wins | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| chartqa | 1.0 | 0.712 | -0.001 | 11 | 3 | 2 | -1 |
| chartqa | 1.5 | 0.712 | -0.001 | 7 | 2 | 1 | -1 |
| chartqa | 2.5 | 0.713 | +0.000 | 3 | 0 | 0 | 0 |
| chartqa | 3.0 | 0.713 | +0.000 | 3 | 0 | 0 | 0 |
| textvqa | 1.0 | 0.772 | +0.002 | 20 | 3 | 5 | 2 |
| textvqa | 1.5 | 0.772 | +0.002 | 20 | 3 | 5 | 2 |
| textvqa | 2.5 | 0.772 | +0.002 | 5 | 0 | 2 | 2 |
| textvqa | 3.0 | 0.772 | +0.002 | 5 | 0 | 2 | 2 |

Average delta across TextVQA+ChartQA:

- w_sup=1.0: +0.0005
- w_sup=1.5: +0.0005
- w_sup=2.5: +0.0010
- w_sup=3.0: +0.0010

## Group 2: CF Gate Sensitivity (strict/default/loose, n=1000, cached)

| Benchmark | Gate | Score(%) | Delta vs ASCA-only(%) | Switch coverage | Switch count | Correct switches | Incorrect switches | Net switch gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| chartqa | strict | 76.30 | +0.00 | 0.10% | 1 | 0 | 0 | 0 |
| chartqa | default | 76.30 | +0.00 | 0.20% | 2 | 0 | 0 | 0 |
| chartqa | loose | 76.30 | +0.00 | 0.20% | 2 | 0 | 0 | 0 |
| textvqa | strict | 71.50 | -0.01 | 0.60% | 6 | 0 | 5 | -5 |
| textvqa | default | 71.10 | -0.01 | 1.00% | 10 | 0 | 9 | -9 |
| textvqa | loose | 71.20 | -0.01 | 0.90% | 9 | 0 | 8 | -8 |
