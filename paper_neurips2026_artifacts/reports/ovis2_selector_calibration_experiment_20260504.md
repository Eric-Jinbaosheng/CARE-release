# Ovis2-2B Selector Calibration Experiment (2026-05-04)

- Goal: keep method unchanged; only calibrate selector in experimental section.
- Protocol: reuse existing generation cache; rerank-only + official evaluator.
- Candidate setting from official sweep: `weight_mode=strict_single`, `margin_tau=0.6`, `min_support=0.875` (`wStrict_g2`).

## Main Comparison (n=1000)

| benchmark   |   base |   ttaug |   asca_full |   asca_calibrated |   delta_cal_vs_full |   delta_cal_vs_ttaug | note                  |
|:------------|-------:|--------:|------------:|------------------:|--------------------:|---------------------:|:----------------------|
| textvqa     | 78.780 |  78.740 |      78.440 |            78.540 |               0.100 |               -0.200 | strict_g2 tuned       |
| ocrvqa      | 76.100 |  70.300 |      69.000 |            69.100 |               0.100 |               -1.200 | strict_g2 tuned       |
| chartqa     | 85.000 |  84.800 |      82.600 |            82.600 |               0.000 |               -2.200 | strict_g2 tuned       |
| gqa         | 34.700 |  54.900 |      50.400 |            50.400 |               0.000 |               -4.500 | strict_g2 tuned       |
| ocrbench    | 84.100 |  83.700 |      80.700 |            80.700 |               0.000 |               -3.000 | not tuned (kept full) |
| mme_rw      |  0.404 |   0.398 |       0.401 |             0.401 |               0.000 |                0.003 | not tuned (kept full) |
| amber       | 84.538 |  87.064 |      84.463 |            84.463 |               0.000 |               -2.601 | not tuned (kept full) |

- Mean delta on tuned 4 benchmarks (calibrated - full): 0.050
- Mean delta on tuned 4 benchmarks (calibrated - ttaug): -2.025

## Interpretation
- Selector can be corrected slightly (+0.1 on TextVQA and +0.1 on OCRVQA), but gains are small.
- For ChartQA and GQA, this strict setting does not improve over ASCA full (delta=0).
- Even after calibration, tuned scores remain below TTAug on the four tuned benchmarks.
- Therefore, second-backbone claim should be phrased as partial transfer / sanity check, not universal improvement.

## Source Files
- `paper_neurips2026_artifacts/second_backbone/ovis2_tuned_gated_sweep_strict_9b/ovis_tuned_gated_sweep_summary.csv`
- `benchmark_results/n_samples_1000/test_config_ovis2_2b_{base,ttaug,asca}_*/...`
