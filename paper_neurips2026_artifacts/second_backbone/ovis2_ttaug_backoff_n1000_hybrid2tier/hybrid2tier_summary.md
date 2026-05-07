# Ovis2-2B Two-Tier Hybrid Routing (from existing official eval outputs)

| Benchmark | Tier | Setting | TTAug | Hybrid | Delta |
|---|---|---|---:|---:|---:|
| ocrbench | loose | `m0p6_s0p75_d0p0` | 83.7000 | 83.8000 | +0.1000 |
| gqa | loose | `m0p6_s0p75_d0p0` | 54.9000 | 55.1000 | +0.2000 |
| textvqa | strict | `m0p6_s0p875_d0p0` | 78.7400 | 78.8100 | +0.0700 |
| chartqa | loose | `m0p6_s0p75_d0p0` | 84.8000 | 84.9000 | +0.1000 |
| ocrvqa | strict | `m0p6_s1p0_d0p0` | 70.3000 | 70.3000 | +0.0000 |
| ai2d | loose | `m0p6_s0p75_d0p0` | 0.8050 | 0.8050 | +0.0000 |
| mme_rw | loose | `m0p6_s0p75_d0p0` | 0.3980 | 0.3980 | +0.0000 |
| coco | loose | `m0p6_s0p75_d0p0` | 13.0290 | 13.0290 | +0.0000 |
| amber | loose | `m0p6_s0p75_d0p0` | 84.5872 | 84.9541 | +0.3670 |

**Average delta over 9 benchmarks**: +0.092997
