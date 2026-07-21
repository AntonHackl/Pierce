# H100 vs RTX Pro 6000 Pierce large_nu_v Query Comparison

Source: completed Slurm logs, because the saved result JSONs were produced before the lowercase-counter parser fix and contain N/A timings.

| nu | unique pairs | H100 steady ms | RTX Pro 6000 steady ms | RTX speedup vs H100 |
|---:|---:|---:|---:|---:|
| 200 | 27306 | 75.084 | 10.843 | 6.92x |
| 400 | 55517 | 80.264 | 15.706 | 5.11x |
| 600 | 82962 | 112.126 | 20.895 | 5.37x |
| 800 | 110491 | 146.442 | 27.110 | 5.40x |

Breakdown columns are available in the CSV/JSON next to this report.
H100 log: benchmarks/slurm_logs/slurm_cmp_h100_large_nu_v_2341263.out
RTX log: benchmarks/slurm_logs/slurm_cmp_rtx6000_large_nu_v_2341264.out
