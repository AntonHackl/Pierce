# Single vs Multi GPU Benchmark

This benchmark compares Pierce single-GPU and multi-GPU query runtime on the
largest default datasets used by the predicate benchmark workflow:

- largest cube pair: 200k cubes vs 1M cubes
- largest vessel-nuclei pair: `nu=800`, `nv=750`
- MICRONS 8 GB split pair

It also includes a MICRONS 16 GB case that runs only with the multi-GPU
configuration, because the dataset is expected not to fit on a single GPU.

Outputs are written under:

```text
benchmarks/single_vs_multi_gpu/runs/
```

Typical invocation from the repository root:

```bash
PYTHONPATH=. python benchmarks/single_vs_multi_gpu/run_single_vs_multi_gpu.py \
  --multi-gpus 2 \
  --runs 5 \
  --warmup-runs 1 \
  --timeout 2400
```

The 16 GB MICRONS case expects source GLB files at:

```text
scripts/microns_data/microns_region_16gb_glb
```

Failures are recorded in `results.json` and `summary.csv` while the remaining
datasets continue to run. Use `--fail-fast` only when debugging and you want the
first failure to stop the job.

Runtime figures are always written in both formats:

```text
figures/runtime_by_dataset_gpu.pdf
figures/runtime_by_dataset_gpu.png
```
