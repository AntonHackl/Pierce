# Pierce

Pierce is the artifact for *Pierce: GPU Ray Tracing for Spatial Joins over
Complex 3D Data*. It contains the implementation and the benchmark workflows
used for the paper, without unrelated historical evaluation code.

## Layout

- `pierce/`: preprocessing and OptiX/CUDA query implementation.
- `baselines/face/`: the Face and TOUCH CPU baselines used in the paper.
- `baselines/tdbase/`: upstream TDBase at the paper's exact commit.
- `baselines/tdbase_extensions/`: compatibility, conversion, and timing
  extensions layered over the pinned TDBase source.
- `benchmarks/overlap/`: overlap scalability, workload, complexity, and
  selectivity experiments.
- `benchmarks/predicates/`: overlap/intersection/containment comparison.
- `benchmarks/datasets/`: preprocessing and loading table.

## Checkout

```bash
git clone --recurse-submodules https://github.com/AntonHackl/Pierce.git
cd Pierce
git submodule update --init --recursive
```

TDBase must resolve to `5058e2f540438a497cd0592b9044e0bcbd745cbb`.

## Build

Prerequisites are Linux, CUDA 12.8, NVIDIA OptiX 7.5 or newer, CMake, CGAL,
Boost, Eigen, OpenMP, GMP, and MPFR. Set `OptiX_INSTALL_DIR` to the OptiX SDK.

Create the component environments once:

```bash
conda env create -f pierce/preprocess/environment-linux.yml
conda env create -f pierce/query/environment-linux.yml
conda env create -f baselines/face/environment.yml
conda env create -f baselines/tdbase_extensions/environment.yml
```

The build script activates the appropriate existing environment for each
component, so it can be invoked directly from the base environment:

```bash
./build_all.sh
./test_all.sh
```

The paper-facing executables are:

- `pierce/preprocess/build/bin/pierce_preprocess`
- `pierce/query/build/bin/pierce_overlap`
- `pierce/query/build/bin/pierce_overlap_two_pass`
- `pierce/query/build/bin/pierce_intersection`
- `pierce/query/build/bin/pierce_containment`
- `baselines/face/build/face_overlap`
- `baselines/face/build/touch_overlap`
- `baselines/tdbase_extensions/build/tdbase`

## Paper Experiments

Run commands from the repository root with `PYTHONPATH=.`:

```bash
# Dataset preprocessing/loading table
python benchmarks/datasets/benchmark.py --help

# Tissue input-size scaling
python benchmarks/overlap/run_nu_scalability.py --help

# MICRONS and cube workload runs used by the overall plot
python benchmarks/overlap/run_microns_overlap.py --help
python benchmarks/overlap/run_cube_scalability.py --help
python benchmarks/overlap/plot_overall_performance.py --help

# Sphere mesh-complexity scaling
python benchmarks/overlap/run_mesh_complexity_benchmark.py --help

# Two-Pass, Estimated, and Fixed-memory selectivity comparison
python benchmarks/overlap/selectivity_test.py --help

# Predicate comparison and breakdown
python benchmarks/predicates/run_nu_scalability.py --help
python benchmarks/predicates/run_microns_query_comparison.py --help
python benchmarks/predicates/plot_overall_performance.py --help

# Export the five paper figures and dataset table
python benchmarks/export_figures.py
```

Generated datasets and runs live under the ignored `data/` and benchmark
output directories. MICRONS download/conversion helpers are in `scripts/`.
Large `tdbase_large` nuclei and nuclei-nuclei datasets can be generated
with `scripts/generate_large_nu_nn_data.sh`.

To generate the neuron datasets used by the MICrONS benchmarks, use the same
mesh-bounding-box workflow for both regional subsets:

```bash
conda env create -f scripts/environment.yml
conda activate pierce_microns
scripts/construct_micron_datasets.sh
```

This is equivalent to running these two commands from the repository root:

```bash
# 4 GB neuron subset (used as Neurons_1 / Neurons_2)
python scripts/download_microns_region_by_mesh_bbox.py \
  --target-gb 4.0 --max-gb 4.3 \
  --x-min-nm 347992 --x-max-nm 1447384 \
  --y-min-nm 300952 --y-max-nm 1116304 \
  --z-min-nm 594000 --z-max-nm 1114320 \
  --format glb --separate --shuffle \
  --download-dir ./scripts/microns_data/microns_region_4gb_npz \
  --export-dir ./scripts/microns_data/microns_region_4gb_glb

# 8 GB neuron subset (used as Neurons_3 / Neurons_4)
python scripts/download_microns_region_by_mesh_bbox.py \
  --target-gb 8.0 --max-gb 9.0 \
  --x-min-nm 800688 --x-max-nm 994688 \
  --y-min-nm 611628 --y-max-nm 805628 \
  --z-min-nm 757160 --z-max-nm 951160 \
  --format glb --separate --shuffle \
  --download-dir ./scripts/microns_data/microns_region_8gb_npz \
  --export-dir ./scripts/microns_data/microns_region_8gb_glb
```

These commands produce the same source directory layout that the evaluation
benchmarks expect:

- `scripts/microns_data/microns_region_4gb_glb`
- `scripts/microns_data/microns_region_8gb_glb`

When running the MICrONS benchmarks in this repo, point them at that root:

```bash
python benchmarks/overlap/run_microns_overlap.py \
  --source-root ./scripts/microns_data \
  --sizes 4 8

python benchmarks/predicates/run_microns_query_comparison.py \
  --source-root ./scripts/microns_data \
  --sizes 4 8
```
