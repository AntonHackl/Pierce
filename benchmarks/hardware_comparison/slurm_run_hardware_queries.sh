#!/bin/bash
# Generic payload for one hardware side of the single- vs. multi-GPU comparison.
# Resource requests are intentionally supplied by submit_hardware_comparison.sh.
set -euo pipefail

REPO=/sc/projects/sci-zacharatou/chair/anton.hackl/Pierce

: "${COMPARE_TAG:?COMPARE_TAG must be set by the submit script}"
: "${GPU_LABEL:?GPU_LABEL must identify this hardware result}"
: "${MULTI_GPUS:=2}"
: "${RUNS:=1}"
: "${WARMUP_RUNS:=1}"
: "${TIMEOUT:=2400}"
: "${SOURCE_ROOT:=$REPO/scripts/microns_data}"
: "${BUILD_JOBS:=${SLURM_CPUS_PER_TASK:-8}}"
: "${EXTRA_ARGS:=}"

COMPARE_DIR="$REPO/benchmarks/hardware_comparison/runs/$COMPARE_TAG"
MARKER="$COMPARE_DIR/$GPU_LABEL.start"

mkdir -p "$COMPARE_DIR" "$REPO/benchmarks/slurm_logs"
touch "$MARKER"

echo "=== Hardware comparison ==="
echo "GPU_LABEL=$GPU_LABEL"
echo "COMPARE_TAG=$COMPARE_TAG"
echo "RUNS=$RUNS"
echo "WARMUP_RUNS=$WARMUP_RUNS"
echo "MULTI_GPUS=$MULTI_GPUS"
echo "=== GPU inventory ==="
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

srun --cpu-bind=none --container-name RaySpace \
     --container-workdir "$REPO" \
     --container-mounts /sc/home/anton.hackl/:/sc/home/anton.hackl/,/sc/projects/sci-zacharatou/chair/anton.hackl/:/sc/projects/sci-zacharatou/chair/anton.hackl/ \
     bash -lc "
        set -euo pipefail
        cd '$REPO'
        source /sc/home/anton.hackl/conda3/etc/profile.d/conda.sh
        set +u
        conda activate spatial_benchmark
        set -u
        export PYTHONPATH=\${PYTHONPATH:-}:.
        export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-8}

        echo '=== Build Pierce preprocess/query ==='
        ./build_all.sh --only preprocess,query --jobs '$BUILD_JOBS'

        python benchmarks/single_vs_multi_gpu/run_single_vs_multi_gpu.py \
          --multi-gpus '$MULTI_GPUS' \
          --runs '$RUNS' \
          --warmup-runs '$WARMUP_RUNS' \
          --timeout '$TIMEOUT' \
          --source-root '$SOURCE_ROOT' \
          \${EXTRA_ARGS:-}
     "

latest_results=$(find "$REPO/benchmarks/single_vs_multi_gpu/runs" \
    -maxdepth 2 -type f -name results.json -newer "$MARKER" \
    -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
if [[ -z "$latest_results" ]]; then
    echo "No single_vs_multi_gpu results.json found after benchmark run" >&2
    exit 1
fi

printf '%s\n' "$latest_results" > "$COMPARE_DIR/$GPU_LABEL.results_path"
cp "$latest_results" "$COMPARE_DIR/$GPU_LABEL.results.json"
echo "Recorded $GPU_LABEL results: $latest_results"
