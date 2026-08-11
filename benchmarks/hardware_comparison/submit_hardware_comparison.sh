#!/bin/bash
# Submit H100 and RTX Pro 6000 runs plus a dependent CPU comparison report.
set -euo pipefail

REPO=/sc/projects/sci-zacharatou/chair/anton.hackl/Pierce
PAYLOAD="$REPO/benchmarks/hardware_comparison/slurm_run_hardware_queries.sh"
REPORT="$REPO/benchmarks/hardware_comparison/compare_hardware_results.py"

: "${COMPARE_TAG:=hardware_comparison_$(date +%Y%m%d_%H%M%S)}"
: "${MULTI_GPUS:=2}"
: "${RUNS:=1}"
: "${WARMUP_RUNS:=1}"
: "${TIMEOUT:=2400}"
: "${SOURCE_ROOT:=$REPO/scripts/microns_data}"
: "${EXTRA_ARGS:=}"
: "${CPUS_PER_TASK:=8}"
: "${MEM:=300G}"
: "${TIME_LIMIT:=24:00:00}"
: "${RUN_GPU_JOBS_CONCURRENTLY:=false}"

COMPARE_DIR="$REPO/benchmarks/hardware_comparison/runs/$COMPARE_TAG"
mkdir -p "$COMPARE_DIR" "$REPO/benchmarks/slurm_logs"

COMMON_EXPORT="ALL,COMPARE_TAG=$COMPARE_TAG,MULTI_GPUS=$MULTI_GPUS,RUNS=$RUNS,WARMUP_RUNS=$WARMUP_RUNS,TIMEOUT=$TIMEOUT,SOURCE_ROOT=$SOURCE_ROOT,EXTRA_ARGS=$EXTRA_ARGS"

h100_job=$(
  sbatch --parsable \
    --account=sci-zacharatou \
    --partition=aisc-batch \
    --nodes=1 \
    --gres=gpu:h100:2 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TIME_LIMIT" \
    --job-name=hwcmp_h100 \
    --output="$REPO/benchmarks/slurm_logs/slurm_hwcmp_h100_%j.out" \
    --error="$REPO/benchmarks/slurm_logs/slurm_hwcmp_h100_%j.err" \
    --export="$COMMON_EXPORT,GPU_LABEL=h100" \
    "$PAYLOAD"
)

rtx_dependency_args=()
if [[ "$RUN_GPU_JOBS_CONCURRENTLY" != "true" ]]; then
  rtx_dependency_args=(--dependency=afterok:"$h100_job")
fi
rtx_job=$(
  sbatch --parsable \
    --account=sci-zacharatou \
    --partition=gpu-batch \
    --nodes=1 \
    --gres=gpu:rtx_pro_6000:2 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TIME_LIMIT" \
    --job-name=hwcmp_rtx6000 \
    --output="$REPO/benchmarks/slurm_logs/slurm_hwcmp_rtx6000_%j.out" \
    --error="$REPO/benchmarks/slurm_logs/slurm_hwcmp_rtx6000_%j.err" \
    "${rtx_dependency_args[@]}" \
    --export="$COMMON_EXPORT,GPU_LABEL=rtx_pro_6000" \
    "$PAYLOAD"
)

compare_dependency="afterok:$rtx_job"
if [[ "$RUN_GPU_JOBS_CONCURRENTLY" == "true" ]]; then
  compare_dependency="afterok:$h100_job:$rtx_job"
fi
compare_job=$(
  sbatch --parsable \
    --account=sci-zacharatou \
    --partition=cpu-batch \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:20:00 \
    --job-name=hwcmp_report \
    --dependency="$compare_dependency" \
    --output="$REPO/benchmarks/slurm_logs/slurm_hwcmp_report_%j.out" \
    --error="$REPO/benchmarks/slurm_logs/slurm_hwcmp_report_%j.err" \
    --wrap="enroot start --root --rw --mount /sc/projects/sci-zacharatou/chair/anton.hackl/:/sc/projects/sci-zacharatou/chair/anton.hackl/ --mount /sc/home/anton.hackl:/sc/home/anton.hackl/ pyxis_RaySpace bash -lc \"cd '$REPO' && python '$REPORT' '$COMPARE_DIR'\""
)

cat <<EOF
Submitted hardware comparison.
  compare tag: $COMPARE_TAG
  H100 job:    $h100_job
  RTX job:     $rtx_job
  report job:  $compare_job
  output dir:  $COMPARE_DIR
  concurrent:  $RUN_GPU_JOBS_CONCURRENTLY

After both GPU jobs finish, the report job writes:
  $COMPARE_DIR/h100_vs_rtx_pro_6000_single_vs_multi_gpu.md
  $COMPARE_DIR/h100_vs_rtx_pro_6000_single_vs_multi_gpu.csv
  $COMPARE_DIR/h100_vs_rtx_pro_6000_single_vs_multi_gpu.json
EOF
