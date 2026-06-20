#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TDBASE_BUILD_DIR="${TDBASE_BUILD_DIR:-$ROOT/baselines/tdbase_extensions/build}"
TDBASE_DATA_DIR="${TDBASE_DATA_DIR:-$ROOT/baselines/tdbase/data}"
OUTPUT_SCENARIO="${OUTPUT_SCENARIO:-}"
if [[ -n "$OUTPUT_SCENARIO" ]]; then
    OUTPUT_DIR_DEFAULT="$ROOT/benchmarks/data_shared/$OUTPUT_SCENARIO/raw"
else
    OUTPUT_DIR_DEFAULT="$ROOT/benchmarks/overlap/data/raw"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_DIR_DEFAULT}"
THREADS="${THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
NV="${NV:-750}"
PREFIX="${PREFIX:-tdbase_large}"
NU_VALUES="${NU_VALUES:-200 400 600 800}"
DATASET_KINDS="${DATASET_KINDS:-n nn}"
STAGE_SPECS="${STAGE_SPECS:-}"

mkdir -p "$OUTPUT_DIR"
test -x "$TDBASE_BUILD_DIR/tdbase" || {
    echo "Build TDBase extensions first: ./build_all.sh" >&2
    exit 1
}

for mesh_path in "$TDBASE_DATA_DIR/nuclei.pt" "$TDBASE_DATA_DIR/vessel.pt"; do
    if [[ ! -s "$mesh_path" ]]; then
        echo "Missing TDBase prototype mesh: $mesh_path" >&2
        exit 1
    fi
    if [[ "$(head -n 1 "$mesh_path")" != "OFF" ]]; then
        echo "TDBase prototype mesh is not an OFF file: $mesh_path" >&2
        exit 1
    fi
done

echo "Writing generated datasets to: $OUTPUT_DIR"

stage_complete() {
    local base="$1"
    local stage_nu="$2"
    local nuclei_file="${base}_n_nv${NV}_nu${stage_nu}_vs100_r30.dt"
    local vessel_file="${base}_v_nv${NV}_nu${stage_nu}_vs100_r30.dt"
    [[ -s "$nuclei_file" && -s "$vessel_file" ]]
}

run_stage() {
    local stage_nu="$1"
    local dataset_kind="$2"
    local base="$OUTPUT_DIR/${PREFIX}_${dataset_kind}_nv${NV}_nu${stage_nu}"

    if stage_complete "$base" "$stage_nu"; then
        echo "Skipping LARGE ${dataset_kind} dataset for nu=${stage_nu} because outputs already exist."
        return
    fi

    echo "Generating LARGE ${dataset_kind} dataset with prefix=$PREFIX, nv=$NV, nu=${stage_nu}..."
    "$TDBASE_BUILD_DIR/tdbase" simulator \
        -n "$TDBASE_DATA_DIR/nuclei.pt" \
        -v "$TDBASE_DATA_DIR/vessel.pt" \
        -o "$base" \
        --hausdorff \
        --nv "$NV" \
        --nu "$stage_nu" \
        -r 30 \
        -i \
        -t "$THREADS"
}

if [[ -n "$STAGE_SPECS" ]]; then
    for stage_spec in $STAGE_SPECS; do
        stage_nu="${stage_spec%%:*}"
        dataset_kind="${stage_spec##*:}"
        if [[ -z "$stage_nu" || -z "$dataset_kind" || "$stage_nu" == "$dataset_kind" ]]; then
            echo "Invalid STAGE_SPECS entry: $stage_spec (expected nu:kind, e.g. 400:n)" >&2
            exit 1
        fi
        run_stage "$stage_nu" "$dataset_kind"
    done
else
    for nu in $NU_VALUES; do
        for dataset_kind in $DATASET_KINDS; do
            run_stage "$nu" "$dataset_kind"
        done
    done
fi
