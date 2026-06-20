#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

expected=(
    "$ROOT/pierce/preprocess/build/bin/pierce_preprocess"
    "$ROOT/pierce/query/build/bin/pierce_overlap"
    "$ROOT/pierce/query/build/bin/pierce_overlap_two_pass"
    "$ROOT/pierce/query/build/bin/pierce_intersection"
    "$ROOT/pierce/query/build/bin/pierce_containment"
    "$ROOT/baselines/face/build/face_overlap"
    "$ROOT/baselines/face/build/touch_overlap"
    "$ROOT/baselines/tdbase_extensions/build/tdbase"
)

for executable in "${expected[@]}"; do
    test -x "$executable" || {
        echo "Missing executable: $executable" >&2
        exit 1
    }
done

PYTHONPATH="$ROOT" python -m compileall -q \
    "$ROOT/benchmarks/common" \
    "$ROOT/benchmarks/overlap" \
    "$ROOT/benchmarks/predicates" \
    "$ROOT/benchmarks/datasets"

for script in \
    benchmarks/datasets/benchmark.py \
    benchmarks/overlap/run_nu_scalability.py \
    benchmarks/overlap/run_mesh_complexity_benchmark.py \
    benchmarks/overlap/run_microns_overlap.py \
    benchmarks/overlap/run_cube_scalability.py \
    benchmarks/overlap/selectivity_test.py \
    benchmarks/predicates/run_nu_scalability.py \
    benchmarks/predicates/run_microns_query_comparison.py; do
    PYTHONPATH="$ROOT" python "$ROOT/$script" --help >/dev/null
done

echo "Pierce artifact checks passed."
