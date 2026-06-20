#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS="${JOBS:-$(nproc)}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CLEAN=false
ONLY_COMPONENTS=()
VALID_COMPONENTS=(preprocess query face tdbase generators)

parse_only_arg() {
    local arg="$1"
    local item
    IFS=',' read -r -a items <<< "$arg"
    for item in "${items[@]}"; do
        if [[ -z "$item" ]]; then
            echo "Empty component in --only: $arg" >&2
            exit 1
        fi
        ONLY_COMPONENTS+=("$item")
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=true
            shift
            ;;
        --only)
            parse_only_arg "${2:?--only requires a component}"
            shift 2
            ;;
        --jobs)
            JOBS="${2:?--jobs requires a number}"
            shift 2
            ;;
        -h|--help)
            cat <<'EOF'
Usage: ./build_all.sh [--clean] [--only COMPONENT] [--jobs N]
       ./build_all.sh [--clean] [--only COMPONENT ...] [--jobs N]
       ./build_all.sh [--clean] [--only COMPONENT1,COMPONENT2,...] [--jobs N]

Components: preprocess, query, face, tdbase, generators

Use either a comma-separated list or repeat --only, for example:
  ./build_all.sh --only preprocess,query
  ./build_all.sh --only preprocess --only query

The script activates an existing component-specific Conda environment.
Create the environments from their YAML files before the first build.
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ${#ONLY_COMPONENTS[@]} -gt 0 ]]; then
    for requested in "${ONLY_COMPONENTS[@]}"; do
        valid=false
        for component in "${VALID_COMPONENTS[@]}"; do
            if [[ "$requested" == "$component" ]]; then
                valid=true
                break
            fi
        done

        if ! $valid; then
            echo "Unknown component for --only: $requested" >&2
            echo "Valid components: ${VALID_COMPONENTS[*]}" >&2
            exit 1
        fi
    done
fi

should_build() {
    if [[ ${#ONLY_COMPONENTS[@]} -eq 0 ]]; then
        return 0
    fi

    local requested
    for requested in "${ONLY_COMPONENTS[@]}"; do
        if [[ "$requested" == "$1" ]]; then
            return 0
        fi
    done

    return 1
}

init_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "Conda is required to build Pierce components." >&2
        exit 1
    fi

    local conda_base
    conda_base="$(conda info --base)"
    # shellcheck disable=SC1090
    source "$conda_base/etc/profile.d/conda.sh"
}

build_cmake() (
    local name="$1"
    local source_dir="$2"
    local env_name="$3"
    shift 3

    local build_dir="$source_dir/build"
    local cmake_args=("$@")

    echo
    echo "==> Building $name"

    # Conda's compiler activation hooks legitimately reference optional
    # backup variables that may be unset when switching from base.
    set +u
    local activate_status
    if conda activate "$env_name"; then
        activate_status=0
    else
        activate_status=$?
    fi
    set -u
    if [[ $activate_status -ne 0 ]]; then
        echo "Conda environment '$env_name' does not exist." >&2
        echo "Create it from the corresponding environment YAML first." >&2
        exit 1
    fi
    echo "    Conda environment: $env_name ($CONDA_PREFIX)"
    cmake_args+=("-DCMAKE_PREFIX_PATH=$CONDA_PREFIX")

    # Keep libraries from Conda, but avoid its cross-compiler/sysroot inside
    # the cluster container. This mirrors the working evaluation build.
    if [[ "$(uname -s)" == "Linux" && -x /usr/bin/g++ ]]; then
        cmake_args+=(
            "-DCMAKE_C_COMPILER=/usr/bin/gcc"
            "-DCMAKE_CXX_COMPILER=/usr/bin/g++"
        )
    fi

    if $CLEAN; then
        rm -rf "$build_dir"
    fi

    cmake -S "$source_dir" -B "$build_dir" --fresh \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        "${cmake_args[@]}"
    cmake --build "$build_dir" --parallel "$JOBS"
)

if should_build preprocess ||
   should_build query ||
   should_build face ||
   should_build tdbase; then
    init_conda
fi

if should_build preprocess; then
    build_cmake \
        "Pierce preprocess" \
        "$ROOT/pierce/preprocess" \
        "pierce_preprocess"
fi

if should_build query; then
    if [[ -z "${OptiX_INSTALL_DIR:-}" ]]; then
        echo "OptiX_INSTALL_DIR must point to the NVIDIA OptiX SDK." >&2
        exit 1
    fi
    build_cmake \
        "Pierce query" \
        "$ROOT/pierce/query" \
        "pierce_query"
fi

if should_build face; then
    build_cmake \
        "Face and TOUCH" \
        "$ROOT/baselines/face" \
        "cgal_spatial"
fi

if should_build tdbase; then
    build_cmake \
        "TDBase extensions" \
        "$ROOT/baselines/tdbase_extensions" \
        "tdbase_env" \
        -DUSE_GPU=ON
fi

if should_build generators; then
    compiler="${CXX:-g++}"
    "$compiler" -O3 -std=c++17 \
        "$ROOT/pierce/scripts/cpp_generator/generate_spheres.cpp" \
        -o "$ROOT/pierce/scripts/cpp_generator/generate_spheres"
fi

echo
echo "All requested Pierce components built successfully."
