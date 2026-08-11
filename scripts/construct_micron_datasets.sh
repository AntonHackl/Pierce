#!/bin/bash
set -e

# This script constructs the MICrONS neuron datasets used by the benchmarks.
# Coordinates are calculated to provide spatial subsets with stable names.

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

echo "--------------------------------------------------"
echo "Constructing small dataset..."
python ./download_microns_region_by_mesh_bbox.py \
    --dataset-name small \
    --target-gb 4.0 --max-gb 4.3 \
    --x-min-nm 347992 --x-max-nm 1447384 \
    --y-min-nm 300952 --y-max-nm 1116304 \
    --z-min-nm 594000 --z-max-nm 1114320 \
    --shuffle

echo "--------------------------------------------------"
echo "Constructing large dataset (194um cube)..."
python ./download_microns_region_by_mesh_bbox.py \
    --dataset-name large \
    --target-gb 8.0 --max-gb 9.0 \
    --x-min-nm 800688 --x-max-nm 994688 \
    --y-min-nm 611628 --y-max-nm 805628 \
    --z-min-nm 757160 --z-max-nm 951160 \
    --shuffle

echo "--------------------------------------------------"
echo "All requested datasets have been processed."
echo "Output directories:"
echo "  - ./microns_data/microns_region_small_glb"
echo "  - ./microns_data/microns_region_large_glb"
