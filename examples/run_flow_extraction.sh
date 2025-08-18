#!/bin/bash

# Flow Cache Extraction - Example Commands
# ========================================
#
# This script contains example commands for running the flow cache extraction tool.
# You can run individual commands or uncomment/modify them as needed.

# Set the traffic volume GeoJSON file path
TV_GEOJSON="/Volumes/CrucialX/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson"

# Check if the GeoJSON file exists
if [ ! -f "$TV_GEOJSON" ]; then
    echo "❌ Traffic volume GeoJSON file not found: $TV_GEOJSON"
    echo "Please update the TV_GEOJSON variable in this script to point to your file"
    exit 1
fi

echo "Flow Cache Extraction - Example Commands"
echo "========================================"
echo "Traffic volumes: $TV_GEOJSON"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Example 1: Quick test with limited hotspots
echo "🔍 Example 1: Quick test with 3 hotspots"
python src/project_tailwind/flow_x/cache_flow_extract.py \
    --tv-geojson "$TV_GEOJSON" \
    --threshold 0.0 \
    --limit-hotspots 3 \
    --output-dir output/examples/quick_test

echo ""
echo "✅ Results saved to: output/examples/quick_test/"
echo ""

# Example 2: Focused analysis (uncomment to run)
# echo "🎯 Example 2: Focused analysis on specific traffic volume and hour"
# python src/project_tailwind/flow_x/cache_flow_extract.py \
#     --tv-geojson "$TV_GEOJSON" \
#     --threshold 0.0 \
#     --only-tv "EDG6PAD" \
#     --only-hour 11 \
#     --output-dir output/examples/focused_tv_hour

# Example 3: Bin mode analysis (uncomment to run)
# echo "📊 Example 3: Bin mode analysis"
# python src/project_tailwind/flow_x/cache_flow_extract.py \
#     --tv-geojson "$TV_GEOJSON" \
#     --mode bin \
#     --threshold 0.0 \
#     --limit-hotspots 5 \
#     --output-dir output/examples/bin_mode_analysis

# Example 4: Custom FlowX parameters (uncomment to run)
# echo "⚙️ Example 4: Custom FlowX parameters"
# python src/project_tailwind/flow_x/cache_flow_extract.py \
#     --tv-geojson "$TV_GEOJSON" \
#     --threshold 0.0 \
#     --limit-hotspots 3 \
#     --max-groups 10 \
#     --alpha 0.05 \
#     --group-size-lam 0.1 \
#     --path-length-gamma 2.0 \
#     --output-dir output/examples/custom_flowx_params

# Example 5: High precision analysis with higher threshold (uncomment to run)
# echo "🔬 Example 5: High precision analysis (only severely overloaded hotspots)"
# python src/project_tailwind/flow_x/cache_flow_extract.py \
#     --tv-geojson "$TV_GEOJSON" \
#     --threshold 5.0 \
#     --limit-hotspots 10 \
#     --output-dir output/examples/high_precision

echo "💡 Tip: Uncomment other examples in this script to run additional scenarios"
echo "📁 Check output/examples/ for all generated results"
echo ""
echo "📖 For more options, run:"
echo "   python src/project_tailwind/flow_x/cache_flow_extract.py --help"