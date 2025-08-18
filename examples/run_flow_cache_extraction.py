#!/usr/bin/env python3
"""
Exemplary Run Script for Flow Cache Extraction

This script demonstrates various ways to use the cache_flow_extract.py tool
with different configurations and parameters.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("stdout:")
            print(e.stdout)
        if e.stderr:
            print("stderr:")
            print(e.stderr)
        return False
    except FileNotFoundError:
        print("❌ FAILED - Command not found")
        return False

def main():
    """Run exemplary flow cache extraction scenarios."""
    
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Path to the flow cache extraction script
    extract_script = project_root / "src" / "project_tailwind" / "flow_x" / "cache_flow_extract.py"
    
    # Common parameters
    tv_geojson = "D:/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson"
    
    print("Flow Cache Extraction - Exemplary Runs")
    print("=====================================")
    print(f"Project root: {project_root}")
    print(f"Extract script: {extract_script}")
    print(f"Traffic volumes: {tv_geojson}")
    
    # Check if required files exist
    if not extract_script.exists():
        print(f"❌ Extract script not found: {extract_script}")
        return
    
    if not Path(tv_geojson).exists():
        print(f"❌ Traffic volume GeoJSON not found: {tv_geojson}")
        print("Please update the path in this script to point to your traffic volume GeoJSON file")
        return
    
    # Change to project root directory for relative paths to work
    import os
    os.chdir(project_root)
    
    success_count = 0
    total_runs = 0
    
    # Example 1: Basic run with limited hotspots
    total_runs += 1
    cmd1 = [
        "python", str(extract_script),
        "--tv-geojson", tv_geojson,
        "--threshold", "0.0",
        # "--limit-hotspots", "5",
        "--output-dir", "output/examples/basic_run"
    ]
    if run_command(cmd1, "Basic run with 5 hotspots"):
        success_count += 1
    
    # # Example 2: Focused analysis on specific traffic volume and hour
    # total_runs += 1
    # cmd2 = [
    #     "python", str(extract_script),
    #     "--tv-geojson", tv_geojson,
    #     "--threshold", "0.0",
    #     "--only-tv", "EDG6PAD",
    #     "--only-hour", "11",
    #     "--output-dir", "output/examples/focused_analysis"
    # ]
    # if run_command(cmd2, "Focused analysis on specific TV and hour"):
    #     success_count += 1
    
    # # Example 3: Bin mode analysis
    # total_runs += 1
    # cmd3 = [
    #     "python", str(extract_script),
    #     "--tv-geojson", tv_geojson,
    #     "--mode", "bin",
    #     "--threshold", "0.0",
    #     "--limit-hotspots", "3",
    #     "--output-dir", "output/examples/bin_mode"
    # ]
    # if run_command(cmd3, "Bin mode analysis"):
    #     success_count += 1
    
    # # Example 4: Custom FlowX parameters
    # total_runs += 1
    # cmd4 = [
    #     "python", str(extract_script),
    #     "--tv-geojson", tv_geojson,
    #     "--threshold", "0.0",
    #     "--limit-hotspots", "3",
    #     "--max-groups", "5",
    #     "--alpha", "0.1",
    #     "--group-size-lam", "0.2",
    #     "--path-length-gamma", "1.5",
    #     "--output-dir", "output/examples/custom_flowx"
    # ]
    # if run_command(cmd4, "Custom FlowX parameters"):
    #     success_count += 1
    
    # # Example 5: High threshold analysis (only severely overloaded hotspots)
    # total_runs += 1
    # cmd5 = [
    #     "python", str(extract_script),
    #     "--tv-geojson", tv_geojson,
    #     "--threshold", "10.0",
    #     "--limit-hotspots", "10",
    #     "--output-dir", "output/examples/high_threshold"
    # ]
    # if run_command(cmd5, "High threshold analysis (threshold=10.0)"):
    #     success_count += 1
    
    # # Example 6: Specific TVTW analysis (if we know a specific overloaded bin)
    # total_runs += 1
    # cmd6 = [
    #     "python", str(extract_script),
    #     "--tv-geojson", tv_geojson,
    #     "--threshold", "0.0",
    #     "--only-tvtw", "5900",
    #     "--mode", "bin",
    #     "--output-dir", "output/examples/specific_tvtw"
    # ]
    # if run_command(cmd6, "Specific TVTW analysis"):
    #     success_count += 1
    
    # # Example 7: Debug mode with verbose output
    # total_runs += 1
    # cmd7 = [
    #     "python", str(extract_script),
    #     "--tv-geojson", tv_geojson,
    #     "--threshold", "0.0",
    #     "--limit-hotspots", "2",
    #     "--debug",
    #     "--output-dir", "output/examples/debug_mode"
    # ]
    # if run_command(cmd7, "Debug mode with verbose output"):
    #     success_count += 1
    
    # # Summary
    # print(f"\n{'='*60}")
    # print("SUMMARY")
    # print(f"{'='*60}")
    # print(f"Successful runs: {success_count}/{total_runs}")
    
    # if success_count == total_runs:
    #     print("🎉 All exemplary runs completed successfully!")
    #     print("\nOutput directories created:")
    #     for example_dir in ["basic_run", "focused_analysis", "bin_mode", "custom_flowx", 
    #                        "high_threshold", "specific_tvtw", "debug_mode"]:
    #         output_dir = project_root / "output" / "examples" / example_dir
    #         if output_dir.exists():
    #             print(f"  - {output_dir}")
    #             # List CSV files in each directory
    #             csv_files = list(output_dir.glob("*.csv"))
    #             for csv_file in csv_files:
    #                 print(f"    └── {csv_file.name}")
    # else:
    #     print(f"⚠️  {total_runs - success_count} runs failed. Check the output above for details.")
    
    # print(f"\nTo explore the results, check the CSV files in:")
    # print(f"  {project_root / 'output' / 'examples'}")


if __name__ == "__main__":
    main()