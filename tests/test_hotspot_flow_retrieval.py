import sys
from pathlib import Path

project_root = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(project_root))

import geopandas as gpd
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.flow_x.flow_extractor import FlowXExtractor

# Pretty printing with rich (optional)
try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
    console = Console()
except Exception:
    RICH_AVAILABLE = False
    console = None


def _export_flow_results_to_txt(hotspot_info, groups, output_path):
    """Export flow extraction results to a text file for use in notebook visualization."""
    import json
    from pathlib import Path
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(exist_ok=True)
    
    # Prepare data for export
    export_data = {
        "hotspot": hotspot_info,
        "groups": groups,
        "export_timestamp": str(Path(__file__).stat().st_mtime)  # Simple timestamp
    }
    
    # Write to text file in JSON format for easy parsing
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"   Flow extraction results exported to {output_path}")
    print(f"   Hotspot: TVTW index {hotspot_info.get('tvtw_index', 'N/A')}")
    print(f"   Total groups found: {len(groups)}")
    for i, group in enumerate(groups, 1):
        print(f"     Group {i}: {group.get('group_size', 0)} flights from reference {group.get('reference_sector', 'N/A')}")


def _render_group_pretty(name: str, result):
    # Accept either a single group dict or a list of group dicts
    groups = []
    if isinstance(result, dict):
        groups = [result]
    elif isinstance(result, list):
        groups = [g for g in result if isinstance(g, dict)]
    if not groups:
        print(f"   No group to display for {name} hotspot")
        return

    # Derive hotspot header info from the first group
    hotspot = groups[0].get("hotspot", {})
    common_header = f"tv={hotspot.get('traffic_volume_id','?')} hour={hotspot.get('hour','?')}"

    if RICH_AVAILABLE:
        table = Table(title=f"{name} Hotspot Groups — {common_header}")
        table.add_column("#", justify="right")
        table.add_column("Reference TV")
        table.add_column("Group Size", justify="right")
        table.add_column("Avg Pairwise Sim", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Sample Flights", overflow="fold")
        for idx, grp in enumerate(groups, start=1):
            sample = ", ".join(grp.get("group_flights", [])[:8])
            table.add_row(
                str(idx),
                str(grp.get("reference_sector", "?")),
                str(grp.get("group_size", 0)),
                f"{grp.get('avg_pairwise_similarity', 0.0):.3f}",
                f"{grp.get('score', 0.0):.3f}",
                sample,
            )
        console.print(table)
    else:
        print(f"{name} Hotspot Groups — {common_header}")
        for idx, grp in enumerate(groups, start=1):
            print(
                f"  [{idx}] reference=", grp.get("reference_sector"),
                " size=", grp.get("group_size", 0),
                " avg_sim=", f"{grp.get('avg_pairwise_similarity', 0.0):.3f}",
                " score=", f"{grp.get('score', 0.0):.3f}",
                " flights=", grp.get("group_flights", [])[:8],
            )


def load_traffic_volumes_gdf():
    """Load the traffic volumes GeoDataFrame."""
    return gpd.read_file(
        "/Volumes/CrucialX/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson"
        # "D:/project-cirrus/output/scenarios/summer_good_wx_well_staffed_low.geojson"
    )


def test_hotspot_flight_retrieval():
    """Compute hotspots and retrieve flight IDs for each hotspot."""

    print("=== Testing Hotspot Flight Retrieval ===")

    # Check if required data files exist
    required_files = [
        "output/tvtw_indexer.json",
        "output/so6_occupancy_matrix_with_times.json",
    ]
    missing_files = [p for p in required_files if not Path(p).exists()]
    if missing_files:
        print(f"X Missing required files: {missing_files}")
        print("Please ensure these files exist in the project root.")
        return

    try:
        # Load traffic volumes and flights
        print("1. Loading traffic volumes GeoDataFrame...")
        traffic_volumes_gdf = load_traffic_volumes_gdf()
        print(f"   OK Loaded {len(traffic_volumes_gdf)} traffic volumes")

        print("2. Loading Flight List...")
        flight_list = FlightList(
            occupancy_file_path="output/so6_occupancy_matrix_with_times.json",
            tvtw_indexer_path="output/tvtw_indexer.json",
        )
        print(
            f"   OK Loaded {flight_list.num_flights} flights with {flight_list.num_tvtws} TVTWs"
        )

        # Initialize evaluator
        print("3. Initializing NetworkEvaluator...")
        evaluator = NetworkEvaluator(traffic_volumes_gdf, flight_list)
        print("   OK NetworkEvaluator initialized")

        # Retrieve hotspots and flight lists (per bin)
        # print("4. Retrieving hotspot flights (per TVTW bin)...")
        # per_bin = evaluator.get_hotspot_flights(threshold=0.0, mode="bin")
        # print(f"   Found {len(per_bin)} hotspot bins")



        # === HOTSPOT SELECTION === 
        hotspot_traffic_volume_id = "MASB5KL" # Paris TVs
        hotspot_hour = 6
        hotspot_bin = hotspot_hour * 4 # 4 bins per hour

        
        # Defer hotspot selection until after per_hour is computed

        # Retrieve hotspots and flight lists (per hour)
        print("5. Retrieving hotspot flights (per TV hour)...")
        per_hour = evaluator.get_hotspot_flights(threshold=0.0, mode="hour")
        print(f"   Found {len(per_hour)} (tv, hour) hotspots")
        
        # List all hotspot IDs and their row index
        for item in per_hour:
            tv_id = item.get("traffic_volume_id")
            hour = item.get("hour")
            if tv_id in flight_list.tv_id_to_idx:
                tv_row_idx = flight_list.tv_id_to_idx[tv_id]
                print(f"   Hotspot: {tv_id} (row index: {tv_row_idx}), hour: {hour}, "
                    f"unique_flights: {item.get('unique_flights', 'N/A')}, "
                    f"hourly_occupancy: {item.get('hourly_occupancy', 'N/A')}, "
                    f"hourly_capacity: {item.get('hourly_capacity', 'N/A')}, "
                    f"is_overloaded: {item.get('is_overloaded', 'N/A')}")
            else:
                print(f"   Hotspot: {tv_id} (row index: NOT FOUND), hour: {hour}, flights: {len(item['flight_ids'])}, hourly_capacity: {item.get('hourly_capacity', 'N/A')}")

        # Locate the designated hotspot (hour or bin) instead of defaulting to the first bin
        bins_per_hour = 60 // flight_list.time_bin_minutes
        num_time_bins_per_tv = flight_list.num_tvtws // len(flight_list.tv_id_to_idx)
        desired_tvtw_index = None
        try:
            tv_row = flight_list.tv_id_to_idx[hotspot_traffic_volume_id]
            desired_tvtw_index = tv_row * num_time_bins_per_tv + int(hotspot_bin)
        except Exception:
            desired_tvtw_index = None

        selected_hour_item = None
        selected_bin_item = None
        if per_hour:
            for item in per_hour:
                if (
                    item.get("traffic_volume_id") == hotspot_traffic_volume_id
                    and int(item.get("hour", -1)) == int(hotspot_hour)
                ):
                    selected_hour_item = item
                    break
        # if per_bin and desired_tvtw_index is not None:
        #     for item in per_bin:
        #         if int(item.get("tvtw_index", -1)) == int(desired_tvtw_index):
        #             selected_bin_item = item
        #             break

        if selected_hour_item:
            print(
                f"   Selected hotspot (hour): tv={selected_hour_item['traffic_volume_id']}, hour={selected_hour_item['hour']}, flights={len(selected_hour_item['flight_ids'])}, hourly_capacity={selected_hour_item.get('hourly_capacity', 'N/A')}"
            )
        elif selected_bin_item:
            print(
                f"   Selected hotspot (bin): tvtw_index={selected_bin_item['tvtw_index']}, flights={len(selected_bin_item['flight_ids'])}, hourly_capacity={selected_bin_item.get('hourly_capacity', 'N/A')}"
            )
        # elif per_bin:
        #     # Fallback if the designated hotspot isn't overloaded
        #     selected_bin_item = per_bin[0]
        #     print(
        #         f"   Designated hotspot not found among overloaded bins; falling back to first bin tvtw_index={selected_bin_item['tvtw_index']} with {len(selected_bin_item['flight_ids'])} flights"
        #     )

        # Print some flights in hotspots
        if selected_bin_item:
            print("   Flights in selected bin hotspot:")
            for flight_id in selected_bin_item["flight_ids"][:5]:  # Show first 5 flights
                flight_meta = flight_list.get_flight_metadata(flight_id)
                print(f"     - {flight_id}: {flight_meta.get('origin', 'N/A')} -> {flight_meta.get('destination', 'N/A')} ({flight_meta.get('takeoff_time', 'N/A')})")
        
        if per_hour:
            print("   Flights in first hotspot hour:")
            for flight_id in per_hour[0]["flight_ids"][:5]:  # Show first 5 flights
                flight_meta = flight_list.get_flight_metadata(flight_id)
                print(f"     - {flight_id}: {flight_meta.get('origin', 'N/A')} -> {flight_meta.get('destination', 'N/A')} ({flight_meta.get('takeoff_time', 'N/A')})")

        if selected_hour_item:
            sample = selected_hour_item
            assert "traffic_volume_id" in sample and "hour" in sample and "flight_ids" in sample
            assert isinstance(sample["flight_ids"], list)
            print(
                f"   Selected hour: tv={sample['traffic_volume_id']}, hour={sample['hour']}, flights={len(sample['flight_ids'])}, hourly_capacity={sample.get('hourly_capacity', 'N/A')}"
            )


        





        # Flow extraction tests
        print("6. Extracting groups from hotspots using FlowXExtractor (spectral)...")
        extractor = FlowXExtractor(flight_list)

        if selected_hour_item:
            selected_item_for_flow = selected_hour_item
            print(f"   Selected hour: tv={selected_item_for_flow['traffic_volume_id']}, hour={selected_item_for_flow['hour']}, flights={len(selected_item_for_flow['flight_ids'])}")
        elif selected_bin_item:
            selected_item_for_flow = selected_bin_item
            print(f"   Selected bin: tvtw_index={selected_item_for_flow['tvtw_index']}, flights={len(selected_item_for_flow['flight_ids'])}")
        else:
            raise ValueError("No hotspot selected")
        



        if selected_item_for_flow:
            result_groups = extractor.find_groups_from_evaluator_item(
                selected_item_for_flow,
                sparsification_alpha=0.05, # adaptive sparsification parameter
                max_groups=207,
                average_objective=False,
                k_max_trajectories_per_group=20,
                group_size_lam=0.0, # required if average_objective is False, higher lam means smaller groups
                # New knobs: prefer upstream evidence and reward longer paths
                path_length_gamma=2.0,  # higher gamma prefers longer paths
            )
            assert isinstance(result_groups, list)
            # Structure checks per-group
            hotspot_fids = set(selected_item_for_flow["flight_ids"]) if selected_item_for_flow["flight_ids"] else set()
            for grp in result_groups:
                assert isinstance(grp, dict)
                assert "reference_sector" in grp
                assert "group_flights" in grp
                assert "group_size" in grp
                assert set(grp.get("group_flights", [])).issubset(hotspot_fids)
            _render_group_pretty("Hour" if "hour" in selected_item_for_flow else "Bin", result_groups)

            # Export flow extraction results to txt file
            print("7. Exporting flow extraction results to output/flow_extraction_results.txt...")
            _export_flow_results_to_txt(selected_item_for_flow, result_groups, "output/flow_extraction_results.txt")

        # if per_hour:
        #     result_hour = extractor.find_groups_from_evaluator_item(per_hour[0], alpha=0.5, max_groups=3)
        #     assert isinstance(result_hour, list)
        #     hotspot_fids = set(per_hour[0]["flight_ids"]) if per_hour[0]["flight_ids"] else set()
        #     for grp in result_hour:
        #         assert isinstance(grp, dict)
        #         assert "reference_sector" in grp
        #         assert "group_flights" in grp
        #         assert "group_size" in grp
        #         assert set(grp.get("group_flights", [])).issubset(hotspot_fids)
        #     _render_group_pretty("Hour", result_hour)

        print("\n=== Hotspot Flight Retrieval Test Completed Successfully ===")

    except Exception as e:
        print(f"X Hotspot flight retrieval test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_hotspot_flight_retrieval()
