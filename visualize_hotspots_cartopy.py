#!/usr/bin/env python3
"""
Visualize hotspots at time_bin 40 (10AM) using cartopy based on the traffic_volume_only_viz.ipynb notebook.
"""

import sys
import json
from pathlib import Path

# Add project source to path
project_root = Path(__file__).parent / "src"
sys.path.insert(0, str(project_root))

import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer

plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 10)


def load_traffic_volumes_gdf():
    """Load the traffic volumes GeoDataFrame."""
    return gpd.read_file(
        "D:/project-cirrus/cases/traffic_volumes_simplified.geojson"
    )


def filter_hotspots_for_time_bin(hotspots_data, target_time_bin=40):
    """
    Filter hotspots for a specific time bin across all traffic volumes.
    
    Args:
        hotspots_data: The loaded hotspots data from the JSON file
        target_time_bin: The time bin to filter for (40 = 10AM)
    
    Returns:
        List of hotspot data for the target time bin
    """
    time_bin_minutes = hotspots_data['metadata']['time_bin_minutes']
    
    # Load the indexer to decode tvtw_index
    indexer = TVTWIndexer.load("output/tvtw_indexer.json")
    
    filtered_hotspots = []
    
    for hotspot in hotspots_data['hotspots']:
        tvtw_index = hotspot['tvtw_index']
        
        # Get the traffic volume and time bin from the tvtw_index
        tvtw_info = indexer.get_tvtw_from_index(tvtw_index)
        if tvtw_info:
            tv_name, time_bin = tvtw_info
            
            if time_bin == target_time_bin:
                human_readable = indexer.get_human_readable_tvtw(tvtw_index)
                hotspot_copy = hotspot.copy()
                hotspot_copy['traffic_volume_id'] = tv_name
                hotspot_copy['time_bin'] = time_bin
                if human_readable:
                    hotspot_copy['time_window'] = human_readable[1]
                filtered_hotspots.append(hotspot_copy)
    
    return filtered_hotspots


def plot_hotspots_cartopy(hotspots_10am, traffic_volumes_gdf, target_time_bin=40):
    """
    Plot hotspots at 10AM using cartopy based on the existing notebook approach.
    """
    if not hotspots_10am:
        print(f"No hotspots found for time bin {target_time_bin}")
        return
    
    print(f"Plotting {len(hotspots_10am)} hotspots for time bin {target_time_bin} (10AM)")
    
    # Create the plot with cartopy
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set extent to Western Europe (similar to the notebook)
    ax.set_extent([-10, 15, 35, 65], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, alpha=0.3, color='lightgray')
    ax.add_feature(cfeature.OCEAN, alpha=0.3, color='lightblue')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    # Extract traffic volume IDs from hotspots
    hotspot_tv_ids = {h['traffic_volume_id'] for h in hotspots_10am}
    print(f"Traffic volumes with hotspots: {len(hotspot_tv_ids)}")
    
    # Create mappings of hotspot data
    tv_to_flights = {h['traffic_volume_id']: len(h['flight_ids']) for h in hotspots_10am}
    tv_to_capacity = {h['traffic_volume_id']: h.get('capacity_per_bin', -1) for h in hotspots_10am}
    
    # Get max flights for color normalization
    max_flights = max(tv_to_flights.values()) if tv_to_flights else 1
    
    plotted_count = 0
    
    # Plot each hotspot traffic volume
    for tv_id in hotspot_tv_ids:
        tv_row = traffic_volumes_gdf[traffic_volumes_gdf['traffic_volume_id'] == tv_id]
        
        if not tv_row.empty:
            tv_geometry = tv_row.geometry.iloc[0]
            num_flights = tv_to_flights[tv_id]
            capacity_per_bin = tv_to_capacity[tv_id]
            
            # Color intensity based on number of flights
            intensity = num_flights / max_flights
            color = plt.cm.Reds(0.3 + 0.7 * intensity)  # Scale from light to dark red
            
            # Plot the traffic volume
            ax.add_geometries([tv_geometry], ccrs.PlateCarree(),
                              facecolor=color, edgecolor='darkred',
                              alpha=0.8, linewidth=1)
            
            # Add label at the centroid for high-traffic volumes
            if num_flights >= max_flights * 0.3:  # Only label significant hotspots
                centroid = tv_geometry.centroid
                
                # Format capacity display
                if capacity_per_bin > 0:
                    capacity_str = f"cap:{capacity_per_bin:.1f}"
                else:
                    capacity_str = "cap:N/A"
                
                label_text = f"{tv_id}\n({num_flights})\n{capacity_str}"
                
                ax.text(centroid.x, centroid.y, label_text,
                        transform=ccrs.PlateCarree(),
                        ha='center', va='center',
                        fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
            
            plotted_count += 1
        else:
            print(f"Warning: Traffic volume {tv_id} not found in GeoDataFrame")
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=max_flights))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Number of Flights', rotation=270, labelpad=15)
    
    # Set title
    time_str = f"{target_time_bin * 15 // 60:02d}:{(target_time_bin * 15) % 60:02d}"
    plt.title(f'Traffic Volume Hotspots at {time_str} (Time Bin {target_time_bin})\n{len(hotspots_10am)} hotspots, {plotted_count} plotted', 
              fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = f"output/hotspots_visualization_time_bin_{target_time_bin}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()


def main():
    """Main function to create the cartopy visualization."""
    
    print("=== Creating Cartopy Visualization for Hotspots at 10AM ===")
    
    # Check required files
    required_files = [
        "output/hotspots_with_tvtw_index.json",
        "output/tvtw_indexer.json"
    ]
    missing_files = [p for p in required_files if not Path(p).exists()]
    if missing_files:
        print(f"X Missing required files: {missing_files}")
        print("Please run the hotspot extraction script first.")
        return
    
    try:
        # Load hotspots data
        print("1. Loading hotspot data...")
        with open("output/hotspots_with_tvtw_index.json", 'r') as f:
            hotspots_data = json.load(f)
        print(f"   Loaded {len(hotspots_data['hotspots'])} total hotspots")
        
        # Load traffic volumes
        print("2. Loading traffic volumes GeoDataFrame...")
        traffic_volumes_gdf = load_traffic_volumes_gdf()
        print(f"   Loaded {len(traffic_volumes_gdf)} traffic volumes")
        
        # Filter for time bin 40 (10AM)
        target_time_bin = 40
        print(f"3. Filtering hotspots for time bin {target_time_bin} (10AM)...")
        hotspots_10am = filter_hotspots_for_time_bin(hotspots_data, target_time_bin)
        print(f"   Found {len(hotspots_10am)} hotspots at 10AM")
        
        if hotspots_10am:
            # Show some sample data
            print("   Sample hotspots:")
            for i, hotspot in enumerate(hotspots_10am[:5], 1):
                capacity_info = f"cap:{hotspot.get('capacity_per_bin', -1):.1f}" if hotspot.get('capacity_per_bin', -1) > 0 else "cap:N/A"
                print(f"     [{i}] {hotspot['traffic_volume_id']} - {hotspot.get('time_window', 'N/A')} - {len(hotspot['flight_ids'])} flights - {capacity_info}")
        
        # Create visualization
        print("4. Creating cartopy visualization...")
        plot_hotspots_cartopy(hotspots_10am, traffic_volumes_gdf, target_time_bin)
        
        print("\n=== Cartopy Visualization Completed Successfully ===")
        
    except Exception as e:
        print(f"X Visualization creation failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()