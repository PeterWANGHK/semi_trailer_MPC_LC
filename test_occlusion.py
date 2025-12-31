#!/usr/bin/env python3
"""
Test script for occlusion region visualization in semi-trailer MPC simulation.

This script creates a simple scenario with trucks to demonstrate occlusion regions.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms as mtransforms
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch, Rectangle, Circle, Polygon as PolygonPatch
from semi_trailer_config import SemiTrailerConfig

def create_test_scenario_with_trucks():
    """Create a test scenario with multiple trucks for occlusion demonstration."""
    obstacles = [
        # Slow truck ahead (same lane) - WILL CREATE OCCLUSION
        [50.0, 0.0, np.deg2rad(0), 4.0, 12.0, 3.5, 1.0],     # Large truck
        
        # Another truck further ahead
        [100.0, 0.5, np.deg2rad(0), 4.5, 10.0, 3.0, 1.0],    # Medium truck
        
        # Small car (won't create occlusion)
        [75.0, -4.0, np.deg2rad(0), 5.0, 4.5, 2.0, 1.0],     # Car
        
        # Oncoming truck in left lane
        [150.0, -3.0, np.deg2rad(180), 5.0, 11.0, 3.5, 1.0], # Oncoming truck
        
        # Far truck (same direction)
        [200.0, 1.0, np.deg2rad(0), 4.0, 14.0, 4.0, 1.0],    # Very large truck
    ]
    return obstacles

def load_vehicle_sprite(fig_dir, *filenames):
    """Load the first available vehicle sprite from candidate filenames."""
    for name in filenames:
        path = os.path.join(fig_dir, name)
        if os.path.exists(path):
            try:
                return plt.imread(path)
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"Warning: failed to load vehicle sprite '{path}': {exc}")
    if filenames:
        joined = ", ".join(filenames)
        print(f"Warning: no vehicle sprite found for [{joined}] in '{fig_dir}'.")
    return None

def get_ego_vehicle_geometry(ego_state, config):
    """
    Compute key pose information for the ego tractor-trailer.
    
    Returns:
        dict with centers and key reference points.
    """
    x_ego, y_ego, theta_t, _, _, theta_s = ego_state
    
    tractor_front_x = x_ego + config.L_t * np.cos(theta_t)
    tractor_front_y = y_ego + config.L_t * np.sin(theta_t)
    tractor_center_x = x_ego + (config.L_t / 2.0) * np.cos(theta_t)
    tractor_center_y = y_ego + (config.L_t / 2.0) * np.sin(theta_t)
    
    trailer_center_x = x_ego - (config.L_s / 2.0) * np.cos(theta_s)
    trailer_center_y = y_ego - (config.L_s / 2.0) * np.sin(theta_s)
    trailer_rear_x = x_ego - config.L_s * np.cos(theta_s)
    trailer_rear_y = y_ego - config.L_s * np.sin(theta_s)
    
    return {
        "tractor_center": (tractor_center_x, tractor_center_y),
        "tractor_front": (tractor_front_x, tractor_front_y),
        "trailer_center": (trailer_center_x, trailer_center_y),
        "trailer_rear": (trailer_rear_x, trailer_rear_y),
        "theta_t": theta_t,
        "theta_s": theta_s,
    }


class LegendImage:
    """Container for legend entries rendered from image sprites."""

    def __init__(self, image):
        self.image = image


class HandlerLegendImage(HandlerBase):
    """Matplotlib legend handler that renders PNG sprites."""

    def __init__(self, zoom=0.18):
        super().__init__()
        self.zoom = zoom

    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        if orig_handle.image is None:
            return []

        image_artist = OffsetImage(orig_handle.image, zoom=self.zoom)
        annotation = AnnotationBbox(
            image_artist,
            (x0 + width / 2.0, y0 + height / 2.0),
            frameon=False,
            pad=0.0,
            box_alignment=(0.5, 0.5),
            transform=trans,
        )
        return [annotation]

def visualize_occlusion_static(ego_state, obstacles, config):
    """
    Create a static visualization of occlusion regions.
    
    Args:
        ego_state: Ego vehicle state [x, y, theta_t, v, delta, theta_s]
        obstacles: List of obstacles
        config: Configuration object
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.subplots_adjust(bottom=0.28)
    ax.set_xlim(-10, 250)
    ax.set_ylim(-8, 8)
    ax.set_aspect('equal')
    ax.set_title("Occlusion Region Visualization (Static View)", fontsize=16, fontweight='bold')
    ax.set_xlabel("X Position (m)", fontsize=12)
    ax.set_ylabel("Y Position (m)", fontsize=12)
    ax.grid(True, alpha=0.3)
    vehicle_fig_dir = os.path.join(os.path.dirname(__file__), "vehicle_figs")
    tractor_sprite = load_vehicle_sprite(vehicle_fig_dir, "tractor.png")
    trailer_sprite = load_vehicle_sprite(vehicle_fig_dir, "trailer.png")
    truck_sprite = load_vehicle_sprite(vehicle_fig_dir, "truck.png")
    car_sprite = load_vehicle_sprite(vehicle_fig_dir, "car.png", "red.png", "yellow.png")
    truck_threshold = getattr(config, "truck_length_threshold", 9.0)

    def add_oriented_sprite(image, center_x, center_y, length, width, heading, *, zorder, alpha=0.95):
        """Draw an oriented sprite centered at the vehicle pose."""
        half_length = length / 2.0
        half_width = width / 2.0
        extent = [
            center_x - half_length,
            center_x + half_length,
            center_y - half_width,
            center_y + half_width,
        ]
        artist = ax.imshow(
            image,
            extent=extent,
            origin="lower",
            zorder=zorder,
            alpha=alpha,
        )
        artist.set_transform(
            mtransforms.Affine2D().rotate_around(center_x, center_y, heading) + ax.transData
        )
        return artist
    
    # Draw road
    ax.axhline(config.y_min, color='gray', linestyle='-', linewidth=3, label='_nolegend_')
    ax.axhline(config.y_max, color='gray', linestyle='-', linewidth=3, label='_nolegend_')
    ax.axhline(0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='_nolegend_')
    
    # Draw ego vehicle
    x_ego, y_ego, theta_t, v_ego, delta, theta_s = ego_state
    geometry = get_ego_vehicle_geometry(ego_state, config)
    tractor_center_x, tractor_center_y = geometry["tractor_center"]
    trailer_center_x, trailer_center_y = geometry["trailer_center"]
    ego_front_x, ego_front_y = geometry["tractor_front"]
    
    # Ego tractor
    if tractor_sprite is not None:
        add_oriented_sprite(
            tractor_sprite,
            tractor_center_x,
            tractor_center_y,
            config.tractor_length,
            config.vehicle_width,
            theta_t,
            zorder=7,
        )
    else:
        tractor_rect = Rectangle(
            (tractor_center_x - config.tractor_length / 2, tractor_center_y - config.vehicle_width / 2),
            config.tractor_length,
            config.vehicle_width,
            angle=np.rad2deg(theta_t),
            color="blue",
            alpha=0.8,
            label="_nolegend_",
            zorder=7,
        )
        ax.add_patch(tractor_rect)
    
    # Ego trailer
    if trailer_sprite is not None:
        add_oriented_sprite(
            trailer_sprite,
            trailer_center_x,
            trailer_center_y,
            config.trailer_length,
            config.vehicle_width,
            theta_s,
            zorder=6,
        )
    else:
        trailer_rect = Rectangle(
            (trailer_center_x - config.trailer_length / 2, trailer_center_y - config.vehicle_width / 2),
            config.trailer_length,
            config.vehicle_width,
            angle=np.rad2deg(theta_s),
            color="lightblue",
            alpha=0.7,
            label="_nolegend_",
            zorder=6,
        )
        ax.add_patch(trailer_rect)
    
    # Mark ego front position
    ax.plot(
        ego_front_x,
        ego_front_y,
        "b*",
        markersize=15,
        label="_nolegend_",
        zorder=8,
    )
    
    # Draw obstacles
    for i, obs in enumerate(obstacles):
        if obs[6] < 0.5:
            continue
        
        x0, y0, heading, v_obs, length, width, _ = obs
        is_truck = length >= truck_threshold
        
        sprite = truck_sprite if is_truck else car_sprite
        if sprite is not None:
            add_oriented_sprite(
                sprite,
                x0,
                y0,
                length,
                width,
                heading,
                zorder=5 if is_truck else 4,
                alpha=1.0 if is_truck else 0.95,
            )
        else:
            color = "darkred" if is_truck else "orange"
            alpha = 0.7 if is_truck else 0.5
            obs_rect = Rectangle(
                (x0 - length / 2, y0 - width / 2),
                length,
                width,
                angle=np.rad2deg(heading),
                color=color,
                alpha=alpha,
                edgecolor="black",
                linewidth=2,
                label="_nolegend_",
            )
            ax.add_patch(obs_rect)
        
        # Add text label
        ax.text(x0, y0, f"{'Truck' if is_truck else 'Car'}\n{length:.0f}m", 
                ha='center', va='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Calculate and draw occlusion regions
    from semi_trailer_v2_with_occlusion import calculate_occlusion_regions
    
    try:
        occlusion_regions = calculate_occlusion_regions(
            ego_state, obstacles, config, config.dt, 
            trajectory_history=None, fov_range=200.0
        )
        
        print(f"\n{'='*60}")
        print(f"OCCLUSION ANALYSIS")
        print(f"{'='*60}")
        detected_count = len(occlusion_regions)
        print(f"Number of occlusion regions detected: {detected_count}")
        if detected_count == 0:
            print("  - No trucks ahead causing occlusion.\n")
            occlusion_regions = []
        else:
            occlusion_regions.sort(key=lambda region: region['distance'])
            primary_region = occlusion_regions[0]
            if detected_count > 1:
                print(f"  - Considering nearest occluding truck only (Obstacle #{primary_region['obstacle_id']}).")
            print()
            occlusion_regions = [primary_region]
        
        print(f"{'='*60}\n")
        
        # Draw each occlusion region
        for i, region in enumerate(occlusion_regions):
            polygon_verts = region['occluded_polygon']
            obs_id = region['obstacle_id']
            distance = region['distance']
            obs_pos = region['obstacle_pos']
            
            # Draw occlusion polygon
            occlusion_patch = PolygonPatch(
                polygon_verts, closed=True,
                facecolor='gray', alpha=0.25,
                edgecolor='darkgray', linewidth=2, linestyle='--',
                label='_nolegend_',
                zorder=2
            )
            ax.add_patch(occlusion_patch)
            
            # Draw tangent lines
            left_pt = region['tangent_left']['point']
            right_pt = region['tangent_right']['point']
            
            ax.plot([ego_front_x, left_pt[0]], [ego_front_y, left_pt[1]], 
                   'r--', linewidth=1.5, alpha=0.7, 
                   label='_nolegend_', zorder=3)
            ax.plot([ego_front_x, right_pt[0]], [ego_front_y, right_pt[1]], 
                   'r--', linewidth=1.5, alpha=0.7, zorder=3)
            
            # Print info
            print(f"Occlusion Region #{i+1}:")
            print(f"  - Caused by Obstacle #{obs_id}")
            print(f"  - Obstacle position: ({obs_pos[0]:.1f}, {obs_pos[1]:.1f}) m")
            print(f"  - Distance: {distance:.1f} m")
            print(f"  - Left tangent: ({left_pt[0]:.1f}, {left_pt[1]:.1f})")
            print(f"  - Right tangent: ({right_pt[0]:.1f}, {right_pt[1]:.1f})")
            print(f"  - Polygon area: {calculate_polygon_area(polygon_verts):.1f} m^2")
            occluded_ids = region.get('occluded_obstacles', [])
            if occluded_ids:
                print(f"  - Occluded obstacle IDs: {occluded_ids}")
            print()

        # Highlight occluded vehicles with circles
        occluded_lookup = {}
        for region in occlusion_regions:
            occluded_ids = region.get('occluded_obstacles', [])
            occluded_positions = region.get('occluded_positions', [])
            for obs_id, pos in zip(occluded_ids, occluded_positions):
                occluded_lookup[obs_id] = pos

        for obs_id, pos in occluded_lookup.items():
            obs = obstacles[obs_id]
            length = obs[4]
            width = obs[5]
            radius = np.hypot(length / 2.0, width / 2.0) + 1.0
            highlight = Circle(
                pos,
                radius=radius,
                edgecolor='yellow',
                linewidth=2.5,
                linestyle='--',
                fill=False,
                alpha=0.9,
                zorder=7,
            )
            ax.add_patch(highlight)
            
        # Draw field of view boundary
        fov_range = 200.0
        fov_arc_angles = np.linspace(-np.pi/2, np.pi/2, 100)
        fov_x = ego_front_x + fov_range * np.cos(theta_t + fov_arc_angles)
        fov_y = ego_front_y + fov_range * np.sin(theta_t + fov_arc_angles)
        ax.plot(fov_x, fov_y, 'b:', linewidth=1, alpha=0.4, label='_nolegend_')
        
    except Exception as e:
        print(f"Error calculating occlusion regions: {e}")
        import traceback
        traceback.print_exc()
    
    # Legend positioned below the road
    legend_handles = [
        Line2D([0], [0], color='gray', linewidth=3),
        Line2D([0], [0], color='green', linestyle='--', linewidth=2),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5),
        Patch(facecolor='gray', edgecolor='darkgray', linestyle='--', linewidth=2, alpha=0.25),
        Line2D([0], [0], color='blue', linestyle=':', linewidth=1),
        Line2D([0], [0], marker='*', color='b', markersize=12, linestyle='None'),
    ]
    legend_labels = [
        'Road Boundary',
        'Road Centerline',
        'Tangent Lines',
        'Occluded Region',
        'FoV Boundary',
        'Ego Front',
    ]

    def _fallback_marker(color):
        return Line2D(
            [0],
            [0],
            marker='s',
            linestyle='None',
            markersize=10,
            markerfacecolor=color,
            markeredgecolor=color,
        )

    if tractor_sprite is not None:
        legend_handles.append(LegendImage(tractor_sprite))
    else:
        legend_handles.append(_fallback_marker('blue'))
    legend_labels.append('Ego Tractor')

    if trailer_sprite is not None:
        legend_handles.append(LegendImage(trailer_sprite))
    else:
        legend_handles.append(_fallback_marker('lightblue'))
    legend_labels.append('Ego Trailer')

    if truck_sprite is not None:
        legend_handles.append(LegendImage(truck_sprite))
    else:
        legend_handles.append(_fallback_marker('darkred'))
    legend_labels.append('Truck')

    if car_sprite is not None:
        legend_handles.append(LegendImage(car_sprite))
    else:
        legend_handles.append(_fallback_marker('orange'))
    legend_labels.append('Car')

    legend_image_zoom = 0.1
    legend = ax.legend(
        legend_handles,
        legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        fontsize=10,
        framealpha=0.9,
        handler_map={LegendImage: HandlerLegendImage(legend_image_zoom)},
    )
    legend.get_frame().set_linewidth(0.8)
    
    # Add annotations
    """ax.text(0.98, 0.98, 
            "Gray region = Nearest occluded area\n"
            "Red dashed lines = Tangent lines\n"
            "Dark red = Trucks (create occlusion)\n"
            "Orange = Cars (no occlusion)",
            transform=ax.transAxes, fontsize=10,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))"""
    
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    output_path = "C:/Users/peter/OneDrive - The University of Hong Kong - Connect/Research/MPhil_study/paper/safety_critical_trailer/4trailermpc/semi_trailer_MPC/occlusion_analysis/occlusion_visualization_static.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print("Static visualization saved to 'occlusion_visualization_static.png'")
    plt.show()

def calculate_polygon_area(vertices):
    """Calculate area of polygon using shoelace formula."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def main():
    """Run occlusion visualization test."""
    print("="*60)
    print("OCCLUSION REGION VISUALIZATION TEST")
    print("="*60)
    
    # Initialize configuration
    config = SemiTrailerConfig()
    
    # Create test scenario with trucks
    obstacles = create_test_scenario_with_trucks()
    
    print(f"\nTest Scenario:")
    print(f"  - Number of obstacles: {len(obstacles)}")
    print(f"  - Truck length threshold: {config.truck_length_threshold} m")
    
    # Count trucks vs cars
    n_trucks = sum(1 for obs in obstacles if obs[4] >= config.truck_length_threshold)
    n_cars = len(obstacles) - n_trucks
    print(f"  - Trucks: {n_trucks}")
    print(f"  - Cars: {n_cars}")
    print()
    
    # Ego vehicle state: [x, y, theta_t, v, delta, theta_s]
    ego_state = np.array([
        0.0,      # x position
        0.0,      # y position
        0.0,      # tractor heading (straight ahead)
        10.0,     # speed (m/s)
        0.0,      # steering angle
        0.0       # trailer heading (aligned)
    ])
    
    print(f"Ego Vehicle State:")
    print(f"  - Position: ({ego_state[0]:.1f}, {ego_state[1]:.1f}) m")
    print(f"  - Heading: {np.rad2deg(ego_state[2]):.1f}°")
    print(f"  - Speed: {ego_state[3]:.1f} m/s ({ego_state[3]*3.6:.0f} km/h)")
    print()
    
    # Create visualization
    visualize_occlusion_static(ego_state, obstacles, config)
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nKey observations:")
    print("  ✓ Only trucks (length >= 9m) create occlusion regions")
    print("  ✓ Occlusion regions are bounded by tangent lines")
    print("  ✓ Regions extend behind obstacles to FoV limit")
    print("  ✓ Cars do not create occlusion (too small)")
    print("\nNext steps:")
    print("  1. Run full simulation: python semi_trailer_v2_with_occlusion.py")
    print("  2. See dynamic occlusion updates in animation")
    print("  3. Integrate occlusion awareness into MPC cost function")

if __name__ == "__main__":
    main()
