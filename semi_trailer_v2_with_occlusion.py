import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Rectangle, Circle
from matplotlib.path import Path
from matplotlib import transforms as mtransforms
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def _smooth_lane_change(progress: float) -> Tuple[float, float]:
    """Return smoothstep position and derivative for 0<=progress<=1."""
    progress = np.clip(progress, 0.0, 1.0)
    value = progress * progress * (3.0 - 2.0 * progress)  # 3p^2 - 2p^3
    derivative = (6.0 * progress - 6.0 * progress * progress)
    return value, derivative


def _evaluate_lane_change(initial_y: float, target_y: float, start_time: float,
                          duration: float, t: float) -> Tuple[float, float]:
    """Compute lateral position and velocity for a smooth lane change profile."""
    if duration <= 1e-3:
        return target_y, 0.0

    if t <= start_time:
        return initial_y, 0.0

    end_time = start_time + duration
    if t >= end_time:
        return target_y, 0.0

    progress = (t - start_time) / duration
    value, derivative = _smooth_lane_change(progress)
    delta_y = target_y - initial_y
    y = initial_y + value * delta_y
    y_dot = derivative * delta_y / duration
    return y, y_dot


def get_obstacle_state_at_time(obstacle: List[float], t: float,
                               behavior: Optional[Dict] = None) -> Dict[str, float]:
    """
    Evaluate the pose/velocity of an obstacle at time t, with optional lane-change behavior.
    Returns dictionary with x, y, heading, speed, vx, vy, length, width, active.
    """
    x0, y0, heading0, speed, length, width, active_flag = obstacle

    base_vx = speed * np.cos(heading0)
    base_vy = speed * np.sin(heading0)
    x = x0 + base_vx * t
    y = y0 + base_vy * t
    vx = base_vx
    vy = base_vy
    heading = heading0
    speed_mag = np.hypot(vx, vy)

    if behavior is not None and behavior.get('type') == 'lane_change':
        start_time = behavior.get('start_time', 0.0)
        duration = max(1e-3, behavior.get('duration', 3.0))
        initial_y = behavior.get('initial_y', y0)
        target_y = behavior.get('target_y', initial_y)
        long_speed = behavior.get('longitudinal_speed', max(speed, 0.1))
        x_anchor = behavior.get('x0', x0)

        x = x_anchor + long_speed * t
        vx = long_speed
        y, vy = _evaluate_lane_change(initial_y, target_y, start_time, duration, t)
        heading = np.arctan2(vy, vx)
        speed_mag = np.hypot(vx, vy)

    return {
        'x': x,
        'y': y,
        'heading': heading,
        'speed': speed_mag,
        'vx': vx,
        'vy': vy,
        'length': length,
        'width': width,
        'active': active_flag
    }


def build_behavior_lookup(obstacle_behaviors: Optional[List[Dict]]) -> Dict[int, Dict]:
    """Create a lookup dictionary keyed by obstacle index for quick access."""
    if not obstacle_behaviors:
        return {}
    lookup = {}
    for behavior in obstacle_behaviors:
        obs_index = behavior.get('obs_index')
        if obs_index is None:
            continue
        lookup[obs_index] = behavior
    return lookup


def build_obstacle_prediction_matrix(obstacles: List[List[float]],
                                     obstacle_behaviors: Optional[List[Dict]],
                                     config, current_time: float) -> np.ndarray:
    """
    Generate obstacle predictions for the full MPC horizon using the dynamic behaviors.
    Returns an array with shape (7 * max_obs, N + 1) matching the MPC expectation.
    """
    lookup = build_behavior_lookup(obstacle_behaviors)
    obs_array = np.zeros((7 * config.max_obs, config.N + 1))
    horizon_times = current_time + np.arange(config.N + 1) * config.dt

    for obs_idx in range(min(len(obstacles), config.max_obs)):
        obstacle = obstacles[obs_idx]
        behavior = lookup.get(obs_idx)
        active_flag = obstacle[6]
        if active_flag < 0.5:
            continue

        for k, t in enumerate(horizon_times):
            state = get_obstacle_state_at_time(obstacle, t, behavior)
            row = obs_idx * 7
            obs_array[row + 0, k] = state['x']
            obs_array[row + 1, k] = state['y']
            obs_array[row + 2, k] = state['heading']
            obs_array[row + 3, k] = state['length']
            obs_array[row + 4, k] = state['width']
            obs_array[row + 5, k] = state['speed']
            obs_array[row + 6, k] = state['active']

    return obs_array

# Import the configuration and MPC classes
# Make sure these files exist or update the import names accordingly
try:
    from semi_trailer_config import SemiTrailerConfig
except ImportError:
    print("Error: Cannot import SemiTrailerConfig from config.py")
    print("Make sure you have saved the config.py file with SemiTrailerConfig class")
    exit()

try:
    from semi_trailer_mpc import SemiTrailerMPC
except ImportError:
    try:
        from semi_trailer_mpc import SemiTrailerMPC
    except ImportError:
        print("Error: Cannot import SemiTrailerMPC")
        print("Make sure you have saved the mpc.py file with SemiTrailerMPC class")
        exit()


def generate_semi_trailer_reference_path(current_state, goal_state, N, dt, config, obstacles=None):
    """Generate reference trajectory for semi-trailer with realistic highway overtaking"""
    current_x = current_state[0]
    current_y = current_state[1]
    current_v = current_state[3]
    goal_x = goal_state[0]
    goal_y = goal_state[1]  # Should be 0.0 for center lane
    
    horizon_distance = max(config.ref_speed * N * dt, 10.0)
    goal_x_target = max(goal_x, current_x + horizon_distance)
    x_ref = np.linspace(current_x, goal_x_target, N + 1)
    y_ref = np.zeros(N + 1)
    theta_t_ref = np.zeros(N + 1)
    theta_s_ref = np.zeros(N + 1)
    
    # Define overtaking phases for REALISTIC HIGHWAY SCENARIO
    FOLLOW = 0        # Following in center lane
    WAIT = 1          # Waiting behind slow vehicle (oncoming traffic in overtaking lane)
    OVERTAKE = 2      # Overtaking in left lane (same lane as oncoming - RISKY!)
    RETURN = 3        # Returning to center lane
    GOAL_APPROACH = 4 # Final approach to goal
    
    # Analyze obstacles for HIGHWAY OVERTAKING scenario
    obstacles_same_direction = []  # Vehicles going same direction (need overtaking)
    obstacles_oncoming = []        # Vehicles coming from opposite direction in OVERTAKING LANE
    
    if obstacles is not None:
        for obs in obstacles:
            if obs[6] < 0.5:  # Skip inactive obstacles
                continue
            
            obs_x0, obs_y0 = obs[0], obs[1]
            obs_v, obs_heading = obs[3], obs[2]
            obs_length = obs[4]
            
            # Determine direction based on heading angle
            if abs(obs_heading) < np.pi/2:  # Heading ~0°, same direction as ego
                # Same direction vehicle (potential overtaking target)
                if (abs(obs_y0) < 1.5 and  # In center lane area
                    obs_x0 > current_x - 5.0):  # Not far behind
                    obstacles_same_direction.append({
                        'x0': obs_x0, 'y0': obs_y0, 'v': obs_v, 'heading': obs_heading,
                        'length': obs_length, 'x_current': obs_x0
                    })
            else:  # Heading ~180°, opposite direction
                # Oncoming vehicle in LEFT LANE (our overtaking lane!)
                if abs(obs_y0 + 2.0) < 1.0:  # Oncoming vehicle in left lane area
                    obstacles_oncoming.append({
                        'x0': obs_x0, 'y0': obs_y0, 'v': obs_v, 'heading': obs_heading,
                        'length': obs_length, 'x_current': obs_x0
                    })
    
    # Generate reference trajectory with TIMING-BASED OVERTAKING
    for i, x_pos in enumerate(x_ref):
        t_future = i * dt
        
        # Determine phase based on position and obstacles
        distance_to_goal = goal_x - x_pos
        
        # Phase determination
        if distance_to_goal < 8.0:
            # GOAL_APPROACH: Force center lane in final approach
            phase = GOAL_APPROACH
            y_target = goal_y  # Must be 0.0 (center lane)
            
        elif len(obstacles_same_direction) == 0:
            # No same-direction obstacles: stay in center or return
            if current_y < -1.0:  # Currently in left lane
                phase = RETURN
                return_distance = 15.0
                return_progress = min(1.0, (x_pos - current_x) / return_distance)
                y_target = -2.2 * (1.0 - return_progress)
            else:
                phase = FOLLOW
                y_target = 0.0
            
        else:
            # Check if we need to overtake same-direction obstacles
            need_overtake = False
            
            for obs in obstacles_same_direction:
                # Predict obstacle position at this time
                obs_x_future = obs['x0'] + obs['v'] * np.cos(obs['heading']) * t_future
                
                # Distance from our future position to obstacle
                distance_to_obs = obs_x_future - x_pos
                
                # Check if obstacle is ahead and blocking
                ego_safety_margin = config.total_length / 2 + 3.0
                obs_safety_margin = obs['length'] / 2 + 3.0
                
                if (distance_to_obs > -ego_safety_margin and 
                    distance_to_obs < obs_safety_margin + 25.0):
                    need_overtake = True
                    break
            
            if need_overtake:
                # CRITICAL: Check if overtaking is SAFE from oncoming traffic
                # This is the FACE-TO-FACE CONFLICT ANALYSIS
                overtaking_safe = True
                time_until_safe = 0.0
                
                for oncoming_obs in obstacles_oncoming:
                    # Predict oncoming vehicle position at overtaking time
                    oncoming_x_future = oncoming_obs['x0'] + oncoming_obs['v'] * np.cos(oncoming_obs['heading']) * t_future
                    
                    # Calculate if we'll have a HEAD-ON COLLISION in left lane
                    ego_front = x_pos + config.total_length / 2
                    ego_rear = x_pos - config.total_length / 2
                    oncoming_front = oncoming_x_future + oncoming_obs['length'] / 2
                    oncoming_rear = oncoming_x_future - oncoming_obs['length'] / 2
                    
                    # CRITICAL SAFETY MARGIN for head-on collision
                    safety_gap = 40.0  # Need 40m separation for semi-trailer overtaking
                    
                    # Check for potential HEAD-ON COLLISION
                    if (ego_rear < oncoming_front + safety_gap and 
                        ego_front > oncoming_rear - safety_gap):
                        overtaking_safe = False
                        
                        # Calculate when it will be safe (oncoming vehicle passes)
                        relative_speed = current_v - oncoming_obs['v'] * np.cos(oncoming_obs['heading'])
                        if relative_speed > 0:
                            time_until_safe = max(0, (oncoming_front - ego_rear + safety_gap) / relative_speed)
                        break
                
                # Decide phase based on FACE-TO-FACE safety analysis
                if overtaking_safe:
                    # SAFE to overtake in left lane
                    phase = OVERTAKE
                    y_target = -2.2  # LEFT lane (same as oncoming direction!)
                else:
                    # DANGEROUS - oncoming vehicle too close
                    phase = WAIT
                    y_target = 0.0   # Stay in center lane behind slow vehicle
            
            elif current_y < -1.0:  # Currently in left lane but obstacles passed
                phase = RETURN
                # Gradual return to center over 15 meters
                return_distance = 15.0
                return_progress = min(1.0, (x_pos - current_x) / return_distance)
                y_target = -2.2 * (1.0 - return_progress)  # Return from left lane
            else:
                phase = FOLLOW
                y_target = 0.0  # Center lane
        
        y_ref[i] = y_target
    
    # Speed reference with OVERTAKING TIMING
    v_ref = np.zeros(N + 1)
    terminal_speed = 15 if len(config.goal_pos) > 3 else config.ref_speed
    total_distance = max(1.0, goal_x - current_x)
    
    for k in range(N + 1):
        distance_to_goal = goal_x - x_ref[k]
        
        if distance_to_goal < 0.1:
            # AT GOAL: hold desired terminal speed instead of stopping
            v_ref[k] = terminal_speed
        elif distance_to_goal < 3.0:
            # FINAL APPROACH: maintain terminal cruising speed
            v_ref[k] = terminal_speed
        elif distance_to_goal < 8.0:
            # GOAL APPROACH: stay near desired terminal speed
            v_ref[k] = max(terminal_speed, config.ref_speed * 0.6)
        else:
            # Check if we're in WAIT phase (need to slow down behind slow vehicle)
            current_phase_here = FOLLOW  # Default
            
            # Quick phase check for this position
            if len(obstacles_same_direction) > 0 and len(obstacles_oncoming) > 0:
                for obs in obstacles_same_direction:
                    distance_to_obs = obs['x0'] - x_ref[k]
                    if distance_to_obs > 0 and distance_to_obs < 30.0:
                        # Check if oncoming prevents overtaking RIGHT NOW
                        for oncoming_obs in obstacles_oncoming:
                            oncoming_distance = abs(oncoming_obs['x0'] - x_ref[k])
                            if oncoming_distance < 50.0:  # Oncoming nearby
                                current_phase_here = WAIT
                                break
            
            # Set speed based on phase
            if current_phase_here == WAIT:
                # WAIT: Slow down to match slow vehicle (don't tailgate)
                slow_vehicle_speed = 4.0  # From obstacle definition
                v_ref[k] = max(slow_vehicle_speed * 0.9, 6.0)  # Slightly slower than obstacle
            elif distance_to_goal > 0.8 * total_distance:
                # ACCELERATION PHASE
                v_ref[k] = min(config.ref_speed, current_v + config.a_max * k * dt)
            elif distance_to_goal > 0.3 * total_distance:
                # CRUISING PHASE
                v_ref[k] = config.ref_speed
            else:
                # DECELERATION PHASE
                decel_factor = distance_to_goal / (0.3 * total_distance)
                v_ref[k] = max(terminal_speed, config.ref_speed * max(0.4, decel_factor))
    
    delta_ref = np.zeros(N + 1)
    
    return np.vstack([x_ref, y_ref, theta_t_ref, v_ref, delta_ref, theta_s_ref])


def animate_semi_trailer_trajectory(state_history, obstacles, mpc, config, n_steps,
                                    obstacle_behaviors=None):
    """Animation function updated for semi-trailer"""
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.subplots_adjust(bottom=0.25, right=0.95)
    ax.set_xlim(config.xlim)
    ax.set_ylim(config.ylim)
    ax.set_aspect('equal')
    ax.set_title("Highway Overtaking Maneuver of Tractor-Trailers", fontsize=16)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")

    # Prepare timestamped outputs for animation and periodic snapshots
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    snapshot_interval = getattr(config, "snapshot_interval", 1.0)
    snapshot_root = os.path.join(os.path.dirname(__file__), "ego_snapshots")
    snapshot_dir = os.path.join(snapshot_root, f"run_{timestamp}")
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_counter = 0
    next_snapshot_time = 0.0
    
    # Draw road boundaries and center line
    ax.axhline(config.y_min, color='gray', linestyle='-', linewidth=3, label='Road Boundary')
    ax.axhline(config.y_max, color='gray', linestyle='-', linewidth=3)
    ax.axhline(0, color='g', linestyle='--', linewidth=2, alpha=0.7, label='Road Centerline')

    # Start and goal markers
    ax.plot(config.start_pos[0], config.start_pos[1], 'go', markersize=10, label='Start')
    ax.plot(config.goal_pos[0], config.goal_pos[1], 'r*', markersize=15, label='Goal')
    
    # Trajectory line
    path_line, = ax.plot([], [], 'b-', lw=2, alpha=0.7, label='Actual Path')
    
    behavior_lookup = build_behavior_lookup(obstacle_behaviors)

    # Semi-trailer visualization
    vehicle_fig_dir = os.path.join(os.path.dirname(__file__), "vehicle_figs")
    tractor_img_path = os.path.join(vehicle_fig_dir, "tractor.png")
    trailer_img_path = os.path.join(vehicle_fig_dir, "trailer.png")

    tractor_artist = None
    trailer_artist = None
    tractor_use_image = False
    trailer_use_image = False

    if os.path.exists(tractor_img_path):
        tractor_image = plt.imread(tractor_img_path)
        tractor_artist = ax.imshow(
            tractor_image,
            extent=[0, 0, 0, 0],
            origin='lower',
            zorder=7,
            alpha=0.95,
            label='Tractor'
        )
        tractor_use_image = True
    else:
        print(f"Warning: tractor image not found at '{tractor_img_path}'. Using rectangle instead.")
        tractor_artist = Rectangle(
            (0, 0), config.tractor_length, config.vehicle_width,
            angle=0, color='blue', alpha=0.8, label='Tractor'
        )
        ax.add_patch(tractor_artist)

    if os.path.exists(trailer_img_path):
        trailer_image = plt.imread(trailer_img_path)
        trailer_artist = ax.imshow(
            trailer_image,
            extent=[0, 0, 0, 0],
            origin='lower',
            zorder=6,
            alpha=0.95,
            label='Trailer'
        )
        trailer_use_image = True
    else:
        print(f"Warning: trailer image not found at '{trailer_img_path}'. Using rectangle instead.")
        trailer_artist = Rectangle(
            (0, 0), config.trailer_length, config.vehicle_width,
            angle=0, color='lightblue', alpha=0.7, label='Trailer'
        )
        ax.add_patch(trailer_artist)
    
    # Articulation line (connection between tractor and trailer)
    articulation_line, = ax.plot([], [], 'r-', lw=3, alpha=0.8, label='Articulation')
    
    # Load obstacle sprite once (falls back to colored rectangles if missing)
    vehicle_fig_path = os.path.join(os.path.dirname(__file__), "vehicle_figs")
    car_image = None
    truck_image = None
    car_path = os.path.join(vehicle_fig_path, "red.png")
    truck_path = os.path.join(vehicle_fig_path, "truck.png")
    if os.path.exists(car_path):
        car_image = plt.imread(car_path)
    else:
        print(f"Warning: obstacle car image not found at '{car_path}'. Using rectangles instead.")
    if os.path.exists(truck_path):
        truck_image = plt.imread(truck_path)
    else:
        print(f"Warning: truck image not found at '{truck_path}'. No truck sprite will be used.")

    truck_length_threshold = getattr(config, "truck_length_threshold", 9.0)

    # Obstacle visualization
    obstacle_artists = []
    safety_boundaries = []
    occlusion_highlights = []
    obstacle_params = []
    for obs_idx, obs in enumerate(obstacles):
        if obs[6] < 0.5:
            continue  # Skip inactive obstacles
        
        x0, y0, heading, v_obs, length, width, active_flag = obs
        half_length = length / 2.0
        half_width = width / 2.0

        sprite = None
        is_truck = length >= truck_length_threshold
        if is_truck and truck_image is not None:
            sprite = truck_image
        elif car_image is not None:
            sprite = car_image

        if sprite is not None:
            extent = [x0 - half_length, x0 + half_length,
                      y0 - half_width, y0 + half_width]
            obs_artist = ax.imshow(
                sprite,
                extent=extent,
                origin='lower',
                zorder=6
            )
            obs_artist.set_transform(
                mtransforms.Affine2D().rotate_around(x0, y0, heading) + ax.transData
            )
        else:
            obs_artist = Rectangle(
                (0, 0), length, width, angle=np.rad2deg(heading),
                color='red', alpha=0.6
            )
            ax.add_patch(obs_artist)

            # Initialize rectangle placement so it is centered around the obstacle
            cos_h = np.cos(heading)
            sin_h = np.sin(heading)
            rot = np.array([[cos_h, -sin_h],
                            [sin_h,  cos_h]])
            offset = rot @ np.array([half_length, half_width])
            lower_left = np.array([x0, y0]) - offset
            obs_artist.set_xy(lower_left)

        obstacle_artists.append(obs_artist)
        
        # Safety boundary ellipse
        ellipse_width = length + config.total_length + 2*config.safety_distance
        ellipse_height = width + config.vehicle_width + 2*config.safety_distance
        ellipse = Ellipse((x0, y0), width=ellipse_width, height=ellipse_height,
                         angle=np.rad2deg(heading), fill=False, edgecolor='purple',
                         linestyle='--', linewidth=0.8, alpha=0.25, label='Safety Boundary')
        ax.add_patch(ellipse)
        safety_boundaries.append(ellipse)

        highlight_radius = np.hypot(length / 2.0, width / 2.0) + 1.0
        highlight = Circle(
            (x0, y0),
            radius=highlight_radius,
            edgecolor='yellow',
            linewidth=2.5,
            linestyle='--',
            fill=False,
            alpha=0.9,
            visible=False,
            zorder=9,
        )
        ax.add_patch(highlight)
        occlusion_highlights.append(highlight)

        # Store obstacle parameters for animation updates
        obstacle_params.append({
            'x0': x0,
            'y0': y0,
            'heading': heading,
            'speed': v_obs,
            'length': length,
            'width': width,
            'artist': obs_artist,
            'use_image': sprite is not None,
            'is_truck': is_truck,
            'index': obs_idx,
            'highlight_artist': highlight,
            'highlight_radius': highlight_radius,
            'base_obstacle': obs,
        })

    # Occlusion region visualization (will be updated dynamically)
    from matplotlib.patches import Polygon as PolygonPatch
    occlusion_patches = []
    max_occlusions = 5  # Maximum number of occlusion regions to track
    for _ in range(max_occlusions):
        occlusion_patch = PolygonPatch(
            np.array([[0, 0]]),  # Dummy initial polygon
            closed=True,
            facecolor='gray',
            alpha=0.2,
            edgecolor='darkgray',
            linewidth=1.5,
            linestyle='--',
            label='Occluded Region' if len(occlusion_patches) == 0 else None,
            zorder=1
        )
        occlusion_patch.set_visible(False)  # Initially hidden
        ax.add_patch(occlusion_patch)
        occlusion_patches.append(occlusion_patch)

    # Predicted trajectory
    pred_line, = ax.plot([], [], 'm--', lw=2, alpha=0.6, label='Predicted Path')
    pred_points = ax.plot([], [], 'mo', markersize=4, alpha=0.6)[0]
    
    # Reference trajectory
    ref_line, = ax.plot([], [], 'g:', lw=2, alpha=0.8, label='Reference Path')

    # Information text
    info_text = ax.text(0.18, -0.18, '', transform=ax.transAxes, fontsize=12,
                        ha='center', va='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    def get_semi_trailer_positions(x, y, theta_t, theta_s):
        """Calculate tractor and trailer positions and orientations"""
        # Tractor rear axle is at (x, y)
        # Tractor front is forward from rear axle
        tractor_front_x = x + config.L_t * np.cos(theta_t)
        tractor_front_y = y + config.L_t * np.sin(theta_t)
        
        # Tractor center for visualization
        tractor_center_x = x + config.L_t/2 * np.cos(theta_t)
        tractor_center_y = y + config.L_t/2 * np.sin(theta_t)
        
        # Trailer is behind tractor rear axle
        trailer_front_x = x
        trailer_front_y = y
        trailer_rear_x = x - config.L_s * np.cos(theta_s)
        trailer_rear_y = y - config.L_s * np.sin(theta_s)
        
        # Trailer center for visualization
        trailer_center_x = x - config.L_s/2 * np.cos(theta_s)
        trailer_center_y = y - config.L_s/2 * np.sin(theta_s)
        
        return (tractor_center_x, tractor_center_y, theta_t,
                trailer_center_x, trailer_center_y, theta_s,
                tractor_front_x, tractor_front_y, trailer_rear_x, trailer_rear_y)

    def update_vehicle_artist(artist, use_image, center_x, center_y, heading, length, width):
        """Update tractor/trailer artist with correct pose"""
        half_length = length / 2.0
        half_width = width / 2.0
        if use_image:
            extent = [center_x - half_length, center_x + half_length,
                      center_y - half_width, center_y + half_width]
            artist.set_extent(extent)
            artist.set_transform(
                mtransforms.Affine2D().rotate_around(center_x, center_y, heading) + ax.transData
            )
        else:
            artist.set_xy((center_x - half_length, center_y - half_width))
            artist.set_angle(np.rad2deg(heading))

    def init_animation():
        path_line.set_data([], [])
        initial_state = state_history[0]
        (tractor_cx, tractor_cy, theta_t,
         trailer_cx, trailer_cy, theta_s,
         _, _, _, _) = get_semi_trailer_positions(
            initial_state[0], initial_state[1], initial_state[2], initial_state[5]
        )
        update_vehicle_artist(
            tractor_artist, tractor_use_image, tractor_cx, tractor_cy, theta_t,
            config.tractor_length, config.vehicle_width
        )
        update_vehicle_artist(
            trailer_artist, trailer_use_image, trailer_cx, trailer_cy, theta_s,
            config.trailer_length, config.vehicle_width
        )
        articulation_line.set_data([], [])
        pred_line.set_data([], [])
        pred_points.set_data([], [])
        ref_line.set_data([], [])
        info_text.set_text('')
        # Hide all occlusion patches initially
        for patch in occlusion_patches:
            patch.set_visible(False)
        for highlight in occlusion_highlights:
            highlight.set_visible(False)
        return ([path_line, tractor_artist, trailer_artist, articulation_line, 
                pred_line, pred_points, ref_line, info_text] + obstacle_artists + 
                safety_boundaries + occlusion_patches + occlusion_highlights)

    def update_animation(frame):
        nonlocal snapshot_counter, next_snapshot_time
        idx = min(frame, len(state_history)-1)
        current_state = state_history[idx]
        x, y, theta_t, v, delta, theta_s = current_state
        
        # Update trajectory
        path_line.set_data(state_history[:idx+1, 0], state_history[:idx+1, 1])
        
        # Get vehicle positions
        (tractor_cx, tractor_cy, theta_t,
         trailer_cx, trailer_cy, theta_s,
         tractor_fx, tractor_fy, trailer_rx, trailer_ry) = get_semi_trailer_positions(x, y, theta_t, theta_s)
        
        # Update tractor and trailer visuals
        update_vehicle_artist(
            tractor_artist, tractor_use_image, tractor_cx, tractor_cy, theta_t,
            config.tractor_length, config.vehicle_width
        )
        update_vehicle_artist(
            trailer_artist, trailer_use_image, trailer_cx, trailer_cy, theta_s,
            config.trailer_length, config.vehicle_width
        )
        
        # Update articulation line (connection point)
        articulation_line.set_data([tractor_cx - config.tractor_length/2 * np.cos(theta_t),
                                   trailer_cx + config.trailer_length/2 * np.cos(theta_s)],
                                  [tractor_cy - config.tractor_length/2 * np.sin(theta_t),
                                   trailer_cy + config.trailer_length/2 * np.sin(theta_s)])
        
        # Calculate articulation angle
        phi = theta_t - theta_s
        
        # Update information text
        info_text.set_text(
            f"Time: {idx*config.dt:.1f}s\n"
            f"Speed: {v:.1f}m/s ({v*3.6:.0f}km/h)\n"
            f"Tractor Heading: {np.rad2deg(theta_t):.1f}°\n"
            f"Trailer Heading: {np.rad2deg(theta_s):.1f}°\n"
            f"Articulation: {np.rad2deg(phi):.1f}°\n"
            f"Steering: {np.rad2deg(delta):.1f}°"
        )
        
        # Save figure snapshots at the requested interval
        current_time = idx * config.dt
        epsilon = config.dt * 0.5
        if current_time + epsilon >= next_snapshot_time:
            snapshot_path = os.path.join(snapshot_dir, f"ego_snapshot_{snapshot_counter:04d}.png")
            fig.savefig(snapshot_path, dpi=150)
            snapshot_counter += 1
            next_snapshot_time += snapshot_interval

        # Update obstacle positions (dynamic obstacles)
        for obs_idx, params in enumerate(obstacle_params):
            obstacle = params['base_obstacle']
            behavior = behavior_lookup.get(params['index'])
            state = get_obstacle_state_at_time(obstacle, current_time, behavior)

            x_obs = state['x']
            y_obs = state['y']
            heading = state['heading']
            length = params['length']
            width = params['width']
            artist = params['artist']
            use_image = params['use_image']

            if use_image:
                half_length = length / 2.0
                half_width = width / 2.0
                extent = [x_obs - half_length, x_obs + half_length,
                          y_obs - half_width, y_obs + half_width]
                artist.set_extent(extent)
                artist.set_transform(
                    mtransforms.Affine2D().rotate_around(x_obs, y_obs, heading) + ax.transData
                )
            else:
                cos_h = np.cos(heading)
                sin_h = np.sin(heading)
                rot = np.array([[cos_h, -sin_h],
                                [sin_h,  cos_h]])
                offset = rot @ np.array([length / 2.0, width / 2.0])
                lower_left = np.array([x_obs, y_obs]) - offset

                artist.set_xy(lower_left)
                artist.set_angle(np.rad2deg(heading))

            try:
                safety_boundaries[obs_idx].set_center((x_obs, y_obs))
            except AttributeError:
                safety_boundaries[obs_idx].center = (x_obs, y_obs)
            safety_boundaries[obs_idx].angle = np.rad2deg(heading)

            highlight = params.get('highlight_artist')
            if highlight is not None:
                highlight.center = (x_obs, y_obs)
                highlight.set_radius(params.get('highlight_radius', 5.0))
                highlight.set_visible(False)
        
        # === Calculate and visualize occlusion regions ===
        try:
            occlusion_regions = calculate_occlusion_regions(
                current_state, obstacles, config, config.dt,
                state_history[:idx+1], fov_range=150.0,
                obstacle_behaviors=obstacle_behaviors,
                current_time=current_time
            )
            
            # Update occlusion patches
            for i, patch in enumerate(occlusion_patches):
                if i < len(occlusion_regions):
                    region = occlusion_regions[i]
                    polygon_verts = region['occluded_polygon']
                    patch.set_xy(polygon_verts)
                    patch.set_visible(True)
                else:
                    patch.set_visible(False)

            occluded_indices = set()
            for region in occlusion_regions:
                occluded_indices.update(region.get('occluded_obstacles', []))

            for params in obstacle_params:
                highlight = params.get('highlight_artist')
                if highlight is None:
                    continue
                if params['index'] in occluded_indices:
                    highlight.set_visible(True)
                else:
                    highlight.set_visible(False)
        except Exception as e:
            # If occlusion calculation fails, hide all patches
            for patch in occlusion_patches:
                patch.set_visible(False)
            for params in obstacle_params:
                highlight = params.get('highlight_artist')
                if highlight is not None:
                    highlight.set_visible(False)
        
        # Update predicted trajectory every few frames
        if frame % 3 == 0 and idx < len(state_history)-1:
            try:
                ref_traj = generate_semi_trailer_reference_path(
                    current_state, config.goal_pos, mpc.config.N, config.dt, config, obstacles)
                
                # Show reference trajectory
                ref_line.set_data(ref_traj[0, :], ref_traj[1, :])

                obs_predictions = build_obstacle_prediction_matrix(
                    obstacles, obstacle_behaviors, config, current_time
                )
                _, X_pred = mpc.solve(
                    current_state, ref_traj, obstacles, obstacle_predictions=obs_predictions
                )
                pred_line.set_data(X_pred[0, :], X_pred[1, :])
                pred_points.set_data(X_pred[0, :], X_pred[1, :])
            except:
                pass  # Skip prediction update if solve fails
        
        return ([path_line, tractor_artist, trailer_artist, articulation_line, 
                 pred_line, pred_points, ref_line, info_text] + obstacle_artists + 
                 safety_boundaries + occlusion_patches + occlusion_highlights)

    # Create animation
    ani = animation.FuncAnimation(
        fig, update_animation, frames=n_steps+1, init_func=init_animation,
        interval=config.animation_interval, blit=True, repeat=False
    )
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper center',
              bbox_to_anchor=(0.75, -0.12), borderaxespad=0.2, framealpha=0.95,
              facecolor='white', ncol=3)
    
    # Save animation
    gif_name = f"semi_trailer_mpc_animation_{timestamp}.gif"
    try:
        ani.save(gif_name, writer='pillow', fps=10)
        print(f"Animation saved as '{gif_name}'")
    except Exception as exc:
        print(f"Could not save animation (pillow/imagemagick not available): {exc}")
    
    print(f"Snapshot frames saved to '{snapshot_dir}' (interval: {snapshot_interval:.1f}s)")
    
    plt.tight_layout(rect=[0.0, 0.25, 1.0, 1.0])
    plt.show()
    return ani

def plot_semi_trailer_results(state_history, pred_history, config, obstacles=None, ttc_history=None):
    """Plot analysis results for semi-trailer including TTC"""
    plt.figure(figsize=(16, 14))
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.07, top=0.95, hspace=0.45, wspace=0.35)
    
    # 1. Trajectory plot
    plt.subplot(2, 3, 1)
    plt.plot(state_history[:, 0], state_history[:, 1], 'b-', linewidth=2, label='Actual Path')
    plt.plot(config.start_pos[0], config.start_pos[1], 'go', markersize=10, label='Start')
    plt.plot(config.goal_pos[0], config.goal_pos[1], 'r*', markersize=12, label='Goal')
    
    # Plot some predicted trajectories
    for i, pred in enumerate(pred_history):
        if i % 10 == 0:
            plt.plot(pred[0, :], pred[1, :], 'm--', alpha=0.3)
    
    plt.axhline(config.y_min, color='gray', linestyle='-', alpha=0.5)
    plt.axhline(config.y_max, color='gray', linestyle='-', alpha=0.5)
    plt.axhline(0, color='g', linestyle='--', alpha=0.5)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Semi-Trailer Trajectory')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(config.xlim)
    plt.ylim(config.ylim)
    
    # 2. Speed and heading evolution
    plt.subplot(2, 3, 2)
    time = np.arange(len(state_history)) * config.dt
    plt.plot(time, state_history[:, 3] * 3.6, 'b-', label='Speed (km/h)')
    plt.plot(time, np.rad2deg(state_history[:, 2]), 'g-', label='Tractor Heading (°)')
    plt.plot(time, np.rad2deg(state_history[:, 5]), 'c-', label='Trailer Heading (°)')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Speed and Heading Evolution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3. Articulation angle (most important for semi-trailer)
    plt.subplot(2, 3, 3)
    phi = np.rad2deg(state_history[:, 2] - state_history[:, 5])
    plt.plot(time, phi, 'r-', linewidth=2, label='Articulation Angle (°)')
    plt.axhline(60, color='r', linestyle='--', alpha=0.5, label='Limit (±60°)')
    plt.axhline(-60, color='r', linestyle='--', alpha=0.5)
    plt.axhline(0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Articulation Angle (°)')
    plt.title('Articulation Angle (Jackknife Prevention)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 4. Control inputs
    plt.subplot(2, 3, 4)
    if len(state_history) > 1:
        acceleration = np.diff(state_history[:, 3]) / config.dt
        delta_dot = np.diff(np.rad2deg(state_history[:, 4])) / config.dt
        time_ctrl = np.arange(len(acceleration)) * config.dt
        
        plt.plot(time_ctrl, acceleration, 'r-', label='Acceleration (m/s²)')
        plt.plot(time_ctrl, delta_dot, 'c-', label='Steering Rate (°/s)')
        plt.axhline(config.a_max, color='r', linestyle='--', alpha=0.5)
        plt.axhline(config.a_min, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Control Input')
        plt.title('Control Signals')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # 5. Lateral error from centerline
    plt.subplot(2, 3, 5)
    lateral_error = state_history[:, 1]
    plt.plot(time, lateral_error, 'm-', linewidth=2, label='Lateral Error (m)')
    plt.axhline(0, color='g', linestyle='--', alpha=0.5, label='Centerline')
    plt.axhline(config.y_max, color='r', linestyle='--', alpha=0.5, label='Road Boundary')
    plt.axhline(config.y_min, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Lateral Position (m)')
    plt.title('Lateral Position (Lane Keeping)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 6. Vehicle path curvature (optional, not necessary for overtaking metrics)
    """
    plt.subplot(3, 3, 6)
    if len(state_history) > 2:
        dx = np.diff(state_history[:, 0])
        dy = np.diff(state_history[:, 1])
        ddx = np.diff(dx)
        ddy = np.diff(dy)
        curvature = np.abs(dx[1:] * ddy - dy[1:] * ddx) / (dx[1:]**2 + dy[1:]**2)**(3/2)
        curvature = np.nan_to_num(curvature)
        
        time_curv = np.arange(len(curvature)) * config.dt
        plt.plot(time_curv, curvature, 'orange', linewidth=2, label='Path Curvature (1/m)')
        plt.xlabel('Time (s)')
        plt.ylabel('Curvature (1/m)')
        plt.title('Path Curvature')
        plt.grid(True, alpha=0.3)
        plt.legend()
    """
    # === NEW: 7. Time-to-Collision (TTC) Analysis ===
    if ttc_history is not None and len(ttc_history) > 0:
        plt.subplot(2, 3, 6)

        ttc_time = np.arange(1, len(ttc_history) + 1) * config.dt
        ttc_tractor = np.array([t['min_ttc_tractor'] for t in ttc_history])
        ttc_trailer = np.array([t['min_ttc_trailer'] for t in ttc_history])
        ttc_overall = np.array([t['min_ttc_overall'] for t in ttc_history])

        plt.plot(ttc_time, ttc_tractor, 'b-', linewidth=2, label='TTC Tractor Front', alpha=0.7)
        plt.plot(ttc_time, ttc_trailer, 'c-', linewidth=2, label='TTC Trailer Rear', alpha=0.7)
        plt.plot(ttc_time, ttc_overall, 'r-', linewidth=2.5, label='TTC Overall (Min)', alpha=0.9)
        
        # Safety thresholds
        plt.axhline(5.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='Warning (5s)')
        plt.axhline(3.0, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Critical (3s)')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Time-to-Collision (s)')
        plt.title('Time-to-Collision Analysis')
        plt.ylim(0, 20)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        """
        # === NEW: 8. TTC Histogram ===
        plt.subplot(3, 3, 8)
        
        # Filter out infinite values
        ttc_overall_finite = [t for t in ttc_overall if t < 20.0]
        
        if len(ttc_overall_finite) > 0:
            plt.hist(ttc_overall_finite, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            plt.axvline(3.0, color='red', linestyle='--', linewidth=2, label='Critical (3s)')
            plt.axvline(5.0, color='orange', linestyle='--', linewidth=2, label='Warning (5s)')
            
            # Statistics
            mean_ttc = np.mean(ttc_overall_finite)
            min_ttc = np.min(ttc_overall_finite)
            plt.axvline(mean_ttc, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_ttc:.1f}s')
            
            plt.xlabel('Time-to-Collision (s)')
            plt.ylabel('Frequency')
            plt.title(f'TTC Distribution (Min: {min_ttc:.1f}s)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # === NEW: 9. Collision Type Distribution ===
        plt.subplot(3, 3, 9)
        
        collision_types = [t['collision_type'] for t in ttc_history]
        type_counts = {
            'same_direction': collision_types.count('same_direction'),
            'oncoming': collision_types.count('oncoming'),
            'none': collision_types.count('none')
        }
        
        # Only plot non-zero categories
        categories = [k for k, v in type_counts.items() if v > 0]
        counts = [type_counts[k] for k in categories]
        colors = {'same_direction': 'blue', 'oncoming': 'red', 'none': 'green'}
        bar_colors = [colors[cat] for cat in categories]
        
        plt.bar(categories, counts, color=bar_colors, alpha=0.7, edgecolor='black')
        plt.xlabel('Conflict Type')
        plt.ylabel('Frequency')
        plt.title('Critical Obstacle Types')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        total = sum(counts)
        for i, (cat, count) in enumerate(zip(categories, counts)):
            percentage = (count / total) * 100
            plt.text(i, count, f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
    """
    plt.show()
"""    
def plot_semi_trailer_results(state_history, pred_history, config):
    # Plot analysis results for semi-trailer 
    plt.figure(figsize=(15, 12))
    
    # 1. Trajectory plot
    plt.subplot(3, 2, 1)
    plt.plot(state_history[:, 0], state_history[:, 1], 'b-', linewidth=2, label='Actual Path')
    plt.plot(config.start_pos[0], config.start_pos[1], 'go', markersize=10, label='Start')
    plt.plot(config.goal_pos[0], config.goal_pos[1], 'r*', markersize=12, label='Goal')
    
    # Plot some predicted trajectories
    for i, pred in enumerate(pred_history):
        if i % 10 == 0:
            plt.plot(pred[0, :], pred[1, :], 'm--', alpha=0.3)
    
    plt.axhline(config.y_min, color='gray', linestyle='-', alpha=0.5)
    plt.axhline(config.y_max, color='gray', linestyle='-', alpha=0.5)
    plt.axhline(0, color='g', linestyle='--', alpha=0.5)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Semi-Trailer Trajectory')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(config.xlim)
    plt.ylim(config.ylim)
    
    # 2. Speed and heading evolution
    plt.subplot(3, 2, 2)
    time = np.arange(len(state_history)) * config.dt
    plt.plot(time, state_history[:, 3] * 3.6, 'b-', label='Speed (km/h)')
    plt.plot(time, np.rad2deg(state_history[:, 2]), 'g-', label='Tractor Heading (°)')
    plt.plot(time, np.rad2deg(state_history[:, 5]), 'c-', label='Trailer Heading (°)')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Speed and Heading Evolution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3. Articulation angle (most important for semi-trailer)
    plt.subplot(3, 2, 3)
    phi = np.rad2deg(state_history[:, 2] - state_history[:, 5])
    plt.plot(time, phi, 'r-', linewidth=2, label='Articulation Angle (°)')
    plt.axhline(60, color='r', linestyle='--', alpha=0.5, label='Limit (±60°)')
    plt.axhline(-60, color='r', linestyle='--', alpha=0.5)
    plt.axhline(0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Articulation Angle (°)')
    plt.title('Articulation Angle (Jackknife Prevention)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 4. Control inputs
    plt.subplot(3, 2, 4)
    if len(state_history) > 1:
        acceleration = np.diff(state_history[:, 3]) / config.dt
        delta_dot = np.diff(np.rad2deg(state_history[:, 4])) / config.dt
        time_ctrl = np.arange(len(acceleration)) * config.dt
        
        plt.plot(time_ctrl, acceleration, 'r-', label='Acceleration (m/s²)')
        plt.plot(time_ctrl, delta_dot, 'c-', label='Steering Rate (°/s)')
        plt.axhline(config.a_max, color='r', linestyle='--', alpha=0.5)
        plt.axhline(config.a_min, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Control Input')
        plt.title('Control Signals')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # 5. Lateral error from centerline
    plt.subplot(3, 2, 5)
    lateral_error = state_history[:, 1]
    plt.plot(time, lateral_error, 'm-', linewidth=2, label='Lateral Error (m)')
    plt.axhline(0, color='g', linestyle='--', alpha=0.5, label='Centerline')
    plt.axhline(config.y_max, color='r', linestyle='--', alpha=0.5, label='Road Boundary')
    plt.axhline(config.y_min, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Lateral Position (m)')
    plt.title('Lateral Position (Lane Keeping)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 6. Vehicle path curvature
    plt.subplot(3, 2, 6)
    if len(state_history) > 2:
        # Calculate path curvature
        dx = np.diff(state_history[:, 0])
        dy = np.diff(state_history[:, 1])
        ddx = np.diff(dx)
        ddy = np.diff(dy)
        curvature = np.abs(dx[1:] * ddy - dy[1:] * ddx) / (dx[1:]**2 + dy[1:]**2)**(3/2)
        curvature = np.nan_to_num(curvature)  # Handle division by zero
        
        time_curv = np.arange(len(curvature)) * config.dt
        plt.plot(time_curv, curvature, 'orange', linewidth=2, label='Path Curvature (1/m)')
        plt.xlabel('Time (s)')
        plt.ylabel('Curvature (1/m)')
        plt.title('Path Curvature')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    plt.show()

"""
def create_occluded_lane_change_scenario(config, seed=None):
    """
    Deterministic scenario that keeps a slow truck in the ego lane while one or more cars
    hidden behind it decide to overtake into the oncoming lane with a smooth lane change.
    Returns a tuple (obstacles, obstacle_behaviors).
    """
    rng = np.random.default_rng(seed)
    longitudinal_offset = rng.uniform(-2.0, 2.0)

    obstacles = [
        # Long truck directly ahead of the ego occupying the center lane (occluding vehicle)
        [55.0 + longitudinal_offset, 0.0, np.deg2rad(0.0), 3.6, 14.0, 4.0, 1.0],
        # Occluded sedan that will initiate a gentle lane change into the oncoming lane
        [92.0 + longitudinal_offset, 0.2, np.deg2rad(0.0), 5.3, 4.8, 2.1, 1.0],
        # Second sedan further ahead, also hidden behind the truck and lane-changing later
        [130.0 + longitudinal_offset, -0.1, np.deg2rad(0.0), 5.8, 4.6, 2.0, 1.0],
        # Oncoming compact car to make the overtake lane contested
        [190.0, -3.0, np.deg2rad(180.0), 6.0, 4.5, 2.0, 1.0],
        # Oncoming truck further out for continued occlusion pressure
        [235.0, -3.5, np.deg2rad(180.0), 5.0, 12.0, 4.0, 1.0],
    ]

    obstacle_behaviors = [
        {
            'obs_index': 1,
            'type': 'lane_change',
            'start_time': 5.5,
            'duration': 3.5,
            'initial_y': obstacles[1][1],
            'target_y': -2.4,
            'longitudinal_speed': obstacles[1][3],
            'x0': obstacles[1][0],
        },
        {
            'obs_index': 2,
            'type': 'lane_change',
            'start_time': 8.0,
            'duration': 4.0,
            'initial_y': obstacles[2][1],
            'target_y': -2.8,
            'longitudinal_speed': obstacles[2][3],
            'x0': obstacles[2][0],
        },
    ]

    return obstacles, obstacle_behaviors


def create_dual_overtake_scenario(config, seed=None):
    """
    Scenario where a slow convoy causes two overtaking maneuvers:
    the ego semi-trailer wants to pass a blocking truck while a passenger car,
    located further ahead, also merges into the passing lane to overtake the
    leading truck. All vehicles travel in the same direction, so there are no
    oncoming conflicts, but timing and spacing become critical.
    """
    rng = np.random.default_rng(seed)
    longitudinal_jitter = float(rng.uniform(-2.0, 2.0))

    # Speeds in m/s (approx 10–25 km/h) to keep the convoy slow relative to ego
    blocker_speed = 3.8
    car_speed = 6.5
    lead_truck_speed = 3.2
    far_truck_speed = 4.2

    obstacles = [
        # Truck directly ahead of the ego (forces ego to overtake)
        [72.0 + longitudinal_jitter, 0.1, np.deg2rad(0.0), blocker_speed, 15.0, 4.0, 1.0],
        # Passenger car between two trucks that will merge left to overtake
        [98.0 + longitudinal_jitter, 0.0, np.deg2rad(0.0), car_speed, 5.0, 2.2, 1.0],
        # Leading truck that the passenger car also wants to pass
        [142.0 + longitudinal_jitter, -0.1, np.deg2rad(0.0), lead_truck_speed, 16.0, 4.2, 1.0],
        # Optional second leading truck to extend the convoy
        [168.0 + longitudinal_jitter, 0.2, np.deg2rad(0.0), far_truck_speed, 14.0, 4.0, 1.0],
    ]

    obstacle_behaviors = [
        {
            'obs_index': 1,  # passenger car
            'type': 'lane_change',
            'start_time': 4.0,
            'duration': 4.5,
            'initial_y': obstacles[1][1],
            'target_y': -5.0,  # passing lane (same travel direction)
            'longitudinal_speed': car_speed,
            'x0': obstacles[1][0],
        },
    ]

    return obstacles, obstacle_behaviors


def create_semi_trailer_scenario(max_obstacles=15, seed=None, oncoming_ratio=0.35, min_spacing=25.0):
    """Randomly generate obstacles for the semi-trailer scenario with spacing constraints.

    Obstacles: [x0, y0, heading, speed, length, width, active_flag]
        * x0 sampled within [0, 400] while maintaining at least `min_spacing` longitudinal gap
          between vehicles traveling in the same direction
        * Same-direction vehicles (heading = 0 rad) use y0 in {1.0, 2.0}
        * Oncoming vehicles (heading = pi rad) use y0 in {-1.0, -2.0}
        * All obstacles travel at 2.0 m/s
        * Each scenario includes 1-2 trucks by sampling longer vehicles (fixed 4.0 m width, truck sprite used when available)
        * Up to `max_obstacles` vehicles are generated (at least 5 for variety)
    """
    # manually design the traffic scenario for testing:
    """ obstacles = [
        # Slow truck ahead (in same lane) - VERY slow for easy overtaking
        [33.0, 2.0, np.deg2rad(0), 2.0, 4.5, 2.0, 1.0],     # slow front vehicle
        [70.0, 2.0, np.deg2rad(0), 2.0, 4.5, 2.0, 1.0],
        [110.0, 2.0, np.deg2rad(0), 2.0, 7.5, 2.0, 1.0],
        [150.0, 2.0, np.deg2rad(0), 2.0, 4.5, 2.0, 1.0],
        [220.0, 2.0, np.deg2rad(0), 2.0, 4.5, 2.0, 1.0],     # slow front vehicle (multiple for length)
        [250.0, 2.0, np.deg2rad(0), 2.5, 7.5, 2.0, 1.0],
        [290.0, 2.0, np.deg2rad(0), 2.5, 4.5, 2.0, 1.0],
        # Oncoming car (in opposite lane)
        # [145.0, -1.0, np.deg2rad(180), 2.0, 7.5, 2.0, 1.0],   # fast oncoming car (further away)
        [170.0, -3.0, np.deg2rad(180), 2.0, 4.5, 2.0, 0.0],
        [210.0, -3.0, np.deg2rad(180), 2.0, 4.5, 2.0, 1.0],
        [280.0, -4.0, np.deg2rad(180), 2.0, 7.5, 2.0, 1.0],
        [325.0, -4.0, np.deg2rad(180), 2.0, 8.5, 4.0, 1.0]
    ]
    
    return obstacles
    """
    # Randomly generate obstacles for testing:
    rng = np.random.default_rng(seed)
    n_obstacles = int(rng.integers(5, max_obstacles + 1))

    car_length_choices = np.array([4.5, 6.0, 7.5])
    car_width_choices = np.array([3.0, 4.0])
    truck_length_choices = np.array([10.0, 12.0, 14.0])
    truck_width = 4.0

    same_lane_y = np.array([3.0, 4.0])
    oncoming_lane_y = np.array([-3.0, -4.0])

    obstacles = []
    headings = []

    n_trucks = int(rng.integers(1, 3))
    n_trucks = min(n_trucks, n_obstacles)
    truck_indices = set()
    if n_obstacles > 0 and n_trucks > 0:
        selected = rng.choice(n_obstacles, size=n_trucks, replace=False)
        truck_indices = set(np.atleast_1d(selected).tolist())

    for idx in range(n_obstacles):
        is_oncoming = rng.random() < oncoming_ratio
        heading = np.deg2rad(180.0) if is_oncoming else np.deg2rad(0.0)
        y0 = float(rng.choice(oncoming_lane_y if is_oncoming else same_lane_y))

        if idx in truck_indices:
            length = float(rng.choice(truck_length_choices))
            width = truck_width
        else:
            length = float(rng.choice(car_length_choices))
            width = float(rng.choice(car_width_choices))

        obstacles.append([0.0, y0, heading, 2.0, length, width, 1.0])
        headings.append(heading)

    # Ensure at least one vehicle in each direction for richer interaction
    if not any(np.isclose(h, np.deg2rad(0.0)) for h in headings):
        obstacles[0][2] = np.deg2rad(0.0)
        obstacles[0][1] = float(rng.choice(same_lane_y))
        headings[0] = np.deg2rad(0.0)
    if not any(np.isclose(h, np.deg2rad(180.0)) for h in headings):
        obstacles[-1][2] = np.deg2rad(180.0)
        obstacles[-1][1] = float(rng.choice(oncoming_lane_y))
        headings[-1] = np.deg2rad(180.0)

    def assign_positions(indices):
        """Assign x positions with guaranteed spacing for a subset of obstacles."""
        if not indices:
            return
        count = len(indices)
        usable_length = 400.0 - min_spacing * (count - 1)
        usable_length = max(0.0, usable_length)
        base_positions = np.sort(rng.uniform(50.0, usable_length, size=count))
        for order, idx in enumerate(indices):
            obstacles[idx][0] = float(base_positions[order] + order * min_spacing)

    same_direction_indices = [
        idx for idx, obs in enumerate(obstacles)
        if np.isclose(obs[2], np.deg2rad(0.0))
    ]
    oncoming_indices = [
        idx for idx, obs in enumerate(obstacles)
        if np.isclose(obs[2], np.deg2rad(180.0))
    ]

    assign_positions(same_direction_indices)
    assign_positions(oncoming_indices)

    obstacles.sort(key=lambda obs: obs[0])
    return obstacles


def calculate_ttc(ego_state, obstacles, config, dt, trajectory_history=None,
                  obstacle_behaviors=None, current_time=None):
    """
    Calculate Time-to-Collision (TTC) for tractor-trailer with all obstacles.
    
    Returns:
        ttc_results: dict with keys:
            - 'min_ttc_tractor': minimum TTC for tractor front
            - 'min_ttc_trailer': minimum TTC for trailer rear
            - 'min_ttc_overall': minimum TTC across both
            - 'ttc_tractor_per_obs': list of TTC values for tractor to each obstacle
            - 'ttc_trailer_per_obs': list of TTC values for trailer to each obstacle
            - 'critical_obstacle_id': index of most critical obstacle
            - 'collision_type': 'same_direction', 'oncoming', or 'none'
    """
    x, y, theta_t, v, delta, theta_s = ego_state
    
    # Calculate tractor front and trailer rear positions
    tractor_front_x = x + config.L_t * np.cos(theta_t)
    tractor_front_y = y + config.L_t * np.sin(theta_t)
    
    trailer_rear_x = x - config.L_s * np.cos(theta_s)
    trailer_rear_y = y - config.L_s * np.sin(theta_s)
    
    # Ego velocity components
    v_ego_x = v * np.cos(theta_t)
    v_ego_y = v * np.sin(theta_t)
    
    ttc_tractor_list = []
    ttc_trailer_list = []
    min_ttc_tractor = np.inf
    min_ttc_trailer = np.inf
    critical_obs_id = -1
    collision_type = 'none'
    
    behavior_lookup = build_behavior_lookup(obstacle_behaviors)

    if current_time is not None:
        t_elapsed = current_time
    elif trajectory_history is not None and len(trajectory_history) > 0:
        t_elapsed = (len(trajectory_history) - 1) * dt
    else:
        t_elapsed = 0.0

    for obs_idx, obs in enumerate(obstacles):
        if obs[6] < 0.5:  # Skip inactive obstacles
            ttc_tractor_list.append(np.inf)
            ttc_trailer_list.append(np.inf)
            continue
        
        obs_x0, obs_y0, obs_heading, obs_v, obs_length, obs_width, _ = obs

        behavior = behavior_lookup.get(obs_idx)
        state = get_obstacle_state_at_time(obs, t_elapsed, behavior)
        obs_x = state['x']
        obs_y = state['y']
        obs_heading = state['heading']
        v_obs_x = state['vx']
        v_obs_y = state['vy']
        
        # === TTC for TRACTOR FRONT ===
        # Relative position and velocity
        dx_tractor = obs_x - tractor_front_x
        dy_tractor = obs_y - tractor_front_y
        dvx_tractor = v_obs_x - v_ego_x
        dvy_tractor = v_obs_y - v_ego_y
        
        # Distance and relative speed
        distance_tractor = np.sqrt(dx_tractor**2 + dy_tractor**2)
        relative_speed_tractor = np.sqrt(dvx_tractor**2 + dvy_tractor**2)
        
        # Check if vehicles are approaching (dot product < 0)
        approach_rate_tractor = -(dx_tractor * dvx_tractor + dy_tractor * dvy_tractor) / (distance_tractor + 1e-6)
        
        if approach_rate_tractor > 0.1:  # Approaching
            # Account for vehicle dimensions
            safety_distance_tractor = (config.tractor_length + obs_length) / 2 + config.safety_distance
            effective_distance_tractor = max(0.0, distance_tractor - safety_distance_tractor)
            ttc_tractor = effective_distance_tractor / approach_rate_tractor
        else:
            ttc_tractor = np.inf  # Not approaching
        
        ttc_tractor_list.append(ttc_tractor)
        
        # === TTC for TRAILER REAR ===
        dx_trailer = obs_x - trailer_rear_x
        dy_trailer = obs_y - trailer_rear_y
        
        # Trailer velocity (different from tractor due to articulation)
        v_trailer_x = v * np.cos(theta_s)
        v_trailer_y = v * np.sin(theta_s)
        
        dvx_trailer = v_obs_x - v_trailer_x
        dvy_trailer = v_obs_y - v_trailer_y
        
        distance_trailer = np.sqrt(dx_trailer**2 + dy_trailer**2)
        approach_rate_trailer = -(dx_trailer * dvx_trailer + dy_trailer * dvy_trailer) / (distance_trailer + 1e-6)
        
        if approach_rate_trailer > 0.1:  # Approaching
            safety_distance_trailer = (config.trailer_length + obs_length) / 2 + config.safety_distance
            effective_distance_trailer = max(0.0, distance_trailer - safety_distance_trailer)
            ttc_trailer = effective_distance_trailer / approach_rate_trailer
        else:
            ttc_trailer = np.inf
        
        ttc_trailer_list.append(ttc_trailer)
        
        # Track minimum TTC
        if ttc_tractor < min_ttc_tractor:
            min_ttc_tractor = ttc_tractor
            critical_obs_id = obs_idx
            # Determine collision type
            if abs(obs_heading) < np.pi/2:
                collision_type = 'same_direction'
            else:
                collision_type = 'oncoming'
        
        if ttc_trailer < min_ttc_trailer:
            min_ttc_trailer = ttc_trailer
            if ttc_trailer < min_ttc_tractor:  # Trailer is more critical
                critical_obs_id = obs_idx
                if abs(obs_heading) < np.pi/2:
                    collision_type = 'same_direction'
                else:
                    collision_type = 'oncoming'
    
    min_ttc_overall = min(min_ttc_tractor, min_ttc_trailer)
    
    # Cap TTC at reasonable maximum for plotting (e.g., 20 seconds)
    MAX_TTC = 20.0
    min_ttc_tractor = min(min_ttc_tractor, MAX_TTC)
    min_ttc_trailer = min(min_ttc_trailer, MAX_TTC)
    min_ttc_overall = min(min_ttc_overall, MAX_TTC)
    
    return {
        'min_ttc_tractor': min_ttc_tractor,
        'min_ttc_trailer': min_ttc_trailer,
        'min_ttc_overall': min_ttc_overall,
        'ttc_tractor_per_obs': ttc_tractor_list,
        'ttc_trailer_per_obs': ttc_trailer_list,
        'critical_obstacle_id': critical_obs_id,
        'collision_type': collision_type
    }

def calculate_occlusion_regions(ego_state, obstacles, config, dt, trajectory_history=None,
                                fov_range=150.0, obstacle_behaviors=None, current_time=None):
    """
    Calculate occlusion regions caused by truck obstacles in front of the ego vehicle.
    
    Args:
        ego_state: Current state of ego vehicle [x, y, theta_t, v, delta, theta_s]
        obstacles: List of obstacle parameters
        config: Configuration object
        dt: Time step
        trajectory_history: Optional trajectory history for obstacle position prediction
        fov_range: Maximum field of view range in meters (default: 150m)
    
    Returns:
        occlusion_regions: List of dictionaries containing occlusion information
    """
    x_ego, y_ego, theta_t, v_ego, delta, theta_s = ego_state
    
    # Ego vehicle position (tractor front as reference point)
    ego_front_x = x_ego + config.L_t * np.cos(theta_t)
    ego_front_y = y_ego + config.L_t * np.sin(theta_t)
    
    # Truck threshold
    truck_length_threshold = getattr(config, 'truck_length_threshold', 9.0)
    
    occlusion_regions = []
    behavior_lookup = build_behavior_lookup(obstacle_behaviors)
    if current_time is not None:
        t_elapsed = current_time
    elif trajectory_history is not None and len(trajectory_history) > 0:
        t_elapsed = (len(trajectory_history) - 1) * dt
    else:
        t_elapsed = 0.0
    
    for obs_idx, obs in enumerate(obstacles):
        if obs[6] < 0.5:  # Skip inactive obstacles
            continue
        
        obs_x0, obs_y0, obs_heading, obs_v, obs_length, obs_width, _ = obs
        
        # Check if obstacle is a truck
        is_truck = obs_length >= truck_length_threshold
        
        if not is_truck:
            continue  # Only trucks cause occlusion
        
        behavior = behavior_lookup.get(obs_idx)
        state = get_obstacle_state_at_time(obs, t_elapsed, behavior)
        obs_x = state['x']
        obs_y = state['y']
        obs_heading = state['heading']
        
        # Check if obstacle is in front of ego vehicle
        dx = obs_x - ego_front_x
        dy = obs_y - ego_front_y
        distance_to_obs = np.sqrt(dx**2 + dy**2)
        
        # Check if obstacle is within FoV
        if distance_to_obs > fov_range or distance_to_obs < 1.0:
            continue
        
        # Check if obstacle is roughly ahead (within ±90 degrees)
        angle_to_obs = np.arctan2(dy, dx)
        relative_angle = angle_to_obs - theta_t
        relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))  # Normalize
        
        if abs(relative_angle) > np.pi / 2:
            continue  # Obstacle is behind ego vehicle
        
        # Check if heading directions are compatible (either same or opposite lane traffic)
        heading_diff = obs_heading - theta_t
        heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
        heading_diff_abs = abs(heading_diff)
        same_direction = heading_diff_abs <= np.pi / 3  # within 60 deg
        opposing_direction = abs(heading_diff_abs - np.pi) <= np.pi / 3  # within 60 deg of head-on
        if not (same_direction or opposing_direction):
            continue
        
        # Calculate tangent lines from ego position to obstacle
        half_length = obs_length / 2.0
        half_width = obs_width / 2.0
        
        # Obstacle corners in world frame
        cos_h = np.cos(obs_heading)
        sin_h = np.sin(obs_heading)
        
        # Four corners of the obstacle
        corners = [
            (obs_x + half_length * cos_h - half_width * sin_h, 
             obs_y + half_length * sin_h + half_width * cos_h),  # Front-left
            (obs_x + half_length * cos_h + half_width * sin_h, 
             obs_y + half_length * sin_h - half_width * cos_h),  # Front-right
            (obs_x - half_length * cos_h + half_width * sin_h, 
             obs_y - half_length * sin_h - half_width * cos_h),  # Rear-right
            (obs_x - half_length * cos_h - half_width * sin_h, 
             obs_y - half_length * sin_h + half_width * cos_h),  # Rear-left
        ]
        
        # Calculate angles to each corner from ego position
        corner_angles = []
        for corner_x, corner_y in corners:
            angle = np.arctan2(corner_y - ego_front_y, corner_x - ego_front_x)
            corner_angles.append((angle, corner_x, corner_y))
        
        # Sort corners by angle
        corner_angles.sort(key=lambda x: x[0])
        
        # Find leftmost and rightmost visible corners (tangent points)
        left_tangent_angle, left_x, left_y = corner_angles[-1]
        right_tangent_angle, right_x, right_y = corner_angles[0]
        
        # Calculate tangent line directions
        left_direction = np.array([left_x - ego_front_x, left_y - ego_front_y])
        left_direction = left_direction / (np.linalg.norm(left_direction) + 1e-6)
        
        right_direction = np.array([right_x - ego_front_x, right_y - ego_front_y])
        right_direction = right_direction / (np.linalg.norm(right_direction) + 1e-6)
        
        # Calculate occlusion polygon (extended behind the obstacle)
        occlusion_depth = fov_range
        
        # Points for occlusion polygon
        left_far = np.array([left_x, left_y]) + left_direction * occlusion_depth
        right_far = np.array([right_x, right_y]) + right_direction * occlusion_depth
        
        # Create polygon vertices
        polygon_vertices = np.array([
            [ego_front_x, ego_front_y],
            [left_x, left_y],
            left_far,
            right_far,
            [right_x, right_y]
        ])

        polygon_path = Path(polygon_vertices)
        occluded_indices = []
        occluded_positions = []
        for other_idx, other in enumerate(obstacles):
            if other_idx == obs_idx or other[6] < 0.5:
                continue
            other_x0, other_y0, other_heading, other_v, other_length, other_width, _ = other
            other_behavior = behavior_lookup.get(other_idx)
            other_state = get_obstacle_state_at_time(other, t_elapsed, other_behavior)
            other_x = other_state['x']
            other_y = other_state['y']
            if not polygon_path.contains_point((other_x, other_y)):
                continue

            other_dx = other_x - ego_front_x
            other_dy = other_y - ego_front_y
            other_distance = np.hypot(other_dx, other_dy)

            # Skip vehicles that are closer to the ego than the blocking truck
            if other_distance <= distance_to_obs + 1e-6:
                continue

            # Additional check: ensure the other vehicle lies roughly beyond the obstacle
            blocker_vec = np.array([dx, dy])
            other_vec = np.array([other_dx, other_dy])
            if np.dot(other_vec, blocker_vec) <= 0.0:
                continue

            occluded_indices.append(other_idx)
            occluded_positions.append((other_x, other_y))
        
        occlusion_info = {
            'obstacle_id': obs_idx,
            'obstacle_pos': (obs_x, obs_y),
            'tangent_left': {'point': (left_x, left_y), 'direction': left_direction},
            'tangent_right': {'point': (right_x, right_y), 'direction': right_direction},
            'occluded_polygon': polygon_vertices,
            'is_truck': is_truck,
            'distance': distance_to_obs,
            'occluded_obstacles': occluded_indices,
            'occluded_positions': occluded_positions,
        }
        
        occlusion_regions.append(occlusion_info)
    
    return occlusion_regions


def main():
    print("Highway Overtaking Maneuver of Tractor-Trailers")
    print("=" * 50)
    
    config = SemiTrailerConfig()
    mpc = SemiTrailerMPC(config)
    
    initial_state = np.array([
        config.start_pos[0], config.start_pos[1], config.start_pos[2],
        config.start_pos[3], 0.0, config.start_pos[5]
    ])
    
    scenario_type = "dual_overtake"
    if scenario_type == "dual_overtake":
        obstacles, obstacle_behaviors = create_dual_overtake_scenario(config)
        scenario_label = "Dual overtaking cascade (ego + passenger car)"
    elif scenario_type == "occluded_lane_change":
        obstacles, obstacle_behaviors = create_occluded_lane_change_scenario(config)
        scenario_label = "Occluded lane-change stress test"
    else:
        obstacles = create_semi_trailer_scenario()
        obstacle_behaviors = []
        scenario_label = "Random traffic snapshot"
    
    print(f"Vehicle Configuration:")
    print(f"  - Total Length: {config.total_length:.1f}m")
    print(f"  - Tractor: {config.tractor_length:.1f}m, Trailer: {config.trailer_length:.1f}m")
    print(f"  - Max Speed: {config.v_max*3.6:.0f} km/h")
    print(f"  - Max Acceleration: {config.a_max:.1f} m/s²")
    print(f"  - Max Articulation: ±{np.rad2deg(config.phi_max):.0f}°")
    print()
    
    print(f"Scenario: {scenario_label}")
    print(f"  - Road Length: {config.road_length:.0f}m")
    print(f"  - Number of Obstacles: {len([obs for obs in obstacles if obs[6] > 0.5])}")
    if obstacle_behaviors:
        for behavior in obstacle_behaviors:
            obs_idx = behavior.get('obs_index', -1)
            start_time = behavior.get('start_time', 0.0)
            duration = behavior.get('duration', 0.0)
            target_y = behavior.get('target_y', 0.0)
            print(f"    * Obstacle #{obs_idx} lane change starts @ {start_time:.1f}s "
                  f"lasting {duration:.1f}s -> target y={target_y:.1f}m")
    print()
    
    n_steps = int(config.sim_time / config.dt)
    state_history = np.zeros((n_steps + 1, 6))
    state_history[0] = initial_state
    current_state = initial_state.copy()
    pred_history = []
    ttc_history = []  # NEW: Store TTC at each timestep

    print("Starting simulation...")
    print(f"Total steps: {n_steps}, Time step: {config.dt}s")
    print()

    successful_steps = n_steps
    min_ttc_overall = np.inf
    critical_timestep = -1
    
    for i in range(n_steps):
        sim_time = i * config.dt
        # Generate reference trajectory
        ref_traj = generate_semi_trailer_reference_path(
            current_state, config.goal_pos, config.N, config.dt, config, obstacles
        )
        obs_predictions = build_obstacle_prediction_matrix(
            obstacles, obstacle_behaviors, config, sim_time
        )
        
        # Solve MPC
        try:
            u_opt, x_pred = mpc.solve(
                current_state, ref_traj, obstacles, obstacle_predictions=obs_predictions
            )
            pred_history.append(x_pred)
            
            # Apply control
            control = u_opt[:, 0]
            next_state = mpc.dynamics(current_state, control).full().flatten()
            
            current_state = next_state
            state_history[i+1] = current_state
            
            # === NEW: Calculate TTC ===
            ttc_result = calculate_ttc(
                current_state, obstacles, config, config.dt, state_history[:i+2],
                obstacle_behaviors=obstacle_behaviors,
                current_time=(i + 1) * config.dt
            )
            ttc_history.append(ttc_result)
            
            # Track minimum TTC
            if ttc_result['min_ttc_overall'] < min_ttc_overall:
                min_ttc_overall = ttc_result['min_ttc_overall']
                critical_timestep = i + 1
            
            # Articulation angle
            phi = current_state[2] - current_state[5]
            
            # Progress report with TTC
            if i % 20 == 0 or i < 10:
                print(f"Step {i+1:3d}/{n_steps}: "
                      f"Pos=({current_state[0]:5.1f}, {current_state[1]:5.1f})m, "
                      f"Speed={current_state[3]*3.6:4.0f}km/h, "
                      f"={np.rad2deg(phi):5.1f}°, "
                      f"TTC={ttc_result['min_ttc_overall']:5.1f}s")
            
            # Check goal reached
            pos_error = np.linalg.norm(current_state[:2] - config.goal_pos[:2])
            if pos_error < config.goal_threshold:
                print(f" Goal reached! Position error: {pos_error:.2f}m")
                print(f"   Completion time: {(i+1)*config.dt:.1f}s")
                successful_steps = i + 1
                # break
            
            # === NEW: Safety warnings based on TTC ===
            if ttc_result['min_ttc_overall'] < 3.0:
                print(f"CRITICAL TTC: {ttc_result['min_ttc_overall']:.1f}s "
                      f"(Type: {ttc_result['collision_type']})")
            elif ttc_result['min_ttc_overall'] < 5.0:
                if i % 40 == 0:  # Don't spam warnings
                    print(f"Warning: Low TTC = {ttc_result['min_ttc_overall']:.1f}s")
            
            # Check articulation safety
            if abs(phi) > config.phi_max * 0.9:
                print(f"High articulation angle: {np.rad2deg(phi):.1f}")
            
        except Exception as e:
            print(f"MPC solve failed at step {i+1}: {e}")
            print("   Stopping simulation...")
            successful_steps = i
            break

    print(f"\n{'='*50}")
    print(f"Simulation completed!")
    print(f"Final position: ({current_state[0]:.1f}, {current_state[1]:.1f})m")
    print(f"Final speed: {current_state[3]*3.6:.0f} km/h")
    print(f"Total time: {successful_steps*config.dt:.1f}s")
    
    # Trim history
    state_history = state_history[:successful_steps+1]
    
    # === NEW: TTC Statistics ===
    if len(ttc_history) > 0:
        ttc_values = [t['min_ttc_overall'] for t in ttc_history]
        ttc_finite = [t for t in ttc_values if t < 20.0]
        
        print(f"\n{'='*50}")
        print(f"Safety Metrics (Time-to-Collision):")
        print(f"  - Minimum TTC: {min_ttc_overall:.2f}s (at t={critical_timestep*config.dt:.1f}s)")
        if len(ttc_finite) > 0:
            print(f"  - Mean TTC: {np.mean(ttc_finite):.2f}s")
            print(f"  - Median TTC: {np.median(ttc_finite):.2f}s")
            
            # Count violations
            critical_count = sum(1 for t in ttc_finite if t < 3.0)
            warning_count = sum(1 for t in ttc_finite if 3.0 <= t < 5.0)
            safe_count = sum(1 for t in ttc_finite if t >= 5.0)
            
            total_count = len(ttc_finite)
            print(f"  - TTC < 3s (Critical): {critical_count} steps ({critical_count/total_count*100:.1f}%)")
            print(f"  - TTC 3-5s (Warning): {warning_count} steps ({warning_count/total_count*100:.1f}%)")
            print(f"  - TTC > 5s (Safe): {safe_count} steps ({safe_count/total_count*100:.1f}%)")
        
        # Separate analysis for tractor vs trailer
        ttc_tractor = [t['min_ttc_tractor'] for t in ttc_history if t['min_ttc_tractor'] < 20.0]
        ttc_trailer = [t['min_ttc_trailer'] for t in ttc_history if t['min_ttc_trailer'] < 20.0]
        
        if len(ttc_tractor) > 0:
            print(f"  - Min TTC (Tractor): {min(ttc_tractor):.2f}s")
        if len(ttc_trailer) > 0:
            print(f"  - Min TTC (Trailer): {min(ttc_trailer):.2f}s")
    
    # Calculate other performance metrics
    max_articulation = np.max(np.abs(np.rad2deg(state_history[:, 2] - state_history[:, 5])))
    max_lateral_dev = np.max(np.abs(state_history[:, 1]))
    avg_speed = np.mean(state_history[:, 3]) * 3.6
    
    print(f"\nPerformance Metrics:")
    print(f"  - Max Articulation Angle: {max_articulation:.1f} (limit: ±{np.rad2deg(config.phi_max):.0f})")
    print(f"  - Max Lateral Deviation: {max_lateral_dev:.2f}m")
    print(f"  - Average Speed: {avg_speed:.0f} km/h")
    print(f"  - Distance Traveled: {current_state[0] - initial_state[0]:.1f}m")

    # Create visualizations
    print("\nGenerating animation...")
    animate_semi_trailer_trajectory(
        state_history, obstacles, mpc, config, successful_steps, obstacle_behaviors=obstacle_behaviors
    )
    
    print("Generating analysis plots...")
    plot_semi_trailer_results(state_history, pred_history, config, obstacles, ttc_history)
    
    print("\n“ Semi-trailer MPC simulation completed successfully!")


"""
def main():
    print("Highway Overtaking Maneuver of Tractor-Trailers")
    print("=" * 50)
    
    # Initialize configuration and MPC controller
    config = SemiTrailerConfig()
    mpc = SemiTrailerMPC(config)
    
    # Initial state: [x, y, Î¸_tractor, v, Î´, Î¸_trailer]
    initial_state = np.array([
        config.start_pos[0],  # x position
        config.start_pos[1],  # y position  
        config.start_pos[2],  # tractor heading angle
        config.start_pos[3],  # speed
        0.0,                  # steering angle (initial)
        config.start_pos[5]   # trailer heading angle (aligned with tractor)
    ])
    
    # Create semi-trailer scenario
    obstacles = create_semi_trailer_scenario()
    
    print(f"Vehicle Configuration:")
    print(f"  - Total Length: {config.total_length:.1f}m")
    print(f"  - Tractor: {config.tractor_length:.1f}m, Trailer: {config.trailer_length:.1f}m")
    print(f"  - Max Speed: {config.v_max*3.6:.0f} km/h")
    print(f"  - Max Acceleration: {config.a_max:.1f} m/s²")
    print(f"  - Max Articulation: ±{np.rad2deg(config.phi_max):.0f}°")
    print()
    
    print(f"Speed Configuration:")
    print(f"  - Ego Semi-Trailer: {initial_state[3]*3.6:.0f} km/h ({initial_state[3]:.1f} m/s)")
    
    for i, obs in enumerate(obstacles):
        if obs[6] > 0.5:  # Active obstacle
            if i == 0:
                print(f"  - Slow Vehicle:     {obs[3]*3.6:.0f} km/h ({obs[3]:.1f} m/s) - DIFFERENCE: {(initial_state[3]-obs[3])*3.6:.0f} km/h")
            else:
                print(f"  - Oncoming Vehicle: {obs[3]*3.6:.0f} km/h ({obs[3]:.1f} m/s)")
    print()
    
    print(f"Scenario:")
    print(f"  - Road Length: {config.road_length:.0f}m")
    print(f"  - Lane Width: {config.y_max - config.y_min:.1f}m")
    print(f"  - Number of Obstacles: {len([obs for obs in obstacles if obs[6] > 0.5])}")
    print()
    
    # Simulation initialization
    n_steps = int(config.sim_time / config.dt)
    state_history = np.zeros((n_steps + 1, 6))  # 6 states for semi-trailer
    state_history[0] = initial_state
    current_state = initial_state.copy()
    pred_history = []

    print("Starting simulation...")
    print(f"Total steps: {n_steps}, Time step: {config.dt}s")
    print()

    successful_steps = n_steps
    for i in range(n_steps):
        # Generate reference trajectory with obstacle awareness
        ref_traj = generate_semi_trailer_reference_path(
            current_state, config.goal_pos, config.N, config.dt, config, obstacles
        )
        
        # Solve MPC
        try:
            u_opt, x_pred = mpc.solve(current_state, ref_traj, obstacles)
            pred_history.append(x_pred)
            
            # Apply first control action
            control = u_opt[:, 0]
            next_state = mpc.dynamics(current_state, control).full().flatten()
            
            current_state = next_state
            state_history[i+1] = current_state
            
            # Calculate articulation angle
            phi = current_state[2] - current_state[5]
            
            # Progress report
            if i % 20 == 0 or i < 10:
                print(f"Step {i+1:3d}/{n_steps}: "
                      f"Pos=({current_state[0]:5.1f}, {current_state[1]:5.1f})m, "
                      f"Speed={current_state[3]*3.6:4.0f}km/h, "
                      f"Ï†={np.rad2deg(phi):5.1f}°")
            
            # Check goal reached
            pos_error = np.linalg.norm(current_state[:2] - config.goal_pos[:2])
            if pos_error < config.goal_threshold:
                print(f"\n Goal reached! Position error: {pos_error:.2f}m")
                print(f"   Completion time: {(i+1)*config.dt:.1f}s")
                successful_steps = i + 1
                break
                
            # Check articulation safety
            if abs(phi) > config.phi_max * 0.9:
                print(f"\nHigh articulation angle: {np.rad2deg(phi):.1f}°")
            
        except Exception as e:
            print(f"\n MPC solve failed at step {i+1}: {e}")
            print("   Stopping simulation...")
            successful_steps = i
            break

    print(f"\nSimulation completed!")
    print(f"Final position: ({current_state[0]:.1f}, {current_state[1]:.1f})m")
    print(f"Final speed: {current_state[3]*3.6:.0f} km/h")
    print(f"Total time: {successful_steps*config.dt:.1f}s")
    
    # Trim history to successful steps
    state_history = state_history[:successful_steps+1]
    
    # Calculate performance metrics
    max_articulation = np.max(np.abs(np.rad2deg(state_history[:, 2] - state_history[:, 5])))
    max_lateral_dev = np.max(np.abs(state_history[:, 1]))
    avg_speed = np.mean(state_history[:, 3]) * 3.6
    
    print(f"\nPerformance Metrics:")
    print(f"  - Max Articulation Angle: {max_articulation:.1f}° (limit: ±{np.rad2deg(config.phi_max):.0f}°)")
    print(f"  - Max Lateral Deviation: {max_lateral_dev:.2f}m")
    print(f"  - Average Speed: {avg_speed:.0f} km/h")
    print(f"  - Distance Traveled: {current_state[0] - initial_state[0]:.1f}m")

    # Create visualizations
    print("\nGenerating animation...")
    animate_semi_trailer_trajectory(state_history, obstacles, mpc, config, successful_steps)
    
    print("Generating analysis plots...")
    plot_semi_trailer_results(state_history, pred_history, config)
    
    print("\n Semi-trailer MPC simulation completed successfully!")
"""

if __name__ == "__main__":
    main()
