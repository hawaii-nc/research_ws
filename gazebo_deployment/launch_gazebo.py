#!/usr/bin/env python3
"""
Auto-spawn launcher for F1Tenth Gazebo deployment.

Reads the centerline CSV for the target map, picks a random on-track
point, computes the correct heading from adjacent points, updates
sim.yaml, then launches the gym bridge.

Usage:
    python3 launch_gazebo.py --map aut
    python3 launch_gazebo.py --map esp
"""

import argparse
import numpy as np
import yaml
import math
import os
import subprocess
import random

MAPS_DIR = '/research_ws/maps'
SIM_YAML = '/sim_ws/install/f1tenth_gym_ros/share/f1tenth_gym_ros/config/sim.yaml'

def get_spawn_pose(map_name, random_start=True):
    """
    Pick a spawn pose on the track centerline.

    Centerline CSV is in f110_gym map-local coordinates (meters).
    Gazebo world coords = map_local + map_origin (from .yaml).
    Heading computed from adjacent centerline points.
    """
    # Load map origin
    map_yaml_path = os.path.join(MAPS_DIR, f'{map_name}.yaml')
    with open(map_yaml_path) as f:
        map_data = yaml.safe_load(f)
    origin_x, origin_y = map_data['origin'][0], map_data['origin'][1]

    # Load centerline
    cl_path = os.path.join(MAPS_DIR, f'{map_name}_centerline.csv')
    cl = np.loadtxt(cl_path, delimiter=',')
    n = len(cl)

    # Pick a random point (avoid first/last 5 to ensure valid neighbors)
    idx = random.randint(5, n - 6) if random_start else 5

    # Compute heading from adjacent points
    dx = cl[idx + 1, 0] - cl[idx - 1, 0]
    dy = cl[idx + 1, 1] - cl[idx - 1, 1]
    heading = math.atan2(dy, dx)

    # Convert to Gazebo world coordinates
    world_x = cl[idx, 0] + origin_x
    world_y = cl[idx, 1] + origin_y

    return world_x, world_y, heading, idx, n


def update_sim_yaml(map_name, sx, sy, stheta):
    """Update sim.yaml with new map and spawn pose."""
    with open(SIM_YAML) as f:
        content = f.read()

    # Update map path
    import re
    content = re.sub(
        r"map_path:.*",
        f"map_path: '/research_ws/maps/{map_name}'",
        content
    )
    # Update spawn
    content = re.sub(r"    sx:.*", f"    sx: {sx:.4f}", content)
    content = re.sub(r"    sy:.*", f"    sy: {sy:.4f}", content)
    content = re.sub(r"    stheta:.*", f"    stheta: {stheta:.4f}", content)

    with open(SIM_YAML, 'w') as f:
        f.write(content)

    print(f"sim.yaml updated: map={map_name}, "
          f"sx={sx:.4f}, sy={sy:.4f}, stheta={stheta:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='aut',
                        choices=['aut', 'esp', 'gbr', 'mco'],
                        help='Map to use')
    parser.add_argument('--fixed_start', action='store_true',
                        help='Use fixed start point instead of random')
    args = parser.parse_args()

    print(f"Setting up spawn for map: {args.map}")
    sx, sy, stheta, idx, total = get_spawn_pose(
        args.map, random_start=not args.fixed_start
    )
    print(f"Spawn point: idx={idx}/{total}, "
          f"world=({sx:.3f}, {sy:.3f}), heading={math.degrees(stheta):.1f} deg")

    update_sim_yaml(args.map, sx, sy, stheta)

    print("\nLaunching Gazebo sim...")
    os.execv('/bin/bash', ['/bin/bash', '-c',
        'source /sim_ws/install/setup.bash && '
        'source /research_ws/install/setup.bash && '
        'ros2 launch f1tenth_gym_ros gym_bridge_launch.py'
    ])


if __name__ == '__main__':
    main()
