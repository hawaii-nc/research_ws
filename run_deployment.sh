#!/bin/bash
# Usage: ./run_deployment.sh <track> <checkpoint_path>
# Example: ./run_deployment.sh esp checkpoints/phase1_lidar_v17/final.pt

set -e

TRACK="${1:-aut}"
CHECKPOINT="${2:-checkpoints/phase1_lidar_v17/final.pt}"

# Validate the track has both a map and a raceline
MAP_PATH="/research_ws/maps/${TRACK}"
RACELINE_PATH="/research_ws/maps/racelines/${TRACK}_raceline.csv"
SIM_YAML="/sim_ws/install/f1tenth_gym_ros/share/f1tenth_gym_ros/config/sim.yaml"

if [ ! -f "${MAP_PATH}.yaml" ]; then
    echo "ERROR: No map at ${MAP_PATH}.yaml"
    echo "Available maps:"
    ls /research_ws/maps/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/.yaml//'
    exit 1
fi
if [ ! -f "${RACELINE_PATH}" ]; then
    echo "ERROR: No raceline at ${RACELINE_PATH}"
    exit 1
fi
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: No checkpoint at ${CHECKPOINT}"
    exit 1
fi

# Get spawn point + heading from the raceline (first row)
# psi column in raceline is normal-to-path, not tangent — compute heading from points
SPAWN=$(python3 -c "
import numpy as np
rl = np.loadtxt('${RACELINE_PATH}', delimiter=',', skiprows=1)
x0, y0 = rl[0, 1], rl[0, 2]
# Look ahead a few points for a stable heading estimate
look = min(5, len(rl) - 1)
dx = rl[look, 1] - x0
dy = rl[look, 2] - y0
heading = float(np.arctan2(dy, dx))
print(f'{x0:.4f} {y0:.4f} {heading:.4f}')
")
SX=$(echo $SPAWN | awk '{print $1}')
SY=$(echo $SPAWN | awk '{print $2}')
STHETA=$(echo $SPAWN | awk '{print $3}')

echo "=== Configuring deployment for track: ${TRACK} ==="
echo "  Map: ${MAP_PATH}"
echo "  Raceline: ${RACELINE_PATH}"
echo "  Spawn: x=${SX} y=${SY} theta=${STHETA}"
echo "  Checkpoint: ${CHECKPOINT}"

# Update sim.yaml in place
python3 << PYEOF
import re
path = "${SIM_YAML}"
with open(path) as f:
    text = f.read()
text = re.sub(r"map_path: '.*'", f"map_path: '${MAP_PATH}'", text)
text = re.sub(r"sx: [-0-9.]+", f"sx: ${SX}", text)
text = re.sub(r"sy: [-0-9.]+", f"sy: ${SY}", text)
text = re.sub(r"stheta: [-0-9.]+", f"stheta: ${STHETA}", text)
with open(path, 'w') as f:
    f.write(text)
print(f"Updated {path}")
PYEOF

# Export track for deployment node
export RMA_TRACK=${TRACK}

echo ""
echo "=== Setup complete ==="
echo "Now in two separate terminals:"
echo "  Terminal 1: ros2 launch f1tenth_gym_ros gym_bridge_launch.py"
echo "  Terminal 2: export RMA_TRACK=${TRACK} && python3 -m f1tenth_research.gazebo_deployment.rma_deployment_node --actor_critic ${CHECKPOINT}"
echo ""
echo "Or use tmux: ./run_deployment.sh ${TRACK} ${CHECKPOINT} run"

if [ "${3:-}" = "run" ]; then
    echo ""
    echo "=== Launching in tmux ==="
    tmux kill-session -t gazebo_deploy 2>/dev/null || true
    tmux new-session -d -s gazebo_deploy "bash -c 'source /opt/ros/foxy/setup.bash && source /sim_ws/install/local_setup.bash && ros2 launch f1tenth_gym_ros gym_bridge_launch.py 2>&1 | tee /tmp/gazebo.log'"
    sleep 5
    tmux new-window -t gazebo_deploy "bash -c 'export RMA_TRACK=${TRACK} && source /opt/ros/foxy/setup.bash && source /sim_ws/install/local_setup.bash && python3 -m f1tenth_research.gazebo_deployment.rma_deployment_node --actor_critic ${CHECKPOINT} 2>&1 | tee /tmp/policy.log'"
    echo "tmux session 'gazebo_deploy' started"
    echo "  Attach with: tmux attach -t gazebo_deploy"
    echo "  View Gazebo at: http://10.82.3.189:8080/vnc.html"
fi
