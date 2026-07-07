#!/bin/bash
MAP=${1:-aut}
CHECKPOINT="/research_ws/src/f1tenth_research/checkpoints/phase1_lidar/final.pt"
SIM_YAML="/sim_ws/install/f1tenth_gym_ros/share/f1tenth_gym_ros/config/sim.yaml"
PP_PY="/research_ws/src/f1tenth_research/f1tenth_research/pure_pursuit.py"

echo "=== F1Tenth RMA Deployment: map=$MAP ==="

tmux kill-session -t sim 2>/dev/null
tmux kill-session -t rma 2>/dev/null
pkill -f rma_deployment 2>/dev/null
pkill -f gym_bridge 2>/dev/null
sleep 2

python3 -c "
import re, yaml, numpy as np, math

# Fix Pure Pursuit waypoints for this map
pp = open('$PP_PY').read()
pp = re.sub(r\"'/research_ws/maps/.*_centerline\.csv'\",
            \"'/research_ws/maps/${MAP}_centerline.csv'\", pp)
open('$PP_PY', 'w').write(pp)

# Compute spawn from centerline (same logic as real_env.py)
# Find nearest centerline point to [0.7, 0.0], compute heading from neighbors
cl = np.loadtxt(f'/research_ws/maps/${MAP}_centerline.csv', delimiter=',')
xy = cl[:, 0:2]
dists = np.sqrt((xy[:,0]-0.7)**2 + (xy[:,1]-0.0)**2)
idx = int(np.argmin(dists))
ip = max(0, idx-2)
in_ = min(len(cl)-1, idx+2)
dx = xy[in_,0] - xy[ip,0]
dy = xy[in_,1] - xy[ip,1]
stheta = float(np.arctan2(dy, dx))  # no correction needed, f110gym stheta=0 = +x

sim = open('$SIM_YAML').read()
sim = re.sub(r'    map_path:.*', \"    map_path: '/research_ws/maps/${MAP}'\", sim)
sim = re.sub(r'    sx:.*', '    sx: 0.7000', sim)
sim = re.sub(r'    sy:.*', '    sy: 0.0000', sim)
sim = re.sub(r'    stheta:.*', f'    stheta: {stheta:.4f}', sim)
open('$SIM_YAML', 'w').write(sim)
print(f'PP waypoints: ${MAP}_centerline.csv')
print(f'Spawn: sx=0.7000, sy=0.0000, stheta={stheta:.4f} ({math.degrees(stheta):.1f} deg)')
print(f'Nearest centerline idx: {idx}, dist: {float(dists[idx]):.3f}m')
"

echo "Building..."
cd /research_ws && colcon build --packages-select f1tenth_research > /dev/null 2>&1
source /research_ws/install/setup.bash

echo "Launching simulator..."
tmux new-session -d -s sim 'source /sim_ws/install/setup.bash && source /research_ws/install/setup.bash && ros2 launch f1tenth_gym_ros gym_bridge_launch.py'
sleep 8
echo "Simulator ready."

echo "Launching RMA node..."
tmux new-session -d -s rma "source /sim_ws/install/setup.bash && source /research_ws/install/setup.bash && ros2 run f1tenth_research rma_deployment --actor_critic $CHECKPOINT 2>&1 | tee /tmp/rma_node.log"
sleep 5

cat /tmp/rma_node.log | tail -4

echo "Launching behavior monitor..."
tmux new-session -d -s monitor "source /sim_ws/install/setup.bash && source /research_ws/install/setup.bash && ros2 run f1tenth_research behavior_monitor 2>&1 | tee /tmp/monitor.log"
sleep 2
echo "Monitor running -- tail /tmp/monitor.log for events"
echo ""
echo "View:    http://10.82.3.189:8080/vnc.html"
echo "Monitor: tmux attach -t rma"
echo "Logs:    tail -f /tmp/rma_node.log"
