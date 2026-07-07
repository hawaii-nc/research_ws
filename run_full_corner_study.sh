#!/bin/bash
# Fully automated corner-by-corner Gazebo study across all maps.
# Run this once and walk away -- no manual intervention needed.

set -e
cd /research_ws/src/f1tenth_research
source /opt/ros/foxy/setup.bash
source /sim_ws/install/local_setup.bash

CHECKPOINT="checkpoints/phase1_lidar_v29/final.pt"
MAPS=(sepang_tight silverstone_tight budapest_tight)

for TRACK in "${MAPS[@]}"; do
    echo "============================================"
    echo "  TESTING TRACK: $TRACK"
    echo "============================================"

    # NUCLEAR clean slate every time -- tmux kill-session alone does NOT
    # reliably kill child gzserver/map_server processes, causing stale
    # map data to leak into the next track's test (confirmed bug).
    pkill -9 -f "gzserver" 2>/dev/null || true
    pkill -9 -f "gzclient" 2>/dev/null || true
    pkill -9 -f "map_server" 2>/dev/null || true
    pkill -9 -f "gym_bridge" 2>/dev/null || true
    pkill -9 -f "rviz2" 2>/dev/null || true
    pkill -9 -f "lifecycle_manager" 2>/dev/null || true
    tmux kill-session -t corner_study 2>/dev/null || true
    sleep 5

    # Configure + launch Gazebo for this map
    ./run_deployment.sh "$TRACK" "$CHECKPOINT" > /tmp/deploy_config_${TRACK}.log 2>&1

    tmux new-session -d -s corner_study -n gazebo \
        "bash -c 'source /opt/ros/foxy/setup.bash && source /sim_ws/install/local_setup.bash && ros2 launch f1tenth_gym_ros gym_bridge_launch.py 2>&1 | tee /tmp/gazebo_${TRACK}.log'"

    # Wait for Gazebo to actually be ready (poll for /scan topic)
    echo "Waiting for Gazebo to load..."
    for i in $(seq 1 30); do
        if ros2 topic list 2>/dev/null | grep -q "^/scan$"; then
            break
        fi
        sleep 2
    done
    sleep 5  # extra settle time

    # VERIFY the correct map is actually loaded before proceeding
    ACTUAL_MAP=$(ros2 param get /map_server yaml_filename 2>/dev/null | grep -oP "(?<=String value is: ).*")
    echo "Verifying map: expected ${TRACK}.yaml, map_server reports: ${ACTUAL_MAP}"
    if [[ "$ACTUAL_MAP" != *"${TRACK}"* ]]; then
        echo "ERROR: Wrong map loaded! Expected ${TRACK}, got ${ACTUAL_MAP}. Skipping this track."
        continue
    fi

    # Launch policy node (no adaptation -- testing base v29 policy)
    tmux new-window -t corner_study -n policy \
        "bash -c 'export RMA_TRACK=${TRACK} && source /opt/ros/foxy/setup.bash && source /sim_ws/install/local_setup.bash && python3 -m f1tenth_research.gazebo_deployment.rma_deployment_node --actor_critic ${CHECKPOINT} 2>&1 | tee /tmp/policy_${TRACK}.log'"

    # Wait for policy node ready
    echo "Waiting for policy node..."
    for i in $(seq 1 20); do
        if grep -q "ready at" /tmp/policy_${TRACK}.log 2>/dev/null; then
            break
        fi
        sleep 2
    done
    sleep 3

    # Run the corner test suite for this track (blocks until done)
    echo "Running corner tests for $TRACK..."
    python3 run_corner_gazebo_test.py "$TRACK"

    echo "Done with $TRACK"
    tmux kill-session -t corner_study 2>/dev/null || true
    sleep 3
done

echo "============================================"
echo "  ALL MAPS COMPLETE -- generating final report"
echo "============================================"
python3 analyze_corner_results.py
