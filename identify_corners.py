"""
Identify corner segments. Fixed: proper min(left,right) clearance,
meters-based heading lookahead using real arc length (not index count).
"""
import numpy as np
import json

MAPS = ['sepang_tight', 'silverstone_tight', 'budapest_tight']
CURVATURE_THRESHOLD = 0.15
MIN_SEGMENT_LEN = 3
LOOKAHEAD_SPAWN_M = 3.0
HEADING_LOOKAHEAD_M = 0.8   # meters ahead for heading estimate (not index count)
CAR_HALF_WIDTH = 0.279 / 2
MIN_CLEARANCE = 0.08

def arc_length(xy):
    """Cumulative arc length along a (possibly looped) point sequence."""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])

def heading_at(cl_xy, cl_s, idx, lookahead_m):
    """Robust heading: find point ~lookahead_m ahead along arc length, not index count."""
    target_s = (cl_s[idx] + lookahead_m) % cl_s[-1]
    ahead_idx = np.argmin(np.abs(cl_s - target_s))
    if ahead_idx == idx:
        ahead_idx = (idx + 1) % len(cl_xy)
    dx = cl_xy[ahead_idx, 0] - cl_xy[idx, 0]
    dy = cl_xy[ahead_idx, 1] - cl_xy[idx, 1]
    return np.arctan2(dy, dx)

def find_corners(track):
    rl = np.loadtxt(f'/research_ws/maps/racelines/{track}_raceline.csv', delimiter=',', skiprows=1)
    cl = np.loadtxt(f'/research_ws/maps/{track}_centerline.csv', delimiter=',')
    kappa = rl[:, 4]
    cl_xy = cl[:, :2]
    cl_s = arc_length(cl_xy)
    w_left, w_right = cl[:, 2], cl[:, 3]
    clearance_both_sides = np.minimum(w_left, w_right) - CAR_HALF_WIDTH

    is_turn = np.abs(kappa) > CURVATURE_THRESHOLD
    corners = []
    i = 0
    while i < len(is_turn):
        if is_turn[i]:
            j = i
            sign = np.sign(kappa[i])
            while j < len(is_turn) and is_turn[j] and np.sign(kappa[j]) == sign:
                j += 1
            if j - i >= MIN_SEGMENT_LEN:
                peak_idx = i + np.argmax(np.abs(kappa[i:j]))
                corner_x, corner_y = rl[peak_idx, 1], rl[peak_idx, 2]
                entry_x, entry_y = rl[i, 1], rl[i, 2]
                entry_cl_idx = np.argmin(np.linalg.norm(cl_xy - [entry_x, entry_y], axis=1))

                # Walk back along ARC LENGTH (not index count) to find safe spawn point
                target_spawn_s = (cl_s[entry_cl_idx] - LOOKAHEAD_SPAWN_M) % cl_s[-1]
                spawn_idx = np.argmin(np.abs(cl_s - target_spawn_s))

                # Verify clearance at chosen spawn; nudge back further if needed
                for _ in range(50):
                    if clearance_both_sides[spawn_idx] >= MIN_CLEARANCE:
                        break
                    spawn_idx = (spawn_idx - 5) % len(cl_xy)

                spawn_x, spawn_y = cl_xy[spawn_idx]
                heading = heading_at(cl_xy, cl_s, spawn_idx, HEADING_LOOKAHEAD_M)
                clearance_at_spawn = float(clearance_both_sides[spawn_idx])

                corners.append({
                    'track': track,
                    'direction': 'left' if sign > 0 else 'right',
                    'peak_kappa': float(kappa[peak_idx]),
                    'spawn_x': float(spawn_x),
                    'spawn_y': float(spawn_y),
                    'spawn_heading': float(heading),
                    'spawn_clearance_m': clearance_at_spawn,
                    'corner_x': float(corner_x),
                    'corner_y': float(corner_y),
                })
            i = j
        else:
            i += 1
    return corners

all_corners = []
for track in MAPS:
    corners = find_corners(track)
    all_corners.extend(corners)
    min_clear = min(c['spawn_clearance_m'] for c in corners) if corners else 0
    print(f"{track}: {len(corners)} corners, min spawn clearance = {min_clear*100:.1f}cm")

with open('/research_ws/corner_test_plan.json', 'w') as f:
    json.dump(all_corners, f, indent=2)
print(f"\nTotal: {len(all_corners)} corners.")
