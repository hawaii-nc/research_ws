"""Aggregate all per-map corner test results into the two requested plots."""
import json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results = []
for f in glob.glob('/research_ws/corner_test_results_*.json'):
    with open(f) as fh:
        results.extend(json.load(fh))

print(f"Total corners tested: {len(results)}")
print(f"Overall pass rate: {sum(r['passed'] for r in results)/len(results)*100:.1f}%")

# --- Plot 1: scatter, curvature vs pass/fail, colored by direction ---
fig, ax = plt.subplots(figsize=(10, 6))
for direction, color in [('left', 'tab:blue'), ('right', 'tab:orange')]:
    subset = [r for r in results if r['direction'] == direction]
    x = [abs(r['peak_kappa']) for r in subset]
    y = [1 if r['passed'] else 0 for r in subset]
    ax.scatter(x, y, label=direction, color=color, s=80, alpha=0.7,
               edgecolors='black', linewidth=0.5)
ax.set_xlabel('Corner Curvature |kappa| (rad/m) -- higher = tighter turn')
ax.set_ylabel('Passed (1) / Failed (0)')
ax.set_yticks([0, 1])
ax.set_title('Corner Difficulty vs Success, by Direction')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/research_ws/corner_curvature_vs_success.png', dpi=130)
print("Saved corner_curvature_vs_success.png")

# --- Plot 2: map overlay heatmaps, one per track ---
tracks = sorted(set(r['track'] for r in results))
fig, axes = plt.subplots(1, len(tracks), figsize=(6*len(tracks), 6))
if len(tracks) == 1:
    axes = [axes]

for ax, track in zip(axes, tracks):
    cl = np.loadtxt(f'/research_ws/maps/{track}_centerline.csv', delimiter=',')
    ax.plot(cl[:,0], cl[:,1], '-', color='lightgray', linewidth=1, zorder=1)
    track_results = [r for r in results if r['track'] == track]
    for r in track_results:
        color = 'green' if r['passed'] else 'red'
        marker = '^' if r['direction']=='left' else 'v'
        ax.scatter(r['corner_x'], r['corner_y'], color=color, marker=marker,
                   s=150, edgecolors='black', linewidth=1, zorder=3)
    ax.set_title(f'{track}\n({sum(r["passed"] for r in track_results)}/{len(track_results)} passed)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

plt.suptitle('Corner Pass/Fail by Location (green=pass, red=fail, ^=left, v=right)', fontsize=13)
plt.tight_layout()
plt.savefig('/research_ws/corner_map_overlay.png', dpi=130)
print("Saved corner_map_overlay.png")
