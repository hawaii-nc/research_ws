"""
Automated corner-by-corner testing in Gazebo.
Fixed: flushes stale pre-teleport scans, enforces a MANDATORY minimum
settle time (not just an eager streak-based check) before every corner.
"""
import sys, json, time, math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

TEST_SECONDS = 8.0
WALL_CONTACT_THRESHOLD = 0.15
MANDATORY_SETTLE_SECONDS = 2.0   # ALWAYS wait this long, no early exit
MAX_EXTRA_SETTLE_WAIT = 5.0      # on top of mandatory, wait up to this much more if still not sane
GOOD_SCAN_STREAK_NEEDED = 5

class CornerTester(Node):
    def __init__(self, track):
        super().__init__('corner_tester')
        with open('/research_ws/corner_test_plan.json') as f:
            all_corners = json.load(f)
        self.corners = [c for c in all_corners if c['track'] == track]
        self.track = track
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.create_subscription(Odometry, '/ego_racecar/odom', self._odom_cb, 10)
        self.min_lidar = 30.0
        self.current_pos = (0.0, 0.0)
        self.results = []
        self.recording_lidar = False
        self.last_scan_sane = False
        self.good_streak = 0
        self.respawn_time = 0.0

    def _scan_cb(self, msg):
        ranges = np.array(msg.ranges)
        finite_positive = ranges[np.isfinite(ranges) & (ranges >= 0.0)]
        frac_bad = 1.0 - (len(finite_positive) / max(len(ranges), 1))
        self.last_scan_sane = frac_bad < 0.05
        if self.last_scan_sane:
            self.good_streak += 1
        else:
            self.good_streak = 0
        if self.recording_lidar and len(finite_positive) > 0:
            self.min_lidar = min(self.min_lidar, float(finite_positive.min()))

    def _odom_cb(self, msg):
        self.current_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def respawn(self, x, y, heading):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z = math.sin(heading/2)
        msg.pose.pose.orientation.w = math.cos(heading/2)
        self.pose_pub.publish(msg)
        self.respawn_time = time.time()

    def flush_stale_scans(self):
        """Drain any queued /scan messages that may predate the teleport."""
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.02)

    def wait_for_settle(self):
        """ALWAYS wait a mandatory minimum, THEN require a clean streak on top."""
        t0 = time.time()
        while time.time() - t0 < MANDATORY_SETTLE_SECONDS:
            rclpy.spin_once(self, timeout_sec=0.1)
        # Now require a genuine good streak, but cap total extra wait
        self.good_streak = 0
        t1 = time.time()
        while self.good_streak < GOOD_SCAN_STREAK_NEEDED and time.time() - t1 < MAX_EXTRA_SETTLE_WAIT:
            rclpy.spin_once(self, timeout_sec=0.1)
        settled = self.good_streak >= GOOD_SCAN_STREAK_NEEDED
        return settled, time.time() - t0

    def run_all(self):
        for i, c in enumerate(self.corners):
            print(f"[{i+1}/{len(self.corners)}] Testing {c['direction']} corner "
                  f"(kappa={c['peak_kappa']:.3f}) at ({c['spawn_x']:.2f},{c['spawn_y']:.2f})")
            self.min_lidar = 30.0
            self.recording_lidar = False
            self.respawn(c['spawn_x'], c['spawn_y'], c['spawn_heading'])
            self.flush_stale_scans()

            settled, settle_time = self.wait_for_settle()
            if not settled:
                print(f"    WARNING: never settled cleanly after {settle_time:.1f}s -- testing anyway")

            self.recording_lidar = True
            start_pos = self.current_pos
            t0 = time.time()
            while time.time() - t0 < TEST_SECONDS:
                rclpy.spin_once(self, timeout_sec=0.1)
            end_pos = self.current_pos
            dist_traveled = math.hypot(end_pos[0]-start_pos[0], end_pos[1]-start_pos[1])
            passed = self.min_lidar > WALL_CONTACT_THRESHOLD and dist_traveled > 0.5
            result = {**c, 'min_lidar': self.min_lidar, 'dist_traveled': dist_traveled,
                      'passed': passed, 'settle_time': settle_time}
            self.results.append(result)
            print(f"    -> {'PASS' if passed else 'FAIL'} (min_lidar={self.min_lidar:.3f}m, "
                  f"traveled={dist_traveled:.2f}m, settle={settle_time:.1f}s)")

        outpath = f'/research_ws/corner_test_results_{self.track}.json'
        with open(outpath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nSaved results to {outpath}")

def main():
    track = sys.argv[1]
    rclpy.init()
    tester = CornerTester(track)
    tester.run_all()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
