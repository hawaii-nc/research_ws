"""
F1Tenth Behavior Monitor
========================
Runs alongside rma_deployment_node. Detects driving events in real time,
auto-resets on crash/out-of-bounds, and prints a run report after each episode.

Controls (type in terminal running this node):
  r - manually reset and start new run
  a - run 10 automatic runs, print combined report at end

Run report format:
  Segments driven (straight/slight/medium/hard turn) with checkmarks
  Speed profile at each turn (pre/in/post)
  Crash type (straight or turn), crash location
  Progress % of centerline completed
  Event log (oscillations, wall contacts, out-of-bounds)
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
import math
import time
import threading
import sys
import select


# ── Tunable thresholds ──────────────────────────────────────────────────────
WALL_CONTACT_THRESHOLD  = 0.3    # meters (real distance)
OUT_OF_BOUNDS_THRESHOLD = 2.5    # meters from nearest centerline point
OUT_OF_BOUNDS_DURATION  = 2.0    # seconds continuously OOB before reset
STUCK_SPEED_CMD         = 1.0    # m/s command threshold to detect stuck
STUCK_VELOCITY          = 0.1    # m/s actual velocity threshold
STUCK_DURATION          = 2.5    # seconds stuck before reset
OSC_FLIPS               = 2      # sign changes in steer buffer = oscillation
OSC_WINDOW              = 25     # samples (~0.5s at 50Hz)
SLIGHT_TURN_RAD         = 0.08   # rad steering threshold
MEDIUM_TURN_RAD         = 0.20
HARD_TURN_RAD           = 0.35
SEGMENT_MIN_STEPS       = 10     # minimum steps to record a segment
CENTERLINE_PATH         = '/research_ws/maps/aut_centerline.csv'
SPAWN_X, SPAWN_Y        = 0.7, 0.0
SPAWN_QZ, SPAWN_QW      = 0.000177, 0.999999
# ────────────────────────────────────────────────────────────────────────────


def classify_turn(steer_rad):
    """Classify steering magnitude into turn type string."""
    s = abs(steer_rad)
    if s < SLIGHT_TURN_RAD:
        return 'straight'
    elif s < MEDIUM_TURN_RAD:
        return 'slight'
    elif s < HARD_TURN_RAD:
        return 'medium'
    else:
        return 'hard'


class Segment:
    """One contiguous driving segment (straight or turn)."""
    def __init__(self, kind, start_idx):
        self.kind = kind          # 'straight','slight','medium','hard'
        self.start_idx = start_idx
        self.end_idx = start_idx
        self.steps = 0
        self.speeds = []
        self.steers = []
        self.success = True       # False if segment ended in crash

    def add(self, speed, steer, cl_idx):
        self.steps += 1
        self.speeds.append(speed)
        self.steers.append(abs(steer))
        self.end_idx = cl_idx

    @property
    def avg_speed(self):
        return float(np.mean(self.speeds)) if self.speeds else 0.0

    @property
    def pre_speed(self):
        return float(self.speeds[0]) if self.speeds else 0.0

    @property
    def mid_speed(self):
        mid = len(self.speeds) // 2
        return float(self.speeds[mid]) if self.speeds else 0.0

    @property
    def post_speed(self):
        return float(self.speeds[-1]) if self.speeds else 0.0

    def label(self):
        arrow = '→'
        direction = ''
        if self.kind != 'straight' and self.steers:
            # detect direction from raw steer (would need signed steer)
            direction = ''
        return f"{self.kind.upper()}"


class RunData:
    """Stores all data for one episode run."""
    def __init__(self, run_number):
        self.run_number = run_number
        self.start_time = time.time()
        self.end_time = None
        self.segments = []
        self.events = []           # (time, event_type, detail)
        self.n_oscillations = 0
        self.n_wall_contacts = 0
        self.n_oob = 0
        self.max_cl_idx = 0
        self.total_cl_points = 475
        self.crash_type = None     # 'straight', 'turn', 'oob', None
        self.crash_reason = None
        self.reset_reason = None

    def log_event(self, event_type, detail):
        t = time.time() - self.start_time
        self.events.append((t, event_type, detail))
        if event_type == 'OSCILLATION':
            self.n_oscillations += 1
        elif event_type == 'WALL_CONTACT':
            self.n_wall_contacts += 1
        elif event_type == 'OUT_OF_BOUNDS':
            self.n_oob += 1

    def progress_pct(self):
        return 100.0 * self.max_cl_idx / max(self.total_cl_points, 1)

    def duration(self):
        end = self.end_time or time.time()
        return end - self.start_time

    def print_report(self):
        dur = self.duration()
        pct = self.progress_pct()
        print()
        print("=" * 55)
        print(f"  RUN {self.run_number} REPORT  |  {dur:.1f}s  |  Progress: {pct:.1f}%")
        print("=" * 55)

        # Segment checklist
        print("\nSEGMENT CHECKLIST:")
        for i, seg in enumerate(self.segments):
            if seg.steps < SEGMENT_MIN_STEPS:
                continue
            mark = "✓" if seg.success else "✗"
            speed_str = ""
            if seg.kind != 'straight':
                speed_str = (f"  [pre={seg.pre_speed:.1f} "
                           f"in={seg.mid_speed:.1f} "
                           f"post={seg.post_speed:.1f} m/s]")
            else:
                speed_str = f"  [{seg.avg_speed:.1f} m/s avg]"
            cl_range = f"cl {seg.start_idx}→{seg.end_idx}"
            print(f"  {mark} {seg.label():<10} {cl_range:<16} {speed_str}")

        # Crash info
        print(f"\nRESET REASON: {self.reset_reason or 'unknown'}")
        if self.crash_type:
            print(f"CRASH TYPE:   {self.crash_type}")

        # Event summary
        print(f"\nEVENT SUMMARY:")
        print(f"  Wall contacts:  {self.n_wall_contacts}")
        print(f"  Oscillations:   {self.n_oscillations}")
        print(f"  Out of bounds:  {self.n_oob}")
        if self.events:
            print(f"\nEVENT LOG (last 5):")
            for t, etype, detail in self.events[-5:]:
                print(f"  [{t:5.1f}s] {etype}: {detail}")
        print("=" * 55)
        print()


def print_combined_report(runs):
    print()
    print("=" * 60)
    print(f"  COMBINED REPORT — {len(runs)} RUNS")
    print("=" * 60)
    print(f"{'Run':<5} {'Progress%':<12} {'Duration':<10} "
          f"{'Walls':<7} {'Osc':<6} {'Crash Type'}")
    print("-" * 60)
    for r in runs:
        print(f"{r.run_number:<5} {r.progress_pct():<12.1f} "
              f"{r.duration():<10.1f} "
              f"{r.n_wall_contacts:<7} {r.n_oscillations:<6} "
              f"{r.crash_type or 'none'}")
    print("-" * 60)
    avg_prog = np.mean([r.progress_pct() for r in runs])
    avg_dur  = np.mean([r.duration() for r in runs])
    print(f"{'AVG':<5} {avg_prog:<12.1f} {avg_dur:<10.1f}")
    print("=" * 60)
    print()


class BehaviorMonitor(Node):

    def __init__(self):
        super().__init__('behavior_monitor')

        # Load centerline
        try:
            cl = np.loadtxt(CENTERLINE_PATH, delimiter=',')
            self.centerline = cl[:, 0:2]
            self.get_logger().info(
                f"Centerline loaded: {len(self.centerline)} points"
            )
        except Exception as e:
            self.centerline = None
            self.get_logger().warn(f"No centerline: {e}")

        # Live state
        self.current_velocity = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_speed_cmd = 0.0
        self.current_steer = 0.0
        self.min_lidar_m = 10.0    # real meters
        self.steer_buf = []

        # Timing
        self._stuck_since = None
        self._oob_since = None
        self._last_wall_t = -99
        self._last_osc_t  = -99

        # Run tracking
        self.run_number = 0
        self.current_run = None
        self.all_runs = []
        self.auto_runs_remaining = 0
        self._resetting = False

        # Segment tracking
        self._prev_seg_kind = None
        self._current_seg = None

        # Subscribers
        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(AckermannDriveStamped, '/drive', self.drive_cb, 10)

        # Publishers
        self.reset_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )

        # Timers
        self.create_timer(0.1, self.monitor_loop)   # 10Hz monitor

        # Keyboard listener thread
        self._kb_thread = threading.Thread(
            target=self._keyboard_loop, daemon=True
        )
        self._kb_thread.start()

        print("\nBehavior Monitor ready.")
        print("  r = start/reset run")
        print("  a = run 10 automatic runs")
        print()

    # ── ROS callbacks ────────────────────────────────────────────────────────

    def odom_cb(self, msg):
        self.current_velocity = float(msg.twist.twist.linear.x)
        self.current_x = float(msg.pose.pose.position.x)
        self.current_y = float(msg.pose.pose.position.y)

    def scan_cb(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        self.min_lidar_m = float(np.min(ranges))

    def drive_cb(self, msg):
        self.current_speed_cmd = float(msg.drive.speed)
        self.current_steer = float(msg.drive.steering_angle)
        self.steer_buf.append(self.current_steer)
        if len(self.steer_buf) > OSC_WINDOW:
            self.steer_buf.pop(0)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _nearest_cl_idx(self):
        if self.centerline is None:
            return 0, 999.0
        dx = self.centerline[:, 0] - self.current_x
        dy = self.centerline[:, 1] - self.current_y
        dists = np.sqrt(dx**2 + dy**2)
        idx = int(np.argmin(dists))
        return idx, float(dists[idx])

    def _detect_oscillation(self):
        if len(self.steer_buf) < OSC_WINDOW:
            return False
        flips = sum(
            1 for i in range(1, len(self.steer_buf))
            if (np.sign(self.steer_buf[i]) != np.sign(self.steer_buf[i-1])
                and abs(self.steer_buf[i]) > 0.05)
        )
        return flips >= OSC_FLIPS

    def _zero_speed(self):
        """Publish zero speed to stop the car."""
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        self.drive_pub.publish(msg)

    def _do_reset(self, reason):
        """Stop car, reset position, start new run."""
        if self._resetting:
            return
        self._resetting = True

        # Stop car
        for _ in range(5):
            self._zero_speed()

        # Finish current run
        if self.current_run is not None:
            if self._current_seg is not None:
                self._current_seg.success = False
                self.current_run.segments.append(self._current_seg)
                self._current_seg = None
            self.current_run.end_time = time.time()
            self.current_run.reset_reason = reason

            # Determine crash type
            seg_kind = (self.current_run.segments[-1].kind
                        if self.current_run.segments else 'unknown')
            if 'oob' in reason.lower() or 'out' in reason.lower():
                self.current_run.crash_type = 'out-of-bounds'
            elif seg_kind == 'straight':
                self.current_run.crash_type = 'straight'
            else:
                self.current_run.crash_type = f'turn ({seg_kind})'

            self.current_run.print_report()
            self.all_runs.append(self.current_run)

        # Publish reset pose
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = SPAWN_X
        msg.pose.pose.position.y = SPAWN_Y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = SPAWN_QZ
        msg.pose.pose.orientation.w = SPAWN_QW
        self.reset_pub.publish(msg)

        time.sleep(0.5)  # let gym bridge process reset

        # Start new run if auto mode
        if self.auto_runs_remaining > 0:
            self.auto_runs_remaining -= 1
            if self.auto_runs_remaining == 0:
                print_combined_report(self.all_runs[-10:])
                print("Auto run complete.")
            self._start_run()
        else:
            print("Car reset. Press 'r' to start next run, 'a' for 10 auto runs.")

        self._resetting = False

    def _start_run(self):
        self.run_number += 1
        self.current_run = RunData(self.run_number)
        if self.centerline is not None:
            self.current_run.total_cl_points = len(self.centerline)
        self._current_seg = None
        self._prev_seg_kind = None
        self._stuck_since = None
        self._oob_since = None
        print(f"\n▶ Run {self.run_number} started")

    # ── Main monitor loop ────────────────────────────────────────────────────

    def monitor_loop(self):
        if self.current_run is None or self._resetting:
            return

        now = time.time()
        run = self.current_run
        cl_idx, cl_dist = self._nearest_cl_idx()

        # Update max progress
        if cl_idx > run.max_cl_idx:
            run.max_cl_idx = cl_idx

        # ── Segment tracking ────────────────────────────────────────────────
        seg_kind = classify_turn(self.current_steer)
        if self._current_seg is None:
            self._current_seg = Segment(seg_kind, cl_idx)
        elif seg_kind != self._current_seg.kind:
            # Segment type changed — save old, start new
            if self._current_seg.steps >= SEGMENT_MIN_STEPS:
                run.segments.append(self._current_seg)
            self._current_seg = Segment(seg_kind, cl_idx)
        self._current_seg.add(self.current_speed_cmd, self.current_steer, cl_idx)

        # ── Event detection ─────────────────────────────────────────────────

        # OSCILLATION
        if self._detect_oscillation() and now - self._last_osc_t > 2.0:
            run.log_event('OSCILLATION',
                f"steer flips detected, speed={self.current_speed_cmd:.1f}m/s")
            print(f"  [t={now-run.start_time:.1f}s] ⚠ OSCILLATION "
                  f"(speed={self.current_speed_cmd:.1f})")
            self._last_osc_t = now

        # WALL CONTACT
        if self.min_lidar_m < WALL_CONTACT_THRESHOLD and now - self._last_wall_t > 1.0:
            run.log_event('WALL_CONTACT',
                f"min_lidar={self.min_lidar_m:.2f}m, "
                f"seg={seg_kind}, speed={self.current_speed_cmd:.1f}m/s")
            print(f"  [t={now-run.start_time:.1f}s] ⚠ WALL_CONTACT "
                  f"({self.min_lidar_m:.2f}m, {seg_kind}, "
                  f"{self.current_speed_cmd:.1f}m/s)")
            self._last_wall_t = now

        # OUT OF BOUNDS (far from centerline + surrounded by walls)
        oob = (cl_dist > OUT_OF_BOUNDS_THRESHOLD
               and self.min_lidar_m < 1.0)
        if oob:
            if self._oob_since is None:
                self._oob_since = now
            elif now - self._oob_since > OUT_OF_BOUNDS_DURATION:
                run.log_event('OUT_OF_BOUNDS',
                    f"dist_from_cl={cl_dist:.2f}m for >{OUT_OF_BOUNDS_DURATION}s")
                print(f"  [t={now-run.start_time:.1f}s] ✗ OUT OF BOUNDS "
                      f"— resetting")
                threading.Thread(
                    target=self._do_reset,
                    args=("out-of-bounds",),
                    daemon=True
                ).start()
                return
        else:
            self._oob_since = None

        # STUCK
        if (self.current_speed_cmd > STUCK_SPEED_CMD
                and abs(self.current_velocity) < STUCK_VELOCITY):
            if self._stuck_since is None:
                self._stuck_since = now
            elif now - self._stuck_since > STUCK_DURATION:
                run.log_event('STUCK',
                    f"cmd={self.current_speed_cmd:.1f}m/s, "
                    f"actual={self.current_velocity:.3f}m/s")
                print(f"  [t={now-run.start_time:.1f}s] ✗ STUCK — resetting")
                threading.Thread(
                    target=self._do_reset,
                    args=("stuck/wall-contact",),
                    daemon=True
                ).start()
                return
        else:
            self._stuck_since = None

    # ── Keyboard input ───────────────────────────────────────────────────────

    def _keyboard_loop(self):
        """Non-blocking keyboard listener."""
        print("Keyboard listener active (r=run, a=10 auto runs)")
        while True:
            try:
                if select.select([sys.stdin], [], [], 0.2)[0]:
                    key = sys.stdin.readline().strip().lower()
                    if key == 'r':
                        if self.current_run is not None:
                            threading.Thread(
                                target=self._do_reset,
                                args=("manual reset",),
                                daemon=True
                            ).start()
                        else:
                            self._start_run()
                    elif key == 'a':
                        self.auto_runs_remaining = 10
                        self.all_runs = []
                        if self.current_run is not None:
                            threading.Thread(
                                target=self._do_reset,
                                args=("auto run start",),
                                daemon=True
                            ).start()
                        else:
                            self._start_run()
                        print("Auto mode: 10 runs queued")
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.all_runs:
            print_combined_report(node.all_runs)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
