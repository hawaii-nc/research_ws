import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import numpy as np
import csv
import math
import os

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit')

        # Publisher
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )

        # Subscriber - car position
        self.odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom',
            self.odom_callback, 10
        )

        # Load waypoints from centerline CSV
        self.waypoints = self.load_waypoints(
            '/research_ws/aut_centerline.csv'
        )
        self.get_logger().info(
            f'Loaded {len(self.waypoints)} waypoints'
        )

        # Pure Pursuit parameters
        self.lookahead_distance = 1.5  # meters ahead to aim for
        self.speed = 2.0               # constant speed m/s
        self.wheelbase = 0.33          # F1Tenth wheelbase in meters

        # State
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.got_odom = False

        # Drive at 20 Hz
        self.timer = self.create_timer(0.05, self.drive)

        self.get_logger().info('Pure Pursuit Ready - driving automatically')

    def load_waypoints(self, path):
        waypoints = []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                x = float(row[0])
                y = float(row[1])
                waypoints.append([x, y])
        return np.array(waypoints)

    def odom_callback(self, msg):
        # Get car position
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # Convert quaternion to yaw angle
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        self.got_odom = True

    def find_lookahead_point(self):
        # Find the waypoint that is closest to lookahead distance ahead

        # Calculate distance from car to every waypoint
        dx = self.waypoints[:, 0] - self.current_x
        dy = self.waypoints[:, 1] - self.current_y
        distances = np.sqrt(dx**2 + dy**2)

        # Find closest waypoint index
        closest_idx = np.argmin(distances)

        # Search forward from closest waypoint
        # to find one that is lookahead_distance away
        n = len(self.waypoints)
        for i in range(n):
            idx = (closest_idx + i) % n  # wrap around track
            dist = distances[idx]
            if dist >= self.lookahead_distance:
                return self.waypoints[idx]

        # Fallback - return closest waypoint
        return self.waypoints[closest_idx]

    def drive(self):
        if not self.got_odom:
            return

        # Find the target point to aim for
        target = self.find_lookahead_point()

        # Calculate angle from car to target point
        # in the car's local coordinate frame
        dx = target[0] - self.current_x
        dy = target[1] - self.current_y

        # Angle to target in world frame
        angle_to_target = math.atan2(dy, dx)

        # Angle to target relative to car's heading
        alpha = angle_to_target - self.current_yaw

        # Normalize angle to [-pi, pi]
        while alpha > math.pi:
            alpha -= 2 * math.pi
        while alpha < -math.pi:
            alpha += 2 * math.pi

        # Pure Pursuit steering formula
        steering = math.atan2(
            2.0 * self.wheelbase * math.sin(alpha),
            self.lookahead_distance
        )

        # Clamp steering to physical limits
        steering = np.clip(steering, -0.4, 0.4)

        # Slow down on sharp turns
        # The sharper the turn, the slower we go
        speed = self.speed * (1.0 - 0.5 * abs(steering) / 0.4)
        speed = max(speed, 0.5)  # never slower than 0.5 m/s

        # Publish drive command
        msg = AckermannDriveStamped()
        msg.drive.speed = speed
        msg.drive.steering_angle = steering
        self.drive_pub.publish(msg)

        self.get_logger().info(
            f'Target: ({target[0]:.2f}, {target[1]:.2f}) | '
            f'Speed: {speed:.2f} | Steering: {steering:.3f}',
            throttle_duration_sec=0.5  # only print every 0.5s
        )


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
