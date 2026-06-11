import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
import sys
import tty
import termios
import numpy as np
import math

class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')

        self.publisher = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )

        # Publisher to reset car position
        self.pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10
        )

        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )

        self.speed = 0.0
        self.steering = 0.0
        self.max_speed = 3.0
        self.max_steering = 0.4
        self.speed_step = 0.25
        self.steer_step = 0.05

        # Crash detection
        self.crash_distance = 0.2
        self.crashed = False
        self.reset_cooldown = 0

        # Starting position from sim.yaml
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_theta = 0.0

        self.timer = self.create_timer(0.05, self.publish_command)
        self.get_logger().info('Teleop Ready')
        self.get_logger().info('W/S=speed A/D=steer Space=stop Q=quit')
        self.get_logger().info('Car will auto-reset to start on crash')

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges), 10.0, ranges)
        ranges = np.where(np.isnan(ranges), 10.0, ranges)

        # Check if any beam is closer than crash distance
        min_distance = np.min(ranges)

        # Only detect crash if cooldown has expired
        if min_distance < self.crash_distance and self.reset_cooldown == 0:
            self.get_logger().warn(
                f'CRASH DETECTED - closest wall: {min_distance:.3f}m - RESETTING'
            )
            self.crashed = True

    def reset_to_start(self):
        # Stop the car first
        self.speed = 0.0
        self.steering = 0.0

        # Build the reset pose message
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        # Position
        msg.pose.pose.position.x = self.start_x
        msg.pose.pose.position.y = self.start_y
        msg.pose.pose.position.z = 0.0

        # Orientation - convert theta angle to quaternion
        # For a 2D rotation, only the z and w components matter
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = math.sin(self.start_theta / 2.0)
        msg.pose.pose.orientation.w = math.cos(self.start_theta / 2.0)

        # Covariance - set small values meaning we're confident in this position
        msg.pose.covariance[0] = 0.1
        msg.pose.covariance[7] = 0.1
        msg.pose.covariance[35] = 0.1

        self.pose_publisher.publish(msg)
        self.get_logger().info('Reset to starting position')

        # Set cooldown so we don't immediately detect another crash
        # 40 ticks at 20Hz = 2 seconds
        self.reset_cooldown = 40
        self.crashed = False

    def get_key(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return key

    def run(self):
        while rclpy.ok():
            key = self.get_key()

            if key == 'w':
                self.speed = min(self.speed + self.speed_step, self.max_speed)
            elif key == 's':
                self.speed = max(self.speed - self.speed_step, -self.max_speed)
            elif key == 'a':
                self.steering = min(self.steering + self.steer_step, self.max_steering)
            elif key == 'd':
                self.steering = max(self.steering - self.steer_step, -self.max_steering)
            elif key == ' ':
                self.speed = 0.0
                self.steering = 0.0
                self.get_logger().info('STOPPED')
            elif key == 'r':
                self.get_logger().info('Manual reset')
                self.reset_to_start()
            elif key == 'q':
                self.speed = 0.0
                self.steering = 0.0
                self.publish_command()
                self.get_logger().info('Quitting')
                break

            self.get_logger().info(
                f'Speed: {self.speed:.2f} | Steering: {self.steering:.3f}'
            )
            rclpy.spin_once(self, timeout_sec=0)

    def publish_command(self):
        # Handle crash reset
        if self.crashed:
            self.reset_to_start()

        # Count down cooldown
        if self.reset_cooldown > 0:
            self.reset_cooldown -= 1

        msg = AckermannDriveStamped()
        msg.drive.speed = self.speed
        msg.drive.steering_angle = self.steering
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleop()
    node.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
