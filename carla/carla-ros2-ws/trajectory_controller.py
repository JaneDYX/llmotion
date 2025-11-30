#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, Odometry
from carla_msgs.msg import CarlaEgoVehicleControl


class TrajectoryControllerNode(Node):
    def __init__(self):
        super().__init__('trajectory_controller_node')

        self.traj_sub = self.create_subscription(
            Path,
            '/planning/trajectory',
            self.traj_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/carla/hero/odometry',
            self.odom_callback,
            10
        )
        self.ctrl_pub = self.create_publisher(
            CarlaEgoVehicleControl,
            '/carla/hero/vehicle_control_cmd',
            10
        )

        self.timer = self.create_timer(0.05, self.timer_callback)

        self.latest_traj = None
        self.latest_odom = None

        self.wheelbase = 2.8
        self.lookahead_distance = 5.0
        self.target_speed = 8.0
        self.max_steer_rad = 0.7

    def traj_callback(self, msg: Path):
        self.latest_traj = msg

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg

    def timer_callback(self):
        if self.latest_traj is None or self.latest_odom is None:
            return
        if len(self.latest_traj.poses) == 0:
            return

        pose = self.latest_odom.pose.pose
        x = pose.position.x
        y = pose.position.y

        qw = pose.orientation.w
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        min_dist = 1e9
        closest_idx = -1
        poses = self.latest_traj.poses

        for i, p in enumerate(poses):
            dx = p.pose.position.x - x
            dy = p.pose.position.y - y
            d = math.hypot(dx, dy)
            if d < min_dist:
                min_dist = d
                closest_idx = i

        if closest_idx < 0:
            return

        accum_dist = 0.0
        lookahead_idx = closest_idx
        for i in range(closest_idx + 1, len(poses)):
            dx = poses[i].pose.position.x - poses[i - 1].pose.position.x
            dy = poses[i].pose.position.y - poses[i - 1].pose.position.y
            accum_dist += math.hypot(dx, dy)
            lookahead_idx = i
            if accum_dist >= self.lookahead_distance:
                break

        target_pose = poses[lookahead_idx].pose

        dx = target_pose.position.x - x
        dy = target_pose.position.y - y

        x_local = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        y_local = math.sin(-yaw) * dx + math.cos(-yaw) * dy

        ld = math.hypot(x_local, y_local)
        if ld < 1e-3:
            return

        steering_angle = math.atan2(2.0 * self.wheelbase * y_local, ld * ld)

        steer_norm = steering_angle / self.max_steer_rad
        steer_norm = max(min(steer_norm, 1.0), -1.0)

        vx = self.latest_odom.twist.twist.linear.x
        speed_error = self.target_speed - vx

        cmd = CarlaEgoVehicleControl()
        cmd.steer = float(steer_norm)

        if speed_error > 0.0:
            throttle_cmd = max(min(speed_error * 0.2, 1.0), 0.0)
            cmd.throttle = float(throttle_cmd)
            cmd.brake = 0.0
        else:
            brake_cmd = max(min(-speed_error * 0.2, 1.0), 0.0)
            cmd.throttle = 0.0
            cmd.brake = float(brake_cmd)

        cmd.hand_brake = False
        cmd.reverse = False
        cmd.manual_gear_shift = False

        self.ctrl_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
