import math
from typing import Optional

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from carla_msgs.msg import CarlaEgoVehicleControl


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """Convert quaternion to yaw (Z-axis rotation)."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class PurePursuitNode(Node):
    """
    Pure pursuit controller for CARLA hero vehicle.

    - Subscribes to /carla/hero/odometry.
    - Subscribes to /planning/trajectory (local reference path).
    - Publishes /carla/hero/vehicle_control_cmd (CarlaEgoVehicleControl).
    """

    def __init__(self) -> None:
        super().__init__('pure_pursuit_node')

        self.odom_sub = self.create_subscription(
            Odometry,
            '/carla/hero/odometry',
            self.odom_cb,
            10,
        )
        self.path_sub = self.create_subscription(
            Path,
            '/planning/trajectory',
            self.path_cb,
            10,
        )

        self.ctrl_pub = self.create_publisher(
            CarlaEgoVehicleControl,
            '/carla/hero/vehicle_control_cmd',
            10,
        )

        self.current_odom: Optional[Odometry] = None
        self.path: Optional[Path] = None

        # Shorter lookahead for tighter tracking
        self.lookahead = 5.0  # [m]
        self.wheelbase = 2.7  # [m]

        # Steering limit (rad)
        self.max_steer = 0.5

        self.timer = self.create_timer(0.05, self.timer_cb)  # 20 Hz
        self.debug_counter = 0

    def odom_cb(self, msg: Odometry) -> None:
        self.current_odom = msg

    def path_cb(self, msg: Path) -> None:
        self.path = msg

    def find_target(self, x: float, y: float):
        """Find a target point on the local path at approximately lookahead distance."""
        if self.path is None or not self.path.poses:
            return None

        # Find closest point on local path
        closest_idx = 0
        min_d = float('inf')
        for i, pose in enumerate(self.path.poses):
            px = pose.pose.position.x
            py = pose.pose.position.y
            d = math.hypot(px - x, py - y)
            if d < min_d:
                min_d = d
                closest_idx = i

        # Move forward until lookahead distance reached
        target_idx = closest_idx
        n = len(self.path.poses)
        while target_idx < n:
            px = self.path.poses[target_idx].pose.position.x
            py = self.path.poses[target_idx].pose.position.y
            d = math.hypot(px - x, py - y)
            if d >= self.lookahead:
                break
            target_idx += 1

        if target_idx >= n:
            target_idx = n - 1

        return self.path.poses[target_idx].pose.position

    def timer_cb(self) -> None:
        if self.current_odom is None or self.path is None:
            return

        pose = self.current_odom.pose.pose
        px = pose.position.x
        py = pose.position.y
        q = pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        # Approx current speed for debug
        vx = self.current_odom.twist.twist.linear.x
        vy = self.current_odom.twist.twist.linear.y
        speed = math.hypot(vx, vy)

        target = self.find_target(px, py)
        if target is None:
            return

        dx = target.x - px
        dy = target.y - py

        # Transform target into vehicle coordinate frame
        # x axis: forward, y axis: left
        local_x = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        local_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy

        if local_x < 0.1:
            local_x = 0.1

        # Alpha: angle between vehicle heading and target point
        alpha = math.atan2(local_y, local_x)

        # Pure pursuit curvature
        ld = math.hypot(local_x, local_y)
        curvature = 2.0 * math.sin(alpha) / ld
        raw_steer = math.atan(curvature * self.wheelbase)

        # Clamp steering
        steer = max(-self.max_steer, min(self.max_steer, raw_steer))

        # Distance to end of local path
        goal = self.path.poses[-1].pose.position
        dist_goal = math.hypot(goal.x - px, goal.y - py)

        # Base throttle (stronger so it can climb slope)
        throttle = 0.55
        brake = 0.0

        # Stop near the end
        if dist_goal < 3.0:
            throttle = 0.0
            brake = 1.0

        cmd = CarlaEgoVehicleControl()
        cmd.throttle = float(throttle)
        cmd.steer = float(steer)
        cmd.brake = float(brake)
        cmd.hand_brake = False
        cmd.reverse = False
        cmd.manual_gear_shift = False

        self.ctrl_pub.publish(cmd)

        # Debug log: alpha, local_y, raw_steer, steer, yaw etc.
        self.debug_counter += 1
        if self.debug_counter % 10 == 0:
            self.get_logger().info(
                f'v={speed:.2f} m/s, alpha={alpha:.3f} rad, '
                f'raw_steer={raw_steer:.3f}, steer={steer:.3f}, '
                f'local_x={local_x:.2f}, local_y={local_y:.2f}, '
                f'yaw={yaw:.3f}, dist_goal={dist_goal:.2f}'
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PurePursuitNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
