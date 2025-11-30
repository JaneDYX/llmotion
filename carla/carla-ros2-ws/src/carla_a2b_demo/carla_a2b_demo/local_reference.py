import math
from typing import Optional

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped


class LocalReferenceNode(Node):
    """
    Local reference path generator.

    - Subscribes to /map/global_path (fixed A->B route).
    - Subscribes to /carla/hero/odometry (current vehicle pose).
    - Finds the closest point on the global path to the vehicle.
    - Extracts a forward segment of the global path as local reference.
    - Publishes the local reference on /planning/trajectory (nav_msgs/Path).

    The key idea:
    - The global path is fixed and does not move with the vehicle.
    - Even if the vehicle deviates from the path, the "closest point" on
      the global path is still on the route, so the local segment points
      the controller back to the route.
    """

    def __init__(self) -> None:
        super().__init__('local_reference_node')

        self.global_path_sub = self.create_subscription(
            Path,
            '/map/global_path',
            self.global_path_cb,
            10,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/carla/hero/odometry',
            self.odom_cb,
            10,
        )

        self.local_path_pub = self.create_publisher(
            Path,
            '/planning/trajectory',
            10,
        )

        self.global_path: Optional[Path] = None
        self.current_odom: Optional[Odometry] = None

        # How many points ahead to include in the local segment
        self.window_size = 40

        # Publish at a fixed rate
        self.timer = self.create_timer(0.05, self.timer_cb)  # 20 Hz

    def global_path_cb(self, msg: Path) -> None:
        self.global_path = msg

    def odom_cb(self, msg: Odometry) -> None:
        self.current_odom = msg

    def find_closest_index(self, x: float, y: float) -> int:
        """Return the index of the closest point on the global path."""
        if self.global_path is None or not self.global_path.poses:
            return 0

        closest_idx = 0
        min_d = float('inf')
        for i, pose in enumerate(self.global_path.poses):
            px = pose.pose.position.x
            py = pose.pose.position.y
            d = math.hypot(px - x, py - y)
            if d < min_d:
                min_d = d
                closest_idx = i
        return closest_idx

    def timer_cb(self) -> None:
        if self.global_path is None or self.current_odom is None:
            return

        # Current vehicle position
        pose = self.current_odom.pose.pose
        x = pose.position.x
        y = pose.position.y

        closest_idx = self.find_closest_index(x, y)

        # Extract a forward segment from the global path
        start = closest_idx
        end = min(closest_idx + self.window_size, len(self.global_path.poses))

        if start >= end:
            return

        local_path = Path()
        local_path.header.frame_id = self.global_path.header.frame_id
        now = self.get_clock().now().to_msg()
        local_path.header.stamp = now

        for p in self.global_path.poses[start:end]:
            q = PoseStamped()
            q.header.frame_id = local_path.header.frame_id
            q.header.stamp = now
            q.pose = p.pose
            local_path.poses.append(q)

        self.local_path_pub.publish(local_path)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LocalReferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
