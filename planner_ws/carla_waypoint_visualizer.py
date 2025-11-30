#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path

from carla_msgs.msg import CarlaDebugMarker
from geometry_msgs.msg import Point


class CarlaDebugVisualizer(Node):
    def __init__(self):
        super().__init__("carla_debug_visualizer")

        self.sub = self.create_subscription(
            Path,
            "/planning/trajectory",
            self.traj_callback,
            10
        )

        self.pub = self.create_publisher(
            CarlaDebugMarker,
            "/carla/debug_marker",
            10
        )

        self.get_logger().info("Carla Debug Visualizer started.")

    def clear_old_markers(self):
        """Send a DELETEALL command to CARLA to clear previous debug markers."""
        clear_msg = CarlaDebugMarker()
        clear_msg.type = CarlaDebugMarker.DELETEALL
        self.pub.publish(clear_msg)
        self.get_logger().info("Cleared old markers.")

    def traj_callback(self, msg: Path):
        if len(msg.poses) == 0:
            return

        # 1) Clear old markers first
        self.clear_old_markers()

        # 2) Draw new trajectory points
        self.get_logger().info(f"Drawing {len(msg.poses)} points in CARLA")

        for pose in msg.poses:
            marker = CarlaDebugMarker()

            marker.type = CarlaDebugMarker.POINT
            marker.size = 0.2

            p = Point()
            p.x = float(pose.pose.position.x)
            p.y = float(pose.pose.position.y)
            p.z = float(pose.pose.position.z) + 0.1

            marker.location = p

            # Color (orange)
            marker.color.r = 255
            marker.color.g = 80
            marker.color.b = 0

            # Only draw for 0 seconds → CARLA keeps them until cleared
            marker.life_time = 0.0

            self.pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = CarlaDebugVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
