#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path


class PathRelay(Node):
    def __init__(self) -> None:
        super().__init__('path_relay')
        self.subscription = self.create_subscription(
            Path,
            '/map/reference_path',
            self.reference_callback,
            10
        )
        self.publisher = self.create_publisher(
            Path,
            '/planning/trajectory',
            10
        )
        self.get_logger().info('PathRelay node started.')

    def reference_callback(self, msg: Path) -> None:
        self.get_logger().info(
            f'Relaying path with {len(msg.poses)} poses'
        )
        self.publisher.publish(msg)


def main() -> None:
    rclpy.init()
    node = PathRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
