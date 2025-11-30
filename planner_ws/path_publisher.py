#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class ReferencePathPublisher(Node):
    def __init__(self):
        super().__init__('reference_path_publisher')
        self.publisher_ = self.create_publisher(Path, '/map/reference_path', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.path_msg = self.build_path()

    def build_path(self) -> Path:
        path = Path()
        path.header.frame_id = 'map'

        control_pts = [
            (10.0, 0.0, 0.0),
            (20.0, 5.0, 0.0),
            (30.0, 10.0, 0.0),
            (40.0, 15.0, 0.0),
        ]

        densify_k = 10  # points inserted per segment

        dense_pts = []
        for i in range(len(control_pts) - 1):
            x0, y0, z0 = control_pts[i]
            x1, y1, z1 = control_pts[i + 1]
            for j in range(densify_k):
                t = j / float(densify_k)
                dense_pts.append((
                    x0 + (x1 - x0) * t,
                    y0 + (y1 - y0) * t,
                    z0 + (z1 - z0) * t
                ))
        dense_pts.append(control_pts[-1])

        for x, y, z in dense_pts:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'map'
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            pose_stamped.pose.position.z = z
            pose_stamped.pose.orientation.x = 0.0
            pose_stamped.pose.orientation.y = 0.0
            pose_stamped.pose.orientation.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            path.poses.append(pose_stamped)

        return path

    def timer_callback(self):
        now = self.get_clock().now().to_msg()
        self.path_msg.header.stamp = now
        for pose in self.path_msg.poses:
            pose.header.stamp = now
        self.publisher_.publish(self.path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ReferencePathPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

