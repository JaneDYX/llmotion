# llmotion/keyboard_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class KeyboardInputNode(Node):
    def __init__(self):
        super().__init__('keyboard_input_node')
        self.pub = self.create_publisher(String, "/llm_input", 10)
        self.get_logger().info("Keyboard input node started. Type commands below.\n")

        self.timer = self.create_timer(0.1, self.read_keyboard)

    def read_keyboard(self):
        try:
            user_input = input("> ").strip()
            if user_input:
                msg = String()
                msg.data = user_input
                self.pub.publish(msg)
                self.get_logger().info(f"Published: {user_input}")
        except EOFError:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardInputNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
