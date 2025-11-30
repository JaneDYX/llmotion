# llmotion/llm_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .dbl_llama import DBL_LLAMA


class LLMotionNode(Node):
    def __init__(self):
        super().__init__('llmotion_node')

        self.subscription = self.create_subscription(
            String, "/llm_input", self.llm_callback, 10)

        self.publisher = self.create_publisher(String, "/llm_action", 10)

        self.get_logger().info("llmotion LLM Node started")

        # load model
        self.llm = DBL_LLAMA(llm_model_name="qwen", template_name="new")

    def llm_callback(self, msg):
        user_cmd = msg.data.strip()
        self.get_logger().info(f"LLM received: {user_cmd}")

        raw, parsed = self.llm.run(user_cmd)

        self.get_logger().info(f"LLM RAW OUTPUT: {raw}")
        self.get_logger().info(f"LLM ACTION: {parsed}")

        out = String()
        out.data = parsed
        self.publisher.publish(out)



def main(args=None):
    rclpy.init(args=args)
    node = LLMotionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
