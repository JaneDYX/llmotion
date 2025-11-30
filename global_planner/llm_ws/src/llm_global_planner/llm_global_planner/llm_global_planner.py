import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO


class LLMGlobalPlannerNode(Node):

    def __init__(self):
        super().__init__("llm_global_planner")

        # Connect to CARLA
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)

        self.world = self.client.get_world()
        self.map = self.world.get_map()

        # ====== Get ego vehicle (hero) ======
        self.ego_vehicle = None
        self.find_ego_vehicle()

        # ====== Global Route Planner ======
        sampling_resolution = 2.0
        dao = GlobalRoutePlannerDAO(self.map, sampling_resolution)
        self.grp = GlobalRoutePlanner(dao, sampling_resolution)

        # LLM commands
        self.cmd_sub = self.create_subscription(
            String,
            "/llm_action",
            self.action_callback,
            10
        )

        # Publish global path to local planner
        self.path_pub = self.create_publisher(Path, "/map/reference_path", 10)

        self.get_logger().info("LLM Global Planner Node READY (using CARLA transform)")

    # ==========================================================
    #  FIND EGO VEHICLE BY ROLE NAME
    # ==========================================================
    def find_ego_vehicle(self):
        actors = self.world.get_actors()
        for actor in actors:
            if actor.attributes.get("role_name") == "hero":
                self.ego_vehicle = actor
                self.get_logger().info(f"Found ego vehicle: id={actor.id}")
                return
        self.get_logger().error("Could not find hero vehicle in CARLA.")

    # ==========================================================
    #  LLM COMMAND CALLBACK
    # ==========================================================
    def action_callback(self, msg: String):
        if self.ego_vehicle is None:
            self.get_logger().error("Ego vehicle not found!")
            return

        command = msg.data.lower().strip()
        self.get_logger().info(f"Received LLM command: {command}")

        # == Get CARLA vehicle location + waypoint ==
        ego_transform = self.ego_vehicle.get_transform()
        start_loc = ego_transform.location

        start_wp = self.map.get_waypoint(
            start_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        # ==========================================================
        #  DETERMINE TARGET WAYPOINT
        # ==========================================================
        if command == "left":
            goal_wp = start_wp.get_left_lane()
        elif command == "right":
            goal_wp = start_wp.get_right_lane()
        elif command == "uturn":
            left = start_wp.get_left_lane()
            goal_wp = left.get_left_lane() if left else None
        elif command == "stop":
            goal_wp = start_wp
        else:  # "forward"
            next_list = start_wp.next(30.0)
            goal_wp = next_list[0] if next_list else None

        if goal_wp is None:
            self.get_logger().error("No valid target waypoint for this command.")
            return

        # ==========================================================
        #  GLOBAL ROUTE
        # ==========================================================
        route = self.grp.trace_route(
            start_wp.transform.location,
            goal_wp.transform.location
        )

        self.publish_path(route)

    # ==========================================================
    #  PUBLISH PATH + DRAW ON CARLA
    # ==========================================================
    def publish_path(self, route):
        path = Path()
        path.header.frame_id = "map"

        debug = self.world.debug
        last_wp = None

        for wp, _ in route:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = wp.transform.location.x
            pose.pose.position.y = wp.transform.location.y
            pose.pose.position.z = wp.transform.location.z
            path.poses.append(pose)

            # Draw point in CARLA
            debug.draw_point(
                wp.transform.location,
                size=0.1,
                color=carla.Color(0, 0, 255),
                life_time=0.3,
                persistent_lines=False
            )

            # Draw line
            if last_wp is not None:
                debug.draw_line(
                    last_wp.transform.location,
                    wp.transform.location,
                    thickness=0.1,
                    color=carla.Color(0, 0, 255),
                    life_time=0.3,
                    persistent_lines=False
                )

            last_wp = wp

        self.path_pub.publish(path)
        self.get_logger().info(f"Published route with {len(path.poses)} points")


def main(args=None):
    rclpy.init(args=args)
    node = LLMGlobalPlannerNode()
    rclpy.spin(node)
    rclpy.shutdown()
