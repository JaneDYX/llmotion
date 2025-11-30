import math
from typing import Optional, List

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

# Try to import CARLA Python API (for lane waypoints and drawing).
try:
    import carla  # type: ignore
except ImportError:
    carla = None


class GlobalPlannerNode(Node):
    """
    Global planner that generates a fixed A->B path in the CARLA world.

    - Connects to CARLA, finds the hero vehicle (role_name == "hero").
    - Uses CARLA's lane waypoint API to follow the lane center ahead.
    - Generates a single global Path and keeps it fixed (no replanning).
    - Publishes the global path on /map/global_path.
    - Draws the waypoints in CARLA as red points for visualization.
    """

    def __init__(self) -> None:
        super().__init__('global_planner_node')

        self.path_pub = self.create_publisher(Path, '/map/global_path', 1)

        # Timer: first calls generate the path; later just publish it.
        self.timer = self.create_timer(0.5, self.timer_cb)

        self.initialized = False
        self.global_path: Optional[Path] = None

        # Connect to CARLA world and map
        self.carla_world = None
        self.carla_map = None
        if carla is not None:
            try:
                client = carla.Client('localhost', 2000)
                client.set_timeout(3.0)
                self.carla_world = client.get_world()
                self.carla_map = self.carla_world.get_map()
                self.get_logger().info('Connected to CARLA for global planning.')
            except Exception as e:
                self.get_logger().warn(
                    f'Failed to connect to CARLA Python API: {e}. '
                    'Global path will not be generated.'
                )
        else:
            self.get_logger().warn(
                'carla Python module not found. Global path will not be generated.'
            )

    # ------------------------------------------------------------------
    # Path generation using CARLA lane waypoints
    # ------------------------------------------------------------------
    def build_global_path(self) -> None:
        """Use CARLA lane waypoints to build a fixed global path ahead of the hero."""
        if self.carla_world is None or self.carla_map is None:
            self.get_logger().warn('No CARLA world/map. Cannot build global path.')
            return

        # Find hero vehicle
        actors = self.carla_world.get_actors().filter('vehicle.*')
        hero = None
        for veh in actors:
            role = veh.attributes.get('role_name', '')
            if role == 'hero':
                hero = veh
                break

        if hero is None:
            self.get_logger().warn('Hero vehicle not found in CARLA yet.')
            return

        hero_loc = hero.get_location()

        # Project hero position to nearest driving lane waypoint
        waypoint = self.carla_map.get_waypoint(
            hero_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            self.get_logger().warn('No driving-lane waypoint found near hero.')
            return

        self.get_logger().info(
            f'Start waypoint: road_id={waypoint.road_id}, lane_id={waypoint.lane_id}, s={waypoint.s:.2f}'
        )

        path = Path()
        path.header.frame_id = 'map'

        distance_step = 2.0   # meters between waypoints
        max_distance = 120.0  # total distance ahead
        accumulated = 0.0

        current_wp = waypoint
        waypoints: List[carla.Waypoint] = [current_wp]

        while accumulated < max_distance:
            next_wps = current_wp.next(distance_step)
            if not next_wps:
                break
            current_wp = next_wps[0]
            waypoints.append(current_wp)
            accumulated += distance_step

        self.get_logger().info(
            f'Global path collected {len(waypoints)} lane waypoints.'
        )

        for wp in waypoints:
            loc = wp.transform.location
            p = PoseStamped()
            p.header.frame_id = 'map'
            p.pose.position.x = float(loc.x)
            p.pose.position.y = float(loc.y)
            p.pose.position.z = float(loc.z)
            p.pose.orientation.w = 1.0
            path.poses.append(p)

        self.global_path = path
        self.initialized = True

        self.get_logger().info(
            f'Global path generated with {len(self.global_path.poses)} points.'
        )

        # Draw the global waypoints in CARLA
        self.draw_waypoints_in_carla()

    def draw_waypoints_in_carla(self) -> None:
        """Draw global path waypoints as red points in CARLA."""
        if self.carla_world is None or self.global_path is None or carla is None:
            self.get_logger().warn(
                'Cannot draw global waypoints in CARLA (missing world/path/module).'
            )
            return

        debug = self.carla_world.debug
        color = carla.Color(255, 0, 0)  # red
        life_time = 0.0                 # persistent
        size = 0.25

        for pose in self.global_path.poses:
            x = pose.pose.position.x
            y = -pose.pose.position.y
            z = pose.pose.position.z + 0.5
            loc = carla.Location(x=float(x), y=float(y), z=float(z))
            debug.draw_point(loc, size=size, color=color, life_time=life_time)

        self.get_logger().info('Global waypoints have been drawn in CARLA (red).')

    # ------------------------------------------------------------------
    # Timer callback: generate once, then publish
    # ------------------------------------------------------------------
    def timer_cb(self) -> None:
        if not self.initialized:
            self.build_global_path()
            if not self.initialized:
                # Not ready yet; wait for next cycle
                return

        if self.global_path is None:
            return

        now = self.get_clock().now().to_msg()
        self.global_path.header.stamp = now
        for p in self.global_path.poses:
            p.header.stamp = now

        self.path_pub.publish(self.global_path)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GlobalPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
