import math
from typing import Optional, List

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

# Try to import CARLA Python API (for drawing waypoints in the simulator).
try:
    import carla  # type: ignore
except ImportError:
    carla = None


class ReferencePathNode(Node):
    """
    Generate a reference path along the lane center using CARLA waypoints,
    starting from the hero vehicle position, and draw the waypoints in CARLA.

    - Queries the CARLA world for the vehicle with role_name == "hero".
    - Uses CARLA's lane waypoint API to follow the road ahead of the hero.
    - Publishes the path on /map/reference_path (nav_msgs/Path).
    - If CARLA Python API is available, draws small red points for each
      waypoint directly in the CARLA world.
    """

    def __init__(self) -> None:
        super().__init__('reference_path_node')

        # Publisher for the reference path
        self.path_pub = self.create_publisher(Path, '/map/reference_path', 1)

        # Timer: first calls build the path, later just publish it
        self.timer = self.create_timer(0.5, self.timer_cb)

        self.initialized = False
        self.path_msg: Optional[Path] = None

        # Connect to CARLA world (optional)
        self.carla_world = None
        self.carla_map = None
        if carla is not None:
            try:
                client = carla.Client('localhost', 2000)
                client.set_timeout(3.0)
                self.carla_world = client.get_world()
                self.carla_map = self.carla_world.get_map()
                self.get_logger().info('Connected to CARLA for waypoint-based path.')
            except Exception as e:
                self.get_logger().warn(
                    f'Failed to connect to CARLA Python API: {e}. '
                    'Waypoints will not be drawn and no lane-based path.'
                )
        else:
            self.get_logger().warn(
                'carla Python module not found. Lane-based path will not be used.'
            )

    # ------------------------------------------------------------------
    # Path generation using CARLA lane waypoints
    # ------------------------------------------------------------------
    def build_path_from_lane(self) -> None:
        """Use CARLA map waypoints to build a lane-following path ahead of the hero."""
        if self.carla_world is None or self.carla_map is None:
            self.get_logger().warn('No CARLA world/map. Cannot build lane-based path.')
            return

        # Find the hero vehicle
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

        # Project hero position to the nearest driving lane waypoint
        waypoint = self.carla_map.get_waypoint(
            hero_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            self.get_logger().warn('No driving-lane waypoint found near hero.')
            return

        self.get_logger().info(
            f'Starting waypoint: road_id={waypoint.road_id}, lane_id={waypoint.lane_id}, '
            f's={waypoint.s:.2f}'
        )

        path = Path()
        path.header.frame_id = 'map'

        # Collect waypoints along the lane center ahead of the hero
        distance_step = 2.0     # meters between waypoints
        max_distance = 80.0     # total distance ahead
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
            f'Collected {len(waypoints)} lane waypoints ahead of hero.'
        )

        # Convert CARLA waypoints to ROS Path
        for wp in waypoints:
            loc = wp.transform.location

            p = PoseStamped()
            p.header.frame_id = 'map'
            p.pose.position.x = float(loc.x)
            p.pose.position.y = float(loc.y)
            # Use lane waypoint Z, slightly lowered so points are closer to the road surface.
            p.pose.position.z = float(loc.z)
            p.pose.orientation.w = 1.0
            path.poses.append(p)

        self.path_msg = path
        self.initialized = True
        self.get_logger().info(
            f'Reference path (lane-based) generated with {len(self.path_msg.poses)} points.'
        )

        # Draw waypoints in CARLA for visualization
        self.draw_waypoints_in_carla()

    def draw_waypoints_in_carla(self) -> None:
        """Draw each waypoint as a small red point in the CARLA world."""
        if self.carla_world is None or self.path_msg is None or carla is None:
            self.get_logger().warn(
                'Cannot draw waypoints in CARLA (no world or path or carla module).'
            )
            return

        debug = self.carla_world.debug
        color = carla.Color(255, 0, 0)  # red
        life_time = 0.0  # persistent until the world is restarted
        size = 0.25

        for pose in self.path_msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            z = pose.pose.position.z + 0.5  # slightly above road surface for visibility
            location = carla.Location(x=float(x), y=float(y), z=float(z))
            debug.draw_point(location, size=size, color=color, life_time=life_time)

        self.get_logger().info('Lane waypoints have been drawn in CARLA (red points).')

    # ------------------------------------------------------------------
    # Timer callback
    # ------------------------------------------------------------------
    def timer_cb(self) -> None:
        """On first calls, try to build the path; later just publish it."""
        if not self.initialized:
            self.build_path_from_lane()
            if not self.initialized:
                # Still not ready; wait for next timer tick.
                return

        if self.path_msg is None:
            return

        now = self.get_clock().now().to_msg()
        self.path_msg.header.stamp = now
        for p in self.path_msg.poses:
            p.header.stamp = now

        self.path_pub.publish(self.path_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ReferencePathNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
