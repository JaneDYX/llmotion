#!/usr/bin/env python3
"""
trajectory_to_carla_pid_adapter.py

ROS2 node that:
1) Subscribes to nav_msgs/Path from external planner (e.g., Xavier LLM/Lattice)
2) Projects poses onto CARLA driving lane center waypoints
3) Injects global plan into CARLA BasicAgent LocalPlanner
4) Uses CARLA’s built-in PID controller to follow the plan

This version includes FULL DEBUG OUTPUT.
"""

import argparse
import math
import threading
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path

import carla
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption


# -----------------------------
# Utility — convert quaternion
# -----------------------------
def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


# ================================================================
# TRAJECTORY ADAPTER NODE
# ================================================================
class TrajectoryToCarlaPID(Node):
    def __init__(self, args):
        super().__init__("trajectory_to_carla_pid_adapter")

        # Declare parameters
        self.declare_parameter("traj_topic", args.traj_topic)
        self.declare_parameter("flip_y", args.flip_y)
        self.declare_parameter("flip_yaw", args.flip_yaw)
        self.declare_parameter("min_plan_len", args.min_plan_len)
        self.declare_parameter("target_speed", float(args.target_speed))
        self.declare_parameter("max_wp_jump_m", float(args.max_wp_jump_m))
        self.declare_parameter("draw_waypoints", args.draw_waypoints)
        self.declare_parameter("draw_lines", args.draw_lines)
        self.declare_parameter("ignore_traffic", args.ignore_traffic)

        # Load ROS params
        self.traj_topic = self.get_parameter("traj_topic").value
        self.flip_y = bool(self.get_parameter("flip_y").value)
        self.flip_yaw = bool(self.get_parameter("flip_yaw").value)
        self.min_plan_len = int(self.get_parameter("min_plan_len").value)
        self.target_speed = float(self.get_parameter("target_speed").value)
        self.max_wp_jump_m = float(self.get_parameter("max_wp_jump_m").value)
        self.draw_waypoints = bool(self.get_parameter("draw_waypoints").value)
        self.draw_lines = bool(self.get_parameter("draw_lines").value)
        self.ignore_traffic = bool(self.get_parameter("ignore_traffic").value)

        # Logging
        self.get_logger().info(
            f"Listening traj_topic={self.traj_topic}, flip_y={self.flip_y}, flip_yaw={self.flip_yaw}, "
            f"min_plan_len={self.min_plan_len}, target_speed={self.target_speed}, "
            f"max_wp_jump_m={self.max_wp_jump_m}, draw_waypoints={self.draw_waypoints}, "
            f"draw_lines={self.draw_lines}, ignore_traffic={self.ignore_traffic}"
        )

        # Connect to CARLA
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(args.timeout)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()

        # Find the hero vehicle
        self.vehicle = self._find_hero_vehicle(args.role_name)
        if self.vehicle is None:
            raise RuntimeError("Could not find hero vehicle. Spawn ego with role_name='hero' first.")

        # Create BasicAgent
        self.agent = BasicAgent(self.vehicle, target_speed=self.target_speed)

        if self.ignore_traffic:
            try:
                self.agent.ignore_traffic_lights(True)
                self.agent.ignore_stop_signs(True)
                self.agent.ignore_vehicles(True)
                self.get_logger().info("Traffic rules ignored for demo.")
            except Exception as e:
                self.get_logger().warn(f"Ignore-traffic flags not supported: {e}")

        self._plan_lock = threading.Lock()
        self._plan_set = False
        self._last_plan_size = 0

        # ROS Subscriber
        self.sub = self.create_subscription(Path, self.traj_topic, self.cb_path, 10)

        # Control loop timer
        self.timer = self.create_timer(args.control_dt, self.tick_agent)

    # ------------------------------------------------------------------------------------
    # Locate hero vehicle
    # ------------------------------------------------------------------------------------
    def _find_hero_vehicle(self, role_name: str):
        for v in self.world.get_actors().filter("vehicle.*"):
            if v.attributes.get("role_name") == role_name:
                self.get_logger().info(f"Found vehicle role_name='{role_name}' id={v.id}")
                return v
        actors = self.world.get_actors().filter("vehicle.*")
        if actors:
            self.get_logger().warn(
                f"No vehicle with role_name='{role_name}', fallback to first vehicle id={actors[0].id}"
            )
            return actors[0]
        return None

    # ------------------------------------------------------------------------------------
    # Convert ROS pose → CARLA Location + Yaw
    # ------------------------------------------------------------------------------------
    def ros_pose_to_carla_location_yaw(self, pose):
        x = pose.position.x
        y = -pose.position.y if self.flip_y else pose.position.y
        z = pose.position.z

        yaw = quat_to_yaw(pose.orientation)
        yaw_deg = math.degrees(yaw)
        if self.flip_yaw:
            yaw_deg = -yaw_deg

        return carla.Location(x=x, y=y, z=z), yaw_deg

    # ------------------------------------------------------------------------------------
    # MAIN CALLBACK — RECEIVING TRAJECTORY
    # ------------------------------------------------------------------------------------
    def cb_path(self, msg: Path):
        n = len(msg.poses)
        if n == 0:
            self.get_logger().warn("Received EMPTY trajectory.")
            return

        # =============================
        # DEBUG RAW ROS PATH
        # =============================
        pt0 = msg.poses[0].pose.position
        ptN = msg.poses[-1].pose.position
        self.get_logger().info(
            f"RAW ROS PATH → N={n} | start=({pt0.x:.1f},{pt0.y:.1f}) end=({ptN.x:.1f},{ptN.y:.1f})"
        )

        if n < self.min_plan_len:
            self.get_logger().warn(f"Trajectory too short (N={n} < {self.min_plan_len}). Ignore.")
            return

        # =============================
        # PROJECT ONTO CARLA ROAD
        # =============================
        plan: List[Tuple[carla.Waypoint, RoadOption]] = []
        prev_wp: Optional[carla.Waypoint] = None

        for ps in msg.poses:
            loc, _yaw_deg = self.ros_pose_to_carla_location_yaw(ps.pose)

            wp = self.carla_map.get_waypoint(
                loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            if wp is None:
                continue

            if prev_wp is not None:
                if wp.transform.location.distance(prev_wp.transform.location) > self.max_wp_jump_m:
                    continue

            plan.append((wp, RoadOption.LANEFOLLOW))
            prev_wp = wp

        if len(plan) < self.min_plan_len:
            self.get_logger().warn(f"After projection, plan too short (N={len(plan)}). Ignore.")
            return

        # =============================
        # DEBUG PROJECTED WAYPOINTS
        # =============================
        first = plan[0][0].transform.location
        last = plan[-1][0].transform.location

        self.get_logger().info(
            f"CARLA PATH → N={len(plan)} | start=({first.x:.1f},{first.y:.1f}) "
            f"end=({last.x:.1f},{last.y:.1f})"
        )

        # OPTIONAL FULL PRINT — uncomment if needed
        # for i, (wp, _) in enumerate(plan):
        #     loc = wp.transform.location
        #     self.get_logger().info(f"  [{i:02d}] ({loc.x:.2f}, {loc.y:.2f})")

        # =============================
        # SET NEW GLOBAL PLAN
        # =============================
        with self._plan_lock:
            if self._plan_set and abs(len(plan) - self._last_plan_size) < 3:
                return

            try:
                self.agent._local_planner.set_global_plan(plan)
            except Exception as e:
                self.get_logger().error(f"Failed to set global plan: {e}")
                return

            self._plan_set = True
            self._last_plan_size = len(plan)

        # =============================
        # OPTIONAL DEBUG DRAWING
        # =============================
        if self.draw_waypoints:
            for (wp, _) in plan:
                self.world.debug.draw_point(
                    wp.transform.location,
                    size=0.12,
                    life_time=0.2,
                    persistent_lines=False
                )

        if self.draw_lines:
            for i in range(len(plan) - 1):
                a = plan[i][0].transform.location
                b = plan[i + 1][0].transform.location
                self.world.debug.draw_line(
                    a, b,
                    thickness=0.05,
                    life_time=0.2,
                    persistent_lines=False
                )

        self.get_logger().info(f"Injected global plan. N={len(plan)}")

    # ------------------------------------------------------------------------------------
    # AGENT CONTROL LOOP
    # ------------------------------------------------------------------------------------
    def tick_agent(self):
        with self._plan_lock:
            if not self._plan_set:
                return

        control = self.agent.run_step()
        self.vehicle.apply_control(control)


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--role-name", default="hero")

    parser.add_argument("--traj-topic", default="/planning/trajectory")
    parser.add_argument("--target-speed", type=float, default=10.0)
    parser.add_argument("--min-plan-len", type=int, default=5)
    parser.add_argument("--max-wp-jump-m", type=float, default=5.0)

    parser.add_argument("--flip-y", type=lambda s: s.lower() in ("1", "true", "yes", "y"), default=True)
    parser.add_argument("--flip-yaw", type=lambda s: s.lower() in ("1", "true", "yes", "y"), default=True)

    parser.add_argument("--draw-waypoints", type=lambda s: s.lower() in ("1", "true", "yes", "y"), default=True)
    parser.add_argument("--draw-lines", type=lambda s: s.lower() in ("1", "true", "yes", "y"), default=False)

    parser.add_argument("--ignore-traffic", type=lambda s: s.lower() in ("1", "true", "yes", "y"), default=False)

    parser.add_argument("--control-dt", type=float, default=0.05)

    args = parser.parse_args()

    rclpy.init()
    node = TrajectoryToCarlaPID(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

