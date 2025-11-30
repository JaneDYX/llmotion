#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from carla_msgs.msg import CarlaEgoVehicleControl
from tf_transformations import euler_from_quaternion


@dataclass
class Pt2D:
    x: float
    y: float


def normalize_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class StanleyController(Node):
    def __init__(self):
        super().__init__("stanley_controller")

        self.odom_sub = self.create_subscription(
            Odometry, "/carla/hero/odometry", self.odom_cb, 10
        )
        self.path_sub = self.create_subscription(
            Path, "/planning/trajectory", self.path_cb, 10
        )

        self.ctrl_pub = self.create_publisher(
            CarlaEgoVehicleControl, "/carla/hero/vehicle_control_cmd", 10
        )
        
        self.debug_pub = self.create_publisher(
            PoseStamped, "/debug/closest_point", 10
        )

        # Stanley parameters - TUNED FOR BETTER TURNING
        self.k_cte = 3.5  # Increased from 2.5
        self.k_soft = 1.0  # Reduced from 1.5
        self.max_steer = 0.7  # Increased from 0.5
        self.max_cte = 5.0
        
        # Speed control
        self.target_speed = 5.0  # Slightly slower for better control
        self.k_speed = 0.3  # More aggressive speed control
        self.max_throttle = 0.5
        self.min_throttle = 0.15

        # Goal handling
        self.goal_stop_dist = 3.0
        self.slowdown_dist = 15.0

        # Steering rate limit - RELAXED
        self.max_steer_rate = 0.15  # Increased from 0.1

        # State
        self.current_odom: Optional[Odometry] = None
        self.path_pts: List[Pt2D] = []
        self.received_path: bool = False
        self.last_closest_idx: int = 0
        self.prev_steer: float = 0.0
        
        # Safety
        self.max_distance_to_path = 15.0  # Increased tolerance
        self.control_enabled = False

        self.timer = self.create_timer(0.05, self.control_loop)
        self._log_count = 0

        self.get_logger().info("Stanley controller (turning-optimized) started.")

    def odom_cb(self, msg: Odometry):
        self.current_odom = msg

    def path_cb(self, msg: Path):
        pts: List[Pt2D] = []
        for ps in msg.poses:
            pts.append(Pt2D(ps.pose.position.x, ps.pose.position.y))
        if len(pts) >= 2:
            self.path_pts = pts
            self.received_path = True
            self.last_closest_idx = 0
            self.control_enabled = False
            
            if hasattr(self, '_waypoints_flipped'):
                delattr(self, '_waypoints_flipped')

    def control_loop(self):
        if self.current_odom is None or not self.received_path:
            return
        if len(self.path_pts) < 2:
            return

        # Y-axis flip if needed
        if not hasattr(self, '_waypoints_flipped'):
            pos = self.current_odom.pose.pose.position
            
            if (pos.y > 0 and self.path_pts[0].y < 0) or (pos.y < 0 and self.path_pts[0].y > 0):
                self.get_logger().warn("Flipping waypoint Y coordinates...")
                for pt in self.path_pts:
                    pt.y = -pt.y
            
            self._waypoints_flipped = True

        # Extract current state
        pos = self.current_odom.pose.pose.position
        ori = self.current_odom.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])

        vx = self.current_odom.twist.twist.linear.x
        vy = self.current_odom.twist.twist.linear.y
        v = math.hypot(vx, vy)
        v_safe = max(0.5, v)

        cur = Pt2D(pos.x, pos.y)

        # Initial alignment check
        if not self.control_enabled:
            min_dist = float('inf')
            closest_global_idx = 0
            for i, pt in enumerate(self.path_pts):
                d = math.hypot(pt.x - cur.x, pt.y - cur.y)
                if d < min_dist:
                    min_dist = d
                    closest_global_idx = i
            
            self.get_logger().info("=" * 70)
            self.get_logger().info(f"Vehicle: ({cur.x:.2f}, {cur.y:.2f}), yaw={math.degrees(yaw):.1f}°")
            self.get_logger().info(f"Closest[{closest_global_idx}]: ({self.path_pts[closest_global_idx].x:.2f}, {self.path_pts[closest_global_idx].y:.2f}), dist={min_dist:.2f}m")
            
            if min_dist > self.max_distance_to_path:
                self.get_logger().error(f"Too far from path ({min_dist:.1f}m)!")
                ctrl = CarlaEgoVehicleControl()
                ctrl.brake = 1.0
                self.ctrl_pub.publish(ctrl)
                return
            
            self.last_closest_idx = closest_global_idx
            self.control_enabled = True
            self.get_logger().info("Control enabled.")
            self.get_logger().info("=" * 70)

        # Find closest point with STRONG forward bias
        closest_idx = self.find_closest_point_aggressive(cur, yaw)
        self.last_closest_idx = closest_idx
        
        # Debug publish
        debug_pose = PoseStamped()
        debug_pose.header.frame_id = "map"
        debug_pose.header.stamp = self.get_clock().now().to_msg()
        debug_pose.pose.position.x = self.path_pts[closest_idx].x
        debug_pose.pose.position.y = self.path_pts[closest_idx].y
        debug_pose.pose.position.z = 0.5
        self.debug_pub.publish(debug_pose)

        # Path heading
        if closest_idx < len(self.path_pts) - 1:
            path_heading = self.segment_heading(closest_idx, closest_idx + 1)
        else:
            path_heading = self.segment_heading(closest_idx - 1, closest_idx)

        # Heading error
        head_err = normalize_angle(path_heading - yaw)

        # Cross-track error
        cte = self.lateral_error_in_path_frame(cur, self.path_pts[closest_idx], path_heading)
        cte_raw = cte
        cte = max(-self.max_cte, min(self.max_cte, cte))

        # Stanley Control Law
        steer_raw = head_err + math.atan2(self.k_cte * cte, v_safe + self.k_soft)
        steer = max(-self.max_steer, min(self.max_steer, steer_raw))

        # Rate limit (but more relaxed)
        steer = self.rate_limit_steer(steer, self.prev_steer, self.max_steer_rate)
        self.prev_steer = steer

        # Distance to goal
        dist_goal = self.dist_to_goal(cur)

        # Speed control
        target_v = self.target_speed
        if dist_goal < self.slowdown_dist:
            target_v = max(1.5, self.target_speed * (dist_goal / self.slowdown_dist))

        throttle, brake = self.speed_control(v, target_v, dist_goal)

        # Publish control
        ctrl = CarlaEgoVehicleControl()
        ctrl.throttle = float(throttle)
        ctrl.brake = float(brake)
        ctrl.steer = float(steer)
        ctrl.hand_brake = False
        ctrl.reverse = False
        ctrl.manual_gear_shift = False
        self.ctrl_pub.publish(ctrl)

        # Logging
        self._log_count += 1
        if self._log_count % 10 == 0:
            dx = self.path_pts[closest_idx].x - cur.x
            dy = self.path_pts[closest_idx].y - cur.y
            angle_to_wp = math.atan2(dy, dx)
            
            self.get_logger().info("=" * 70)
            self.get_logger().info(f"Frame {self._log_count}")
            self.get_logger().info(f"Vehicle: ({cur.x:.2f},{cur.y:.2f}), yaw={math.degrees(yaw):.1f}°, v={v:.2f}m/s")
            self.get_logger().info(f"Closest[{closest_idx}/{len(self.path_pts)}]: ({self.path_pts[closest_idx].x:.2f},{self.path_pts[closest_idx].y:.2f})")
            self.get_logger().info(f"Angle to WP: {math.degrees(angle_to_wp):.1f}°, Path heading: {math.degrees(path_heading):.1f}°")
            self.get_logger().info(f"Head err: {math.degrees(head_err):.1f}° ({'TURN LEFT' if head_err > 0 else 'TURN RIGHT'})")
            self.get_logger().info(f"CTE: {cte_raw:.3f}m ({'LEFT' if cte_raw > 0 else 'RIGHT'})")
            self.get_logger().info(f"Steer_raw: {steer_raw:.3f}, Steer_final: {steer:.3f} ({math.degrees(steer):.1f}°)")
            self.get_logger().info(f"Throttle: {throttle:.2f}, Goal: {dist_goal:.2f}m")
            self.get_logger().info("=" * 70)

    def find_closest_point_aggressive(self, cur: Pt2D, yaw: float) -> int:
        """
        Aggressive forward-biased closest point finder.
        Never goes backward, strongly prefers points ahead.
        """
        n = len(self.path_pts)
        
        # Search window: only look forward
        search_start = self.last_closest_idx
        search_end = min(n - 1, self.last_closest_idx + 30)

        best_i = search_start
        best_score = float("inf")

        for i in range(search_start, search_end + 1):
            dx = self.path_pts[i].x - cur.x
            dy = self.path_pts[i].y - cur.y
            
            dist = math.hypot(dx, dy)
            forward_proj = dx * math.cos(yaw) + dy * math.sin(yaw)
            
            # Scoring: distance minus forward projection bonus
            if forward_proj < 0:
                score = dist + 1000.0  # Huge penalty for behind
            else:
                score = dist - forward_proj * 0.2  # Bonus for ahead
            
            if score < best_score:
                best_score = score
                best_i = i
        
        # Never go backward
        if best_i < self.last_closest_idx:
            best_i = self.last_closest_idx
        
        return best_i

    def segment_heading(self, i0: int, i1: int) -> float:
        p0 = self.path_pts[i0]
        p1 = self.path_pts[i1]
        return math.atan2(p1.y - p0.y, p1.x - p0.x)

    def lateral_error_in_path_frame(self, cur: Pt2D, ref: Pt2D, path_heading: float) -> float:
        dx = cur.x - ref.x
        dy = cur.y - ref.y
        nx = -math.sin(path_heading)
        ny = math.cos(path_heading)
        return dx * nx + dy * ny

    def rate_limit_steer(self, steer: float, prev: float, max_rate: float) -> float:
        delta = steer - prev
        if delta > max_rate:
            steer = prev + max_rate
        elif delta < -max_rate:
            steer = prev - max_rate
        return steer

    def dist_to_goal(self, cur: Pt2D) -> float:
        goal = self.path_pts[-1]
        return math.hypot(goal.x - cur.x, goal.y - cur.y)

    def speed_control(self, v: float, target_v: float, dist_goal: float) -> Tuple[float, float]:
        if dist_goal < self.goal_stop_dist:
            return 0.0, 1.0

        e_v = target_v - v
        throttle = self.k_speed * e_v
        throttle = max(0.0, min(self.max_throttle, throttle))

        if throttle > 0.0:
            throttle = max(self.min_throttle, throttle)

        brake = 0.0
        if e_v < -0.5:
            brake = min(0.4, (-e_v) * 0.2)

        return throttle, brake


def main():
    rclpy.init()
    node = StanleyController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()