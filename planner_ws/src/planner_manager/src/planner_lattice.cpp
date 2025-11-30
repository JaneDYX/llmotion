#include "planner_lattice.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>

size_t LatticePlanner::findNearestIdx(const nav_msgs::msg::Path& path, double x, double y) {
  size_t best = 0; double best_d2 = 1e18;
  for (size_t i = 0; i < path.poses.size(); ++i) {
    const auto& p = path.poses[i].pose.position;
    double d2 = dist2(x, y, p.x, p.y);
    if (d2 < best_d2) { best_d2 = d2; best = i; }
  }
  return best;
}

double LatticePlanner::yawBetween(const geometry_msgs::msg::PoseStamped& a,
                                  const geometry_msgs::msg::PoseStamped& b) {
  double dx = b.pose.position.x - a.pose.position.x;
  double dy = b.pose.position.y - a.pose.position.y;
  return std::atan2(dy, dx);
}

double LatticePlanner::curvatureAt(const nav_msgs::msg::Path& path, size_t i) {
  if (i == 0 || i + 1 >= path.poses.size()) return 0.0;
  const auto& p0 = path.poses[i-1].pose.position;
  const auto& p1 = path.poses[i].pose.position;
  const auto& p2 = path.poses[i+1].pose.position;
  double x1 = p1.x - p0.x, y1 = p1.y - p0.y;
  double x2 = p2.x - p1.x, y2 = p2.y - p1.y;
  double cross = x1*y2 - y1*x2;
  double d1 = std::hypot(x1, y1), d2 = std::hypot(x2, y2), d = std::hypot(p2.x - p0.x, p2.y - p0.y);
  if (d1 < 1e-6 || d2 < 1e-6 || d < 1e-6) return 0.0;
  return 2.0 * cross / (d1 * d2 * d);
}

nav_msgs::msg::Path LatticePlanner::plan(const nav_msgs::msg::Path& ref,
                                         const nav_msgs::msg::Odometry& odom,
                                         const std::string& frame_id,
                                         rclcpp::Time stamp) {
  nav_msgs::msg::Path out;
  out.header.frame_id = frame_id;
  out.header.stamp = stamp;

  if (ref.poses.size() < 3) return out;

  const double x = odom.pose.pose.position.x;
  const double y = odom.pose.pose.position.y;

  size_t i0 = findNearestIdx(ref, x, y);
  double acc_dist = 0.0;
  size_t i_end = i0;
  while (i_end + 1 < ref.poses.size() && acc_dist < P.horizon_length_m) {
    const auto& a = ref.poses[i_end].pose.position;
    const auto& b = ref.poses[i_end+1].pose.position;
    acc_dist += std::hypot(b.x - a.x, b.y - a.y);
    ++i_end;
  }
  if (i_end <= i0) return out;

  double best_cost = 1e18;
  std::vector<geometry_msgs::msg::PoseStamped> best_path;

  for (double dlat : P.lateral_offsets_m) {
    std::vector<geometry_msgs::msg::PoseStamped> candidate;
    candidate.reserve(i_end - i0 + 1);

    for (size_t i = i0; i <= i_end; ++i) {
      auto ps = ref.poses[i];
      if (i + 1 < ref.poses.size()) {
        double yaw = yawBetween(ref.poses[i], ref.poses[i+1]);
        double nx = -std::sin(yaw), ny = std::cos(yaw);
        ps.pose.position.x += dlat * nx;
        ps.pose.position.y += dlat * ny;
      }
      candidate.push_back(ps);
    }

    nav_msgs::msg::Path cand_path;
    cand_path.header = out.header;
    cand_path.poses = candidate;

    double cost_offset = std::abs(dlat) * P.w_offset;

    double cost_curv = 0.0;
    for (size_t k = 1; k + 1 < cand_path.poses.size(); ++k) {
      double kappa = curvatureAt(cand_path, k);
      cost_curv += kappa * kappa;
    }
    cost_curv *= P.w_curv;

    // obstacle cost + collision check
    double cost_obs = 0.0;
    bool collision = false;
    if (!obstacles_.empty()) {
      for (const auto& pose : cand_path.poses) {
        const double px = pose.pose.position.x;
        const double py = pose.pose.position.y;
        for (const auto& ob : obstacles_) {
          const double rinfl = ob.r + collision_inflation_m_;
          const double d2 = dist2(px, py, ob.x, ob.y);
          if (d2 < rinfl * rinfl) { collision = true; break; }
          const double d = std::sqrt(d2) + 1e-3;
          cost_obs += 1.0 / d;
        }
        if (collision) break;
      }
      cost_obs *= w_obs_;
    }

    if (collision) continue;
    double J = cost_offset + cost_curv + cost_obs;

    if (J < best_cost) {
      best_cost = J;
      best_path = std::move(candidate);
    }
  }

  if (best_path.empty()) {
    return out; // no feasible candidate
  }

  out.poses = std::move(best_path);
  return out;
}
