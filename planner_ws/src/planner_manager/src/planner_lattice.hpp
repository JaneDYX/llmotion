#pragma once
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include "obstacle_types.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

struct LatticeParams {
  double horizon_length_m = 25.0;
  double sample_ds_m      = 0.5;
  std::vector<double> lateral_offsets_m = {-1.5, -0.75, 0.0, 0.75, 1.5};
  double w_offset = 1.0;
  double w_curv   = 0.3;
};

class LatticePlanner {
public:
  explicit LatticePlanner(const LatticeParams& p) : P(p) {}

  nav_msgs::msg::Path plan(const nav_msgs::msg::Path& ref,
                           const nav_msgs::msg::Odometry& odom,
                           const std::string& frame_id,
                           rclcpp::Time stamp);

  void setObstacles(const Obstacles& obs) { obstacles_ = obs; }
  void setObstacleWeights(double w_obs, double collision_inflation) {
    w_obs_ = w_obs; collision_inflation_m_ = collision_inflation;
  }

private:
  LatticeParams P;

  Obstacles obstacles_;
  double w_obs_{2.0};
  double collision_inflation_m_{0.8};

  size_t findNearestIdx(const nav_msgs::msg::Path& path, double x, double y);
  static double yawBetween(const geometry_msgs::msg::PoseStamped& a,
                           const geometry_msgs::msg::PoseStamped& b);
  static double curvatureAt(const nav_msgs::msg::Path& path, size_t i);
  static double clamp(double v, double lo, double hi) { return std::max(lo, std::min(v, hi)); }
  static double dist2(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2, dy = y1 - y2; return dx*dx + dy*dy;
  }
};
