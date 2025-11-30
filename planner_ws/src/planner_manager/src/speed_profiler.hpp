#pragma once
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <vector>
#include <cmath>

struct SpeedParams {
  double v_max_mps      = 15.0;
  double a_max_mps2     = 2.0;
  double b_max_mps2     = 3.5;
  double alat_max_mps2  = 2.5;
  double min_speed_mps  = 0.5;
  bool   stop_at_end    = false;
};

class SpeedProfiler {
public:
  explicit SpeedProfiler(const SpeedParams& p) : P(p) {}
  std_msgs::msg::Float32MultiArray compute(const nav_msgs::msg::Path& path);

private:
  SpeedParams P;
  static double curvatureAt(const nav_msgs::msg::Path& path, size_t i);
  static std::vector<double> arclengths(const nav_msgs::msg::Path& path);
};
