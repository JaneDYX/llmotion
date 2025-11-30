#include "speed_profiler.hpp"

double SpeedProfiler::curvatureAt(const nav_msgs::msg::Path& path, size_t i) {
  if (i == 0 || i + 1 >= path.poses.size()) return 0.0;
  const auto& p0 = path.poses[i-1].pose.position;
  const auto& p1 = path.poses[i].pose.position;
  const auto& p2 = path.poses[i+1].pose.position;
  double x1 = p1.x - p0.x, y1 = p1.y - p0.y;
  double x2 = p2.x - p1.x, y2 = p2.y - p1.y;
  double cross = x1*y2 - y1*x2;
  double d1 = std::hypot(x1, y1);
  double d2 = std::hypot(x2, y2);
  double d  = std::hypot(p2.x - p0.x, p2.y - p0.y);
  if (d1 < 1e-6 || d2 < 1e-6 || d < 1e-6) return 0.0;
  return 2.0 * cross / (d1 * d2 * d);
}

std::vector<double> SpeedProfiler::arclengths(const nav_msgs::msg::Path& path) {
  std::vector<double> s(path.poses.size(), 0.0);
  for (size_t i = 1; i < path.poses.size(); ++i) {
    const auto& a = path.poses[i-1].pose.position;
    const auto& b = path.poses[i].pose.position;
    s[i] = s[i-1] + std::hypot(b.x - a.x, b.y - a.y);
  }
  return s;
}

std_msgs::msg::Float32MultiArray SpeedProfiler::compute(const nav_msgs::msg::Path& path) {
  std_msgs::msg::Float32MultiArray out;
  const size_t N = path.poses.size();
  if (N == 0) return out;

  std::vector<double> v(N, P.v_max_mps);

  for (size_t i = 0; i < N; ++i) {
    double kappa = curvatureAt(path, i);
    double vmax_curv = (std::abs(kappa) < 1e-6) ? P.v_max_mps
                     : std::sqrt(std::max(1e-6, P.alat_max_mps2 / std::abs(kappa)));
    v[i] = std::min(v[i], std::max(P.min_speed_mps, vmax_curv));
  }

  auto s = arclengths(path);
  if (s.back() < 1e-3) {
    out.data.assign(N, 0.0f);
    return out;
  }

  v[0] = std::min(v[0], P.v_max_mps);
  for (size_t i = 1; i < N; ++i) {
    double ds = std::max(1e-4, s[i] - s[i-1]);
    double v_prev = v[i-1];
    double v_lim = std::sqrt(std::max(0.0, v_prev*v_prev + 2.0 * P.a_max_mps2 * ds));
    v[i] = std::min(v[i], v_lim);
  }

  if (P.stop_at_end) v[N-1] = 0.0;
  for (size_t i = N - 1; i > 0; --i) {
    double ds = std::max(1e-4, s[i] - s[i-1]);
    double v_next = v[i];
    double v_lim  = std::sqrt(std::max(0.0, v_next*v_next + 2.0 * P.b_max_mps2 * ds));
    v[i-1] = std::min(v[i-1], v_lim);
  }

  out.data.resize(N);
  for (size_t i = 0; i < N; ++i) out.data[i] = static_cast<float>(v[i]);
  return out;
}
