#pragma once
#include <vector>
struct Obstacle {
  double x{0.0};
  double y{0.0};
  double r{0.6};
};
using Obstacles = std::vector<Obstacle>;
