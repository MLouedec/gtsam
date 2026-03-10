#pragma once

#include <cmath>

namespace parnav {

inline auto deg2rad(double deg) -> double { return deg * M_PI / 180.0; }

inline auto rad2deg(double rad) -> double { return rad * 180.0 / M_PI; }

inline auto ssa(double angle) -> double {
  return std::fmod(angle + M_PI, 2.0 * M_PI) - M_PI;
}

}  // namespace parnav
