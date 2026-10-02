#pragma once

#include <cmath>

namespace parnav {

inline auto deg2rad(double deg) -> double { return deg * M_PI / 180.0; }

inline auto rad2deg(double rad) -> double { return rad * 180.0 / M_PI; }

inline auto ssa(double angle) -> double {
  // std::fmod returns a remainder with the sign of the dividend (not a
  // floored/always-non-negative remainder), so angle < -pi falls through
  // unwrapped without this correction.
  double r = std::fmod(angle + M_PI, 2.0 * M_PI);
  if (r < 0.0) r += 2.0 * M_PI;
  return r - M_PI;
}

}  // namespace parnav
