#pragma once

#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>

namespace parnav {

template <typename LieGroup> struct JacobianTraits;

template <> struct JacobianTraits<gtsam::Pose3> {
  static constexpr int Cols = 6;
  static constexpr int PosIdx = 3;
};

// Se23 tangent order is (theta, nu, rho) = (att, vel, pos), so the
// position block lives at tangent index 6.
template <> struct JacobianTraits<gtsam::Se23> {
  static constexpr int Cols = 9;
  static constexpr int PosIdx = 6;
};

// Uniform position accessor: Pose3 exposes translation(); the upstream 4.3.0
// Se23 (= ExtendedPose3<2>) exposes its position as translation column x(1).
inline gtsam::Point3 posOf(const gtsam::Pose3& X) { return X.translation(); }
inline gtsam::Point3 posOf(const gtsam::Se23& X) { return X.x(1); }

}  // namespace parnav
