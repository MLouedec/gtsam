#pragma once

#include "JacobianTraits.hpp"

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace parnav {

/// Yaw-only measurement factor. Residual = -ssa(atan2(R21,R11) - measured).
template <typename Pose>
class CompassFactor : public gtsam::NoiseModelFactor1<Pose> {
 private:
  using Jacobian = Eigen::Matrix<double, 1, JacobianTraits<Pose>::Cols>;
  double measured_yaw_;

 public:
  CompassFactor(gtsam::Key key, double measured_yaw,
                const gtsam::SharedNoiseModel& model)
      : gtsam::NoiseModelFactor1<Pose>(model, key),
        measured_yaw_(measured_yaw) {}

  auto evaluateError(const Pose& pose,
                     gtsam::OptionalMatrixType H = OptionalNone) const
      -> gtsam::Vector override;
};

}  // namespace parnav
