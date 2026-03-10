#pragma once

#include "JacobianTraits.hpp"

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace parnav {

/// Attitude-from-baseline factor.
/// Measurement model: z = R_nb * b_body  (baseline expressed in nav frame).
/// Residual = R_nb * b_body - measured.
/// Jacobian (attitude block): -R_nb * skew(b_body).
template <typename Pose>
class GNSSAttFactor : public gtsam::NoiseModelFactor1<Pose> {
 private:
  using Jacobian = Eigen::Matrix<double, 3, JacobianTraits<Pose>::Cols>;
  gtsam::Point3 baseline_body_;
  gtsam::Point3 measured_;

 public:
  GNSSAttFactor(gtsam::Key key, const gtsam::Point3& measured,
                const gtsam::Point3& baseline_body,
                const gtsam::SharedNoiseModel& model)
      : gtsam::NoiseModelFactor1<Pose>(model, key),
        baseline_body_(baseline_body),
        measured_(measured) {}

  auto evaluateError(const Pose& pose,
                     gtsam::OptionalMatrixType H = OptionalNone) const
      -> gtsam::Vector override;
};

}  // namespace parnav
