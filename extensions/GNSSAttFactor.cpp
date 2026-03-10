#include "GNSSAttFactor.hpp"

namespace parnav {

template <typename Pose>
auto GNSSAttFactor<Pose>::evaluateError(const Pose& pose,
                                        gtsam::OptionalMatrixType H) const
    -> gtsam::Vector {
  const gtsam::Matrix3 R = pose.rotation().matrix();
  const gtsam::Vector3 pred = R * baseline_body_;
  const gtsam::Vector3 error = pred - measured_;

  if (H) {
    Jacobian H_full = Jacobian::Zero();
    // Attitude block lives at tangent indices 0..2 for both Pose3 and Se23.
    H_full.template block<3, 3>(0, 0) = -R * gtsam::skewSymmetric(baseline_body_);
    *H = H_full;
  }
  return error;
}

template class GNSSAttFactor<gtsam::Pose3>;
template class GNSSAttFactor<gtsam::Se23>;

}  // namespace parnav
