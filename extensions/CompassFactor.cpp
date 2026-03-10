#include "CompassFactor.hpp"

#include "utils.hpp"

namespace parnav {

template <typename Pose>
auto CompassFactor<Pose>::evaluateError(const Pose& pose,
                                        gtsam::OptionalMatrixType H) const
    -> gtsam::Vector {
  const gtsam::Matrix3 R = pose.rotation().matrix();
  const double r11 = R(0, 0);
  const double r21 = R(1, 0);
  const double r12 = R(0, 1);
  const double r22 = R(1, 1);
  const double r13 = R(0, 2);
  const double r23 = R(1, 2);

  const double predicted_yaw = std::atan2(r21, r11);
  const double error = -ssa(predicted_yaw - measured_yaw_);

  if (H) {
    const double denom = (r11 * r11) + (r21 * r21);
    Jacobian jac = Jacobian::Zero();
    jac(0, 1) = (r11 * r23 - r21 * r13) / denom;
    jac(0, 2) = (-r11 * r22 + r21 * r12) / denom;
    *H = jac;
  }
  return gtsam::Vector1(error);
}

template class CompassFactor<gtsam::Pose3>;
template class CompassFactor<gtsam::Se23>;

}  // namespace parnav
