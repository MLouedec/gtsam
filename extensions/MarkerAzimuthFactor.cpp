#include "MarkerAzimuthFactor.hpp"

#include "utils.hpp"

#include <gtsam/geometry/ExtendedPose3.h>

namespace parnav {

template <typename Pose>
gtsam::Vector MarkerAzimuthFactor<Pose>::evaluateError(
    const Pose& p, gtsam::OptionalMatrixType H1) const {
  const double px = parnav::posOf(p).x();
  const double py = parnav::posOf(p).y();
  // Yaw extracted via ZYX convention so it matches the simulator's planar
  // heading. The planar pin keeps roll/pitch near zero, so the Jacobian
  // approximation `dyaw/dδR ≈ (0,0,1)` is accurate near the linearization
  // point.
  const double yaw = p.rotation().rpy().z();

  const double dx = m_l_.x() - px;
  const double dy = m_l_.y() - py;
  const double r2 = dx * dx + dy * dy;

  const double bearing_world = std::atan2(dy, dx);
  const double predicted = ssa(bearing_world - yaw - sensor_offset_);
  const double residual = ssa(predicted - ssa(m_z_));

  if (H1) {
    // Tangent order for Pose3: (rx, ry, rz, tx, ty, tz).
    // Rotation: under the planar pin only omega_z affects yaw, so
    //   d(error)/d(omega_z) = d(predicted)/d(yaw) = -1.
    // Translation: the tangent rho is in BODY frame; world translation is
    //   t_new = t + R * rho, so chain through R.
    gtsam::Matrix H = gtsam::Matrix::Zero(1, JacobianTraits<Pose>::Cols);
    H(0, 2) = -1.0;
    Eigen::Matrix<double, 1, 3> J_T_world;
    J_T_world << dy / r2, -dx / r2, 0.0;
    Eigen::Matrix<double, 1, 3> J_T_body =
        J_T_world * p.rotation().matrix();
    const int p_idx = JacobianTraits<Pose>::PosIdx;
    H(0, p_idx + 0) = J_T_body(0);
    H(0, p_idx + 1) = J_T_body(1);
    H(0, p_idx + 2) = J_T_body(2);
    *H1 = H;
  }

  return (gtsam::Vector(1) << residual).finished();
}

template class MarkerAzimuthFactor<gtsam::Pose3>;
template class MarkerAzimuthFactor<gtsam::Se23>;

}  // namespace parnav
