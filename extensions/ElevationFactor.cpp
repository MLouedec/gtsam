#include "ElevationFactor.hpp"

#include "utils.hpp"

namespace parnav {

template <typename Pose>
auto ElevationFactor<Pose>::evaluateError(
    const Pose& p, gtsam::OptionalMatrixType H1) const -> gtsam::Vector {
  gtsam::Vector3 p_rb_r_hat = R_rn_.transpose() * (parnav::posOf(p) - m_l);

  if (H1) {
    gtsam::Matrix13 H_alpha = H_c(p);
    PosJacobian H_p = PosJacobian::Zero();
    H_p.template block<3, 3>(0, parnav::JacobianTraits<Pose>::PosIdx) =
        R_rn_.transpose() * p.rotation().matrix();
    (*H1) = H_alpha * H_p;
  }

  gtsam::Vector1 error_vec{ssa(h_c(p) - ssa(m_z))};
  return error_vec;
}

template <typename Pose>
auto ElevationFactor<Pose>::h_c(const Pose& p) const -> double {
  gtsam::Vector3 p_rb_r_hat = R_rn_.transpose() * (parnav::posOf(p) - m_l);
  return ssa(std::atan2(-p_rb_r_hat.z(), p_rb_r_hat.head(2).norm()));
}

template <typename Pose>
auto ElevationFactor<Pose>::H_c(const Pose& p) const -> gtsam::Matrix13 {
  gtsam::Vector3 p_rb_r_hat = R_rn_.transpose() * (parnav::posOf(p) - m_l);

  double rho_u_bar = p_rb_r_hat.head(2).norm();
  double scale = 1.0 / std::pow(p_rb_r_hat.norm(), 2);

  double H_alpha_x = (p_rb_r_hat.x() * p_rb_r_hat.z()) / rho_u_bar;
  double H_alpha_y = (p_rb_r_hat.y() * p_rb_r_hat.z()) / rho_u_bar;
  double H_alpha_z = -rho_u_bar;
  gtsam::Matrix13 H_alpha =
      scale *
      (gtsam::Matrix(1, 3) << H_alpha_x, H_alpha_y, H_alpha_z).finished();

  return H_alpha;
}

template class ElevationFactor<gtsam::Pose3>;
template class ElevationFactor<gtsam::Se23>;

}  // namespace parnav
