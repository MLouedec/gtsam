#include "RangeFactor.hpp"

namespace parnav {

template <typename Pose>
auto RangeFactor<Pose>::h_c(const Pose& p) const -> double {
  gtsam::Vector3 p_rb_r_hat = R_rn_.transpose() * (parnav::posOf(p) - m_l);
  return p_rb_r_hat.norm();
}

template <typename Pose>
auto RangeFactor<Pose>::H_c(const Pose& p) const -> gtsam::Matrix13 {
  gtsam::Vector3 p_rb_r_hat = R_rn_.transpose() * (parnav::posOf(p) - m_l);
  return p_rb_r_hat.transpose() / p_rb_r_hat.norm();
}

template <typename Pose>
auto RangeFactor<Pose>::evaluateError(
    const Pose& p, gtsam::OptionalMatrixType H1) const -> gtsam::Vector {
  gtsam::Vector3 p_rb_r_hat = R_rn_.transpose() * (parnav::posOf(p) - m_l);

  if (H1) {
    gtsam::Matrix H_rho = H_c(p);
    PosJacobian H_p = PosJacobian::Zero();
    H_p.template block<3, 3>(0, parnav::JacobianTraits<Pose>::PosIdx) =
        R_rn_.transpose() * p.rotation().matrix();
    (*H1) = H_rho * H_p;
  }

  gtsam::Vector1 error_vec{h_c(p) - m_z};
  return error_vec;
}

template class RangeFactor<gtsam::Pose3>;
template class RangeFactor<gtsam::Se23>;

}  // namespace parnav
