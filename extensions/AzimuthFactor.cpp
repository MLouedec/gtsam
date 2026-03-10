#include "AzimuthFactor.hpp"

#include "utils.hpp"

namespace parnav {

template <typename Pose>
auto AzimuthFactor<Pose>::evaluateError(
    const Pose& p, gtsam::OptionalMatrixType H1) const -> gtsam::Vector {
  gtsam::Vector3 p_rb_r_hat = R_rn_.transpose() * (parnav::posOf(p) - m_l);

  if (H1) {
    gtsam::Matrix13 H_Psi = H_c(p);
    PosJacobian H_p = PosJacobian::Zero();
    H_p.template block<3, 3>(0, parnav::JacobianTraits<Pose>::PosIdx) =
        R_rn_.transpose() * p.rotation().matrix();
    (*H1) = H_Psi * H_p;
  }

  gtsam::Vector1 error_vec{ssa(h_c(p) - ssa(m_z))};
  return error_vec;
}

template <typename Pose>
auto AzimuthFactor<Pose>::h_c(const Pose& p) const -> double {
  gtsam::Vector3 p_rb_r_hat = R_rn_.transpose() * (parnav::posOf(p) - m_l);
  return ssa(std::atan2(p_rb_r_hat.y(), p_rb_r_hat.x()));
}

template <typename Pose>
auto AzimuthFactor<Pose>::H_c(const Pose& p) const -> gtsam::Matrix13 {
  gtsam::Vector3 p_rb_r_hat = R_rn_.transpose() * (parnav::posOf(p) - m_l);
  double denom =
      (p_rb_r_hat.x() * p_rb_r_hat.x()) + (p_rb_r_hat.y() * p_rb_r_hat.y());
  double scale = 1.0 / denom;
  gtsam::Matrix13 H_Psi =
      scale *
      (gtsam::Matrix(1, 3) << -p_rb_r_hat.y(), p_rb_r_hat.x(), 0.0)
          .finished();
  return H_Psi;
}

template class AzimuthFactor<gtsam::Pose3>;
template class AzimuthFactor<gtsam::Se23>;

}  // namespace parnav
