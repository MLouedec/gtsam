#include "BaroFactor.hpp"

namespace parnav {

template <typename Pose>
auto BaroFactor<Pose>::evaluateError(const Pose& pose, const double& bias,
                                     gtsam::OptionalMatrixType H1,
                                     gtsam::OptionalMatrixType H2) const
    -> gtsam::Vector {
  const gtsam::Matrix3 R = pose.rotation().matrix();
  const double p_D = parnav::posOf(pose).z();
  const double z_pred = -p_D + bias;

  constexpr int PosIdx = JacobianTraits<Pose>::PosIdx;
  constexpr int Cols = JacobianTraits<Pose>::Cols;

  if (H1) {
    H1->resize(1, Cols);
    H1->setZero();
    // zvec = (0, 0, -1); derivative w.r.t. position tangent = zvec' * R.
    const Eigen::RowVector3d zvec(0.0, 0.0, -1.0);
    H1->template block<1, 3>(0, PosIdx) = zvec * R;
  }
  if (H2) {
    H2->resize(1, 1);
    (*H2)(0, 0) = 1.0;
  }
  return gtsam::Vector1(z_pred - height_rel_);
}

template class BaroFactor<gtsam::Pose3>;
template class BaroFactor<gtsam::Se23>;

}  // namespace parnav
