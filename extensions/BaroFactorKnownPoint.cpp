#include "BaroFactorKnownPoint.hpp"

namespace parnav {

template <typename Pose>
auto BaroFactorKnownPoint<Pose>::evaluateError(
    const Pose& pose, gtsam::OptionalMatrixType H1) const -> gtsam::Vector {
  const gtsam::Matrix3 R = pose.rotation().matrix();
  const double p_D = parnav::posOf(pose).z();
  const double z_pred = -p_D;

  constexpr int PosIdx = JacobianTraits<Pose>::PosIdx;
  constexpr int Cols = JacobianTraits<Pose>::Cols;

  if (H1) {
    H1->resize(1, Cols);
    H1->setZero();
    const Eigen::RowVector3d zvec(0.0, 0.0, -1.0);
    H1->template block<1, 3>(0, PosIdx) = zvec * R;
  }
  return gtsam::Vector1(z_pred - height_rel_);
}

template class BaroFactorKnownPoint<gtsam::Pose3>;
template class BaroFactorKnownPoint<gtsam::Se23>;

}  // namespace parnav
