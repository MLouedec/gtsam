#pragma once

#include "JacobianTraits.hpp"

#include <gtsam/base/Matrix.h>
#include <gtsam/base/types.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace parnav {

using gtsam::symbol_shorthand::X;

template <typename Pose>
class RangeFactor : public gtsam::NoiseModelFactor1<Pose> {
 private:
  using PosJacobian =
      Eigen::Matrix<double, 3, parnav::JacobianTraits<Pose>::Cols>;

 public:
  RangeFactor(gtsam::Key j, const gtsam::SharedNoiseModel& model, double range,
              const gtsam::Point3& l,
              const gtsam::Matrix3& R_r_n = gtsam::Matrix3::Identity())
      : gtsam::NoiseModelFactor1<Pose>{model, j},
        m_z{range},
        m_l{l},
        R_rn_{R_r_n} {}

  auto evaluateError(const Pose& p,
                     gtsam::OptionalMatrixType H1 = OptionalNone) const
      -> gtsam::Vector override;

  [[nodiscard]] auto h_c(const Pose& p) const -> double;
  [[nodiscard]] auto H_c(const Pose& p) const -> gtsam::Matrix13;

 private:
  double m_z;
  gtsam::Point3 m_l;
  gtsam::Matrix3 R_rn_;
};

}  // namespace parnav
