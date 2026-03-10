#pragma once

#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace parnav {

/// Position-only GPS measurement on Se23.
/// Residual = position(X) - measured.
class GPSFactorSE23
    : public gtsam::NoiseModelFactorN<gtsam::Se23> {
  gtsam::Point3 measured_;

 public:
  GPSFactorSE23(gtsam::Key key, const gtsam::Point3& measured,
                const gtsam::SharedNoiseModel& model)
      : gtsam::NoiseModelFactorN<gtsam::Se23>(model, key),
        measured_(measured) {}

  gtsam::Vector evaluateError(
      const gtsam::Se23& X,
      gtsam::OptionalMatrixType H = OptionalNone) const override {
    if (H) {
      gtsam::Matrix H_p(3, 9);
      gtsam::Point3 p = X.x(1, H_p);
      *H = H_p;
      return p - measured_;
    }
    return parnav::posOf(X) - measured_;
  }
};

}  // namespace parnav
