#pragma once

// Azimuth-to-known-marker factor for a ship-mounted polar sensor.
//
// The existing parnav::AzimuthFactor models a *fixed* observer (PARS beacon)
// measuring a moving body — its `R_rn_` is the beacon's known rotation, set
// once at construction. For a ship-mounted radar / camera the observer's
// orientation IS the pose being optimized, so R_rn is time-varying and a
// different factor is needed.
//
// Measurement model (planar lift):
//     az_meas = atan2(m.y - p.y, m.x - p.x) - (yaw(p) + sensor_yaw_offset)
// Residual: ssa(predicted - measured) with predicted re-evaluated at every
// linearization.
//
// Sensor xy-offset in the ship frame is assumed zero — true for every polar
// sensor in the current simulator config (radar at [0,0,0], RGB cameras at
// [0,0,±60deg]). If that changes, add the offset to the prediction.

#include "JacobianTraits.hpp"

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace parnav {

template <typename Pose>
class MarkerAzimuthFactor : public gtsam::NoiseModelFactor1<Pose> {
 public:
  MarkerAzimuthFactor(gtsam::Key key, const gtsam::SharedNoiseModel& model,
                      double measured_az, const gtsam::Point3& marker_world,
                      double sensor_yaw_offset_rad)
      : gtsam::NoiseModelFactor1<Pose>(model, key),
        m_z_(measured_az),
        m_l_(marker_world),
        sensor_offset_(sensor_yaw_offset_rad) {}

  gtsam::Vector evaluateError(
      const Pose& p,
      gtsam::OptionalMatrixType H1 = OptionalNone) const override;

 private:
  double m_z_;
  gtsam::Point3 m_l_;
  double sensor_offset_;
};

}  // namespace parnav
