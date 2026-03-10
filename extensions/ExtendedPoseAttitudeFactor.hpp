#pragma once

#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/geometry/Unit3.h>
#include <gtsam/navigation/AttitudeFactor.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace parnav {

/// Attitude factor for Se23: 2D Unit3 residual on rotation.
///
/// Upstream GTSAM 4.3.0 made AttitudeFactor a class template that already
/// handles a generic VALUE by extracting VALUE::rotation(H) (a 3x9 Jacobian for
/// Se23) and forming the 2x9 error Jacobian with the rotation block in columns
/// 0..2 -- exactly what this factor used to do by hand. So it is now just an
/// alias for AttitudeFactor<Se23>.
using ExtendedPoseAttitudeFactor = gtsam::AttitudeFactor<gtsam::Se23>;

}  // namespace parnav
