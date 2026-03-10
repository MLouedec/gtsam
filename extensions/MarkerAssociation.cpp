#include "MarkerAssociation.hpp"

#include "utils.hpp"

#include <cmath>
#include <limits>

namespace parnav {

AssocResult associateMarker(const gtsam::Pose3& ship_pose,
                            double sensor_yaw_offset_rad, double range_m,
                            double azimuth_rad, double sigma_range_m,
                            double sigma_az_rad,
                            const std::vector<double>& markers, int n_markers,
                            double gate_chi2) {
  AssocResult best;
  best.maha = std::numeric_limits<double>::infinity();

  const double sx = ship_pose.translation().x();
  const double sy = ship_pose.translation().y();
  const double yaw = ship_pose.rotation().rpy().z();
  const double sensor_yaw_w = yaw + sensor_yaw_offset_rad;

  const double sigma_r2 = sigma_range_m * sigma_range_m;
  const double sigma_az2 = sigma_az_rad * sigma_az_rad;

  for (int i = 0; i < n_markers; ++i) {
    const double mx = markers[i * 3 + 0];
    const double my = markers[i * 3 + 1];
    const double dx = mx - sx;
    const double dy = my - sy;
    const double pred_range = std::hypot(dx, dy);
    const double pred_az_world = std::atan2(dy, dx);
    const double pred_az_sensor = ssa(pred_az_world - sensor_yaw_w);

    const double dr = range_m - pred_range;
    const double da = ssa(azimuth_rad - pred_az_sensor);
    const double maha = (dr * dr) / sigma_r2 + (da * da) / sigma_az2;

    if (maha < best.maha) {
      best.maha = maha;
      best.marker_idx = i;
      best.marker_world = gtsam::Point3(mx, my, 0.0);
    }
  }

  if (best.marker_idx < 0 || best.maha > gate_chi2) {
    best.marker_idx = -1;
  }
  return best;
}

AssocResult associateShoreline(const gtsam::Pose3& ship_pose,
                               double sensor_yaw_offset_rad, double range_m,
                               double azimuth_rad, double sigma_range_m,
                               double sigma_az_rad,
                               const std::vector<double>& shoreline,
                               int n_shoreline, double gate_chi2) {
  AssocResult best;
  best.maha = std::numeric_limits<double>::infinity();

  const double sx = ship_pose.translation().x();
  const double sy = ship_pose.translation().y();
  const double yaw = ship_pose.rotation().rpy().z();
  const double sensor_yaw_w = yaw + sensor_yaw_offset_rad;

  // World-frame projected detection (ray endpoint from ship through bearing).
  const double world_az = sensor_yaw_w + azimuth_rad;
  const double det_x = sx + range_m * std::cos(world_az);
  const double det_y = sy + range_m * std::sin(world_az);

  const double sigma_r2 = sigma_range_m * sigma_range_m;
  const double sigma_az2 = sigma_az_rad * sigma_az_rad;

  for (int i = 0; i < n_shoreline; ++i) {
    // Segment endpoints (Ns × 2 × 3 row-major, take xy).
    const double ax = shoreline[(i * 2 + 0) * 3 + 0];
    const double ay = shoreline[(i * 2 + 0) * 3 + 1];
    const double bx = shoreline[(i * 2 + 1) * 3 + 0];
    const double by = shoreline[(i * 2 + 1) * 3 + 1];

    // Project detection onto the segment (clamped to [0,1]).
    const double abx = bx - ax;
    const double aby = by - ay;
    const double seg_len2 = abx * abx + aby * aby;
    if (seg_len2 <= 0.0) continue;
    double tparam =
        ((det_x - ax) * abx + (det_y - ay) * aby) / seg_len2;
    if (tparam < 0.0) tparam = 0.0;
    if (tparam > 1.0) tparam = 1.0;
    const double cx = ax + tparam * abx;
    const double cy = ay + tparam * aby;

    // Predicted range / az from ship to the closest point on this segment.
    const double dx = cx - sx;
    const double dy = cy - sy;
    const double pred_range = std::hypot(dx, dy);
    const double pred_az_world = std::atan2(dy, dx);
    const double pred_az_sensor = ssa(pred_az_world - sensor_yaw_w);

    const double dr = range_m - pred_range;
    const double da = ssa(azimuth_rad - pred_az_sensor);
    const double maha = (dr * dr) / sigma_r2 + (da * da) / sigma_az2;

    if (maha < best.maha) {
      best.maha = maha;
      best.marker_idx = i;
      best.marker_world = gtsam::Point3(cx, cy, 0.0);
    }
  }

  if (best.marker_idx < 0 || best.maha > gate_chi2) {
    best.marker_idx = -1;
  }
  return best;
}

AssocResult associateMarkerBearing(const gtsam::Pose3& ship_pose,
                                   double sensor_yaw_offset_rad,
                                   double azimuth_rad, double sigma_az_rad,
                                   const std::vector<double>& markers,
                                   int n_markers, double gate_chi2) {
  AssocResult best;
  best.maha = std::numeric_limits<double>::infinity();

  const double sx = ship_pose.translation().x();
  const double sy = ship_pose.translation().y();
  const double yaw = ship_pose.rotation().rpy().z();
  const double sensor_yaw_w = yaw + sensor_yaw_offset_rad;
  const double sigma_az2 = sigma_az_rad * sigma_az_rad;

  for (int i = 0; i < n_markers; ++i) {
    const double mx = markers[i * 3 + 0];
    const double my = markers[i * 3 + 1];
    const double pred_az_world = std::atan2(my - sy, mx - sx);
    const double pred_az_sensor = ssa(pred_az_world - sensor_yaw_w);
    const double da = ssa(azimuth_rad - pred_az_sensor);
    const double maha = (da * da) / sigma_az2;

    if (maha < best.maha) {
      best.maha = maha;
      best.marker_idx = i;
      best.marker_world = gtsam::Point3(mx, my, 0.0);
    }
  }

  if (best.marker_idx < 0 || best.maha > gate_chi2) {
    best.marker_idx = -1;
  }
  return best;
}

}  // namespace parnav
