#pragma once

// Nearest-neighbor data association for polar detections against known markers.
// Mirrors the Python `_associate_polar` path (without the trust hooks — those
// live one layer up). Inputs are in the SHIP frame; sensor xy-offset is
// assumed zero (true in the current simulator config) and the sensor's yaw
// offset is supplied separately.

#include "MultiModalSimLoader.hpp"

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <vector>

namespace parnav {

struct AssocResult {
  int marker_idx{-1};      // -1 if no marker accepted
  double maha{0.0};         // Mahalanobis distance squared (range² + az²)
  // Marker world position (filled when marker_idx >= 0).
  gtsam::Point3 marker_world{0, 0, 0};
};

// Associate a single (range, azimuth) detection from a ship-mounted polar
// sensor to a known marker.
//
// - `ship_pose`: predicted ship pose at this step.
// - `sensor_yaw_offset_rad`: sensor yaw offset in the ship frame (rad).
// - `range_m`, `azimuth_rad`: the detection in the sensor frame.
// - `sigma_range_m`, `sigma_az_rad`: per-sensor noise (post-deg conversion).
// - `markers`: world-frame marker positions (Nm × 3, row-major).
// - `gate_chi2`: χ² cutoff for the 2-DoF Mahalanobis gate (default 5.99 @ 95%).
AssocResult associateMarker(
    const gtsam::Pose3& ship_pose, double sensor_yaw_offset_rad,
    double range_m, double azimuth_rad, double sigma_range_m,
    double sigma_az_rad, const std::vector<double>& markers,
    int n_markers, double gate_chi2 = 5.99);

// Associate a shoreline detection to the nearest shoreline segment, then
// snap to the closest point on that segment. The "virtual marker" returned
// in `marker_world` is the closest-point projection; downstream, range and
// azimuth factors against this point constrain the ship pose just like a
// known-marker observation.
//
// - `shoreline`: world-frame endpoints, layout (Ns × 2 × 3) row-major.
// - `n_shoreline`: number of segments.
AssocResult associateShoreline(
    const gtsam::Pose3& ship_pose, double sensor_yaw_offset_rad,
    double range_m, double azimuth_rad, double sigma_range_m,
    double sigma_az_rad, const std::vector<double>& shoreline,
    int n_shoreline, double gate_chi2 = 5.99);

// Bearing-only association for a camera detection (no range). Matches the
// bearing to the nearest marker by angular residual alone.
// - `gate_chi2`: χ² cutoff for the 1-DoF gate (default 3.84 @ 95%).
// `AssocResult::maha` holds the azimuth-only Mahalanobis distance squared.
AssocResult associateMarkerBearing(
    const gtsam::Pose3& ship_pose, double sensor_yaw_offset_rad,
    double azimuth_rad, double sigma_az_rad,
    const std::vector<double>& markers, int n_markers,
    double gate_chi2 = 3.84);

// Result of testing a land sensor's detection against both a marker pool
// and the current ship-position estimate — land sensors' raw detection
// stream mixes static-marker sightings with the ship's own live position,
// with no per-detection tag saying which is which, so both hypotheses are
// tested and the nearer one (if within gate) wins.
// `target_world` is only meaningful when `matched && !is_ship` (marker
// case) — on a ship match the caller uses the sensor's own fixed
// world position + yaw directly (same pattern as the NTNU_bluetooth
// ship-tracker factors), so no ship coordinate is echoed back here.
struct LandAssocResult {
  bool matched{false};
  bool is_ship{false};
  double maha{0.0};
  gtsam::Point3 target_world{0, 0, 0};
};

// Range+bearing version, for land-mounted Polar sensors (land_w_radar,
// land_lidar, land_rgb_cam_stereo).
// - `observer_pose`: the land sensor's own fixed pose — pass
//   Pose3(Rot3::Identity(), world_xy) with the sensor's absolute world
//   heading folded into `sensor_yaw_offset_rad` (matches the convention
//   already used for land-sensor ship-tracker factors elsewhere).
// - `ship_position_estimate`: current predicted ship position (world xy).
// Same single nearest-neighbor-then-gate design as associateMarker, just
// with one extra candidate (the ship) added to the pool: a marker
// sitting very close to the ship's current position has the same
// (already-accepted) ambiguity risk as ordinary marker-vs-marker
// association, not a new failure mode.
LandAssocResult associateMarkerOrShip(
    const gtsam::Pose3& observer_pose, double sensor_yaw_offset_rad,
    double range_m, double azimuth_rad, double sigma_range_m,
    double sigma_az_rad, const std::vector<double>& markers, int n_markers,
    const gtsam::Point3& ship_position_estimate, double gate_chi2 = 5.99);

// Bearing-only version, for land-mounted Camera sensors (land_ir_cam).
LandAssocResult associateMarkerBearingOrShip(
    const gtsam::Pose3& observer_pose, double sensor_yaw_offset_rad,
    double azimuth_rad, double sigma_az_rad,
    const std::vector<double>& markers, int n_markers,
    const gtsam::Point3& ship_position_estimate, double gate_chi2 = 3.84);

// Range-only association for a ship-mounted range sensor (no bearing).
// `gate_chi2`: χ² cutoff for the 1-DoF gate (default 3.84 @ 95%).
// `AssocResult::maha` holds the range-only Mahalanobis distance squared.
AssocResult associateMarkerRangeOnly(
    const gtsam::Pose3& ship_pose, double range_m, double sigma_range_m,
    const std::vector<double>& markers, int n_markers,
    double gate_chi2 = 3.84);

// Range-only hybrid marker-or-ship association for land-mounted range
// sensors: same reasoning as associateMarkerOrShip (raw stream mixes
// static-marker sightings with the ship's own live position, no per-
// detection tag), just without a bearing to help discriminate.
LandAssocResult associateMarkerOrShipRangeOnly(
    const gtsam::Pose3& observer_pose, double range_m, double sigma_range_m,
    const std::vector<double>& markers, int n_markers,
    const gtsam::Point3& ship_position_estimate, double gate_chi2 = 3.84);

}  // namespace parnav
