#pragma once

// Loader for the multi-modale-simulator 3D export
// (`simulation_results_3d.npz` + `simulation_results_3d.meta.json`).
//
// The schema is fixed by `f_export_3d.py` in the simulator repo. Every array
// is plain float64; cnpy reads it directly. Sensor identities live in the
// JSON sidecar so cnpy never has to handle strings or object arrays.

#include <Eigen/Core>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace parnav {

struct GnssSensorMeta {
  std::string name;
  double frequency{0.0};  // nominal measurement rate (Hz); 0 = unknown
  int ship{-1};
  std::array<double, 3> relative_pose{{0.0, 0.0, 0.0}};
  double noise_xy{0.0};
  double noise_heading{0.0};
};

struct PolarSensorMeta {
  std::string name;
  double frequency{0.0};  // nominal measurement rate (Hz); 0 = unknown
  int ship{-1};
  std::array<double, 3> relative_pose{{0.0, 0.0, 0.0}};
  double range_noise{0.0};
  double angle_noise_deg{0.0};
  // Non-empty = this sensor tracks the listed ships directly (e.g. a
  // Bluetooth transponder receiver), not static markers. Empty (default) =
  // normal static-marker sensor.
  std::vector<int> detect_ship_ids;
};

struct CameraSensorMeta {
  std::string name;
  double frequency{0.0};  // nominal measurement rate (Hz); 0 = unknown
  int ship{-1};
  std::array<double, 3> relative_pose{{0.0, 0.0, 0.0}};
  double angle_noise_deg{0.0};
};

struct OdomSensorMeta {
  std::string name;
  double frequency{0.0};  // nominal measurement rate (Hz); 0 = unknown
  int ship{-1};
  std::array<double, 3> relative_pose{{0.0, 0.0, 0.0}};
  double noise_pos{0.0};  // m, sigma on dx/dy per delta
  double noise_yaw{0.0};  // rad, sigma on dyaw per delta
};

struct RangeSensorMeta {
  std::string name;
  double frequency{0.0};  // nominal measurement rate (Hz); 0 = unknown
  int ship{-1};
  std::array<double, 3> relative_pose{{0.0, 0.0, 0.0}};
  double range_noise{0.0};
  // Non-empty = this sensor's readings already carry the listed ship's
  // identity (e.g. an acoustic/UWB exchange), same convention as
  // PolarSensorMeta::detect_ship_ids. Empty (default) = static-marker
  // sensor, needs range-only association.
  std::vector<int> detect_ship_ids;
};

struct TrustConfig {
  // Gating layer: innovation pre-checks (bad votes), robust kernels, ALARM1
  // reject windows. Implies subjective_opinion.
  bool activate_gating{false};
  // Track per-sensor Beta counters / opinion (good votes only unless gating
  // is active) and write _trust.csv. Never changes the estimate.
  bool subjective_opinion{false};
  double alpha1{0.9};
  double alpha2{0.99};
  // Continuous-time forgetting (see SelfTrust::setForgetting): fraction of the
  // good/bad counters forgotten per nominal measurement period. Negative =
  // unset -> legacy per-step alpha1/alpha2 behaviour.
  double forget_good{-1.0};
  double forget_bad{-1.0};
  double gnss_pos_thresh{5.99};      // chi2_2 @ 95%
  double gnss_hdg_thresh{3.84};      // chi2_1 @ 95%
  double odom_thresh{7.81};          // chi2_3 @ 95% (dx, dy, dyaw)
  double range_thresh{3.84};         // chi2_1 @ 95%
  double robust_k_mult{3.0};
};

struct GatingConfig {
  double assoc_gate_chi2{5.99};  // chi2_2 @ 95%
};

// Defaults match the pre-existing hardcoded values in MultiModalFixedLag3D.cpp
// so meta.json files without an "imu" section keep behaving identically.
struct ImuNoiseConfig {
  double sigma_acc{0.05};    // m/s^2/sqrt(Hz)
  double sigma_gyro{0.005};  // rad/s/sqrt(Hz)
  double sigma_b_acc{1e-4};
  double sigma_b_gyro{1e-5};
};

// Sensor-subset agent for distributed-FG divergence monitoring. `sensors`
// empty = all sensors (matches the YAML `sensors: null` convention).
struct AgentMeta {
  std::string id;
  std::vector<std::string> sensors;
  // Per-agent overrides (unset = use the global sidecar/CLI value).
  std::optional<bool> activate_gating;
  std::optional<bool> subjective_opinion;
};

struct SimMeta {
  std::string scenario;
  int n_ships{0};
  int T{0};
  int imu_substeps{1};
  double dt_nominal{1.0};
  double gravity{9.81};
  std::vector<GnssSensorMeta> gnss;
  std::vector<PolarSensorMeta> polar;
  std::vector<CameraSensorMeta> camera;
  std::vector<OdomSensorMeta> odom;
  std::vector<RangeSensorMeta> range;
  TrustConfig trust;
  GatingConfig gating;
  ImuNoiseConfig imu;
  std::vector<AgentMeta> agents;
};

// Numeric payload. Indices match SimMeta::{gnss,polar,camera} ordering.
struct SimData3D {
  // (T,)
  std::vector<double> time;
  // (T * M)  per-sub-step dt (s)
  std::vector<double> imu_dt;
  // (T * M)  per-sub-step validity; empty = absent from the npz (old
  // synthetic exports), treat every substep as valid.
  std::vector<std::uint8_t> imu_valid;

  int n_ships{0};
  int T{0};
  int M{1};  // IMU sub-steps per smoother interval

  // (n_ships * T * 7)  layout [x,y,z,qw,qx,qy,qz]
  std::vector<double> gt_pose;
  // (n_ships * T * 3)
  std::vector<double> gt_vel;
  // (n_ships * T * M * 6)  body-frame [wx,wy,wz, ax,ay,az]; imu[.,t] = M
  // sub-samples covering interval (t-1, t]; imu[.,0] unused.
  std::vector<double> imu;

  // GNSS position and heading are independent streams.
  int n_gnss{0};
  std::vector<double> gnss_pos;              // (Ng * T * 3)
  std::vector<std::uint8_t> gnss_pos_valid;  // (Ng * T)
  std::vector<double> gnss_yaw;              // (Ng * T)
  std::vector<std::uint8_t> gnss_yaw_valid;  // (Ng * T)
  // (Ng * T * 2) [std_x, std_y] (m), per-fix dynamic sigma; empty = absent
  // from the npz (older/synthetic exports), fall back to the static
  // per-sensor noise_xy. Even when present, an individual entry <= 0 falls
  // back the same way (real exporter never emits a non-positive std for a
  // valid fix, so this only guards degenerate/placeholder data).
  std::vector<double> gnss_pos_std;

  // Polar marker detections: (Np * T * Kpm * 3) [range, az, el]; valid (Np*T*Kpm)
  int n_polar{0};
  int kmax_polar_marker{0};
  std::vector<double> polar_marker;
  std::vector<std::uint8_t> polar_marker_valid;
  // (Np * T * Kpm * 2) [range_std, azimuth_std] (m, rad); empty/non-positive
  // entry = fall back to the static per-sensor range_noise/angle_noise_deg,
  // same convention as gnss_pos_std above.
  std::vector<double> polar_marker_std;
  // Polar shoreline detections: (Np * T * Kps * 3) [range, az, el]; valid (Np*T*Kps)
  int kmax_polar_shoreline{0};
  std::vector<double> polar_shoreline;
  std::vector<std::uint8_t> polar_shoreline_valid;
  // (Np * T * Kps * 2) [range_std, azimuth_std]; same fallback convention.
  std::vector<double> polar_shoreline_std;

  // Camera: (Nc * T * Kc * 2) [az, el]; valid (Nc * T * Kc)
  int n_camera{0};
  int kmax_camera{0};
  std::vector<double> camera;
  std::vector<std::uint8_t> camera_valid;
  // (Nc * T * Kc) azimuth_std (rad), bearing-only; same fallback convention.
  std::vector<double> camera_std;

  // Odometry (relative pose): (No * T * 3) [dx, dy, dyaw]; index 0 per
  // sensor is unused (no prior state). Empty when n_odom == 0 (older
  // exports, or a config with no Odometry sensors).
  int n_odom{0};
  std::vector<double> odom_delta;
  std::vector<std::uint8_t> odom_delta_valid;
  // (No * T * 3) [std_dx, std_dy, std_dyaw]; empty/non-positive entry ->
  // fall back to the static per-sensor noise_pos/noise_yaw, same
  // convention as gnss_pos_std.
  std::vector<double> odom_delta_std;
  // (No * T) int64 — grid index t' < t that the delta at t is relative to.
  // Empty (older exports) -> default t' = t-1 for every entry.
  std::vector<std::int64_t> odom_ref_step;

  // Range-only detections: (Nr * T * Krm) [range]; valid (Nr * T * Krm).
  // Same bucket serves both static-marker detections (need association)
  // and known-ID ship-tracking detections (RangeSensorMeta::detect_ship_ids
  // set) -- mirrors polar_marker's dual use for Polar ship-trackers.
  int n_range{0};
  int kmax_range_marker{0};
  std::vector<double> range_marker;
  std::vector<std::uint8_t> range_marker_valid;
  // (Nr * T * Krm) range_std (m); same fallback convention as above.
  std::vector<double> range_marker_std;

  // Static
  std::vector<double> markers;            // (Nm * 3)
  int n_markers{0};
  std::vector<double> shoreline;          // (Ns * 2 * 3)
  int n_shoreline{0};

  // Accessors -------------------------------------------------------------
  Eigen::Map<const Eigen::Matrix<double, 7, 1>> gtPose(int ship, int t) const {
    return Eigen::Map<const Eigen::Matrix<double, 7, 1>>(
        gt_pose.data() + (ship * T + t) * 7);
  }
  Eigen::Map<const Eigen::Matrix<double, 3, 1>> gtVel(int ship, int t) const {
    return Eigen::Map<const Eigen::Matrix<double, 3, 1>>(
        gt_vel.data() + (ship * T + t) * 3);
  }
  // k-th IMU sub-sample of interval (t-1, t] for a ship: [wx,wy,wz, ax,ay,az].
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> imuSub(int ship, int t,
                                                       int k) const {
    return Eigen::Map<const Eigen::Matrix<double, 6, 1>>(
        imu.data() + (((ship * T + t) * M) + k) * 6);
  }
  double imuSubDt(int t, int k) const { return imu_dt[t * M + k]; }
  bool imuValid(int t, int k) const {
    return imu_valid.empty() || imu_valid[t * M + k] != 0;
  }
  bool gnssPosValid(int s, int t) const {
    return gnss_pos_valid[s * T + t] != 0;
  }
  Eigen::Map<const Eigen::Matrix<double, 3, 1>> gnssPos(int s, int t) const {
    return Eigen::Map<const Eigen::Matrix<double, 3, 1>>(
        gnss_pos.data() + (s * T + t) * 3);
  }
  bool gnssYawValid(int s, int t) const {
    return gnss_yaw_valid[s * T + t] != 0;
  }
  double gnssYaw(int s, int t) const {
    return gnss_yaw[s * T + t];
  }
  bool hasGnssPosStd() const { return !gnss_pos_std.empty(); }
  Eigen::Map<const Eigen::Matrix<double, 2, 1>> gnssPosStd(int s, int t) const {
    return Eigen::Map<const Eigen::Matrix<double, 2, 1>>(
        gnss_pos_std.data() + (s * T + t) * 2);
  }
  bool polarMarkerValid(int s, int t, int k) const {
    return polar_marker_valid[(s * T + t) * kmax_polar_marker + k] != 0;
  }
  Eigen::Map<const Eigen::Matrix<double, 3, 1>> polarMarkerReading(int s, int t, int k) const {
    return Eigen::Map<const Eigen::Matrix<double, 3, 1>>(
        polar_marker.data() + ((s * T + t) * kmax_polar_marker + k) * 3);
  }
  bool hasPolarMarkerStd() const { return !polar_marker_std.empty(); }
  Eigen::Map<const Eigen::Matrix<double, 2, 1>> polarMarkerStd(int s, int t, int k) const {
    return Eigen::Map<const Eigen::Matrix<double, 2, 1>>(
        polar_marker_std.data() + ((s * T + t) * kmax_polar_marker + k) * 2);
  }
  bool polarShorelineValid(int s, int t, int k) const {
    return polar_shoreline_valid[(s * T + t) * kmax_polar_shoreline + k] != 0;
  }
  Eigen::Map<const Eigen::Matrix<double, 3, 1>> polarShorelineReading(int s, int t, int k) const {
    return Eigen::Map<const Eigen::Matrix<double, 3, 1>>(
        polar_shoreline.data() + ((s * T + t) * kmax_polar_shoreline + k) * 3);
  }
  bool hasPolarShorelineStd() const { return !polar_shoreline_std.empty(); }
  Eigen::Map<const Eigen::Matrix<double, 2, 1>> polarShorelineStd(int s, int t, int k) const {
    return Eigen::Map<const Eigen::Matrix<double, 2, 1>>(
        polar_shoreline_std.data() + ((s * T + t) * kmax_polar_shoreline + k) * 2);
  }
  bool cameraValid(int s, int t, int k) const {
    return camera_valid[(s * T + t) * kmax_camera + k] != 0;
  }
  Eigen::Map<const Eigen::Matrix<double, 2, 1>> cameraReading(int s, int t, int k) const {
    return Eigen::Map<const Eigen::Matrix<double, 2, 1>>(
        camera.data() + ((s * T + t) * kmax_camera + k) * 2);
  }
  bool hasCameraStd() const { return !camera_std.empty(); }
  double cameraStd(int s, int t, int k) const {
    return camera_std[(s * T + t) * kmax_camera + k];
  }
  bool odomDeltaValid(int s, int t) const {
    return odom_delta_valid[s * T + t] != 0;
  }
  Eigen::Map<const Eigen::Matrix<double, 3, 1>> odomDelta(int s, int t) const {
    return Eigen::Map<const Eigen::Matrix<double, 3, 1>>(
        odom_delta.data() + (s * T + t) * 3);
  }
  // Grid index this delta is relative to; defaults to t-1 when odom_ref_step
  // is absent from the npz (older exports).
  long odomRefStep(int s, int t) const {
    if (odom_ref_step.empty()) return static_cast<long>(t) - 1;
    return static_cast<long>(odom_ref_step[s * T + t]);
  }
  bool hasOdomDeltaStd() const { return !odom_delta_std.empty(); }
  Eigen::Map<const Eigen::Matrix<double, 3, 1>> odomDeltaStd(int s, int t) const {
    return Eigen::Map<const Eigen::Matrix<double, 3, 1>>(
        odom_delta_std.data() + (s * T + t) * 3);
  }
  bool rangeMarkerValid(int s, int t, int k) const {
    return range_marker_valid[(s * T + t) * kmax_range_marker + k] != 0;
  }
  double rangeMarkerReading(int s, int t, int k) const {
    return range_marker[(s * T + t) * kmax_range_marker + k];
  }
  bool hasRangeMarkerStd() const { return !range_marker_std.empty(); }
  double rangeMarkerStd(int s, int t, int k) const {
    return range_marker_std[(s * T + t) * kmax_range_marker + k];
  }
};

// Throws std::runtime_error on schema mismatch or IO errors.
SimMeta loadSimMeta(const std::string& json_path);
SimData3D loadSimData(const std::string& npz_path, const SimMeta& meta);

}  // namespace parnav
