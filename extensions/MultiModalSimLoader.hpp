#pragma once

// Loader for the multi-modale-simulator 3D export
// (`simulation_results_3d.npz` + `simulation_results_3d.meta.json`).
//
// The schema is fixed by `f_export_3d.py` in the simulator repo. Every array
// is plain float64; cnpy reads it directly. Sensor identities live in the
// JSON sidecar so cnpy never has to handle strings or object arrays.

#include <Eigen/Core>
#include <cstdint>
#include <string>
#include <vector>

namespace parnav {

struct GnssSensorMeta {
  std::string name;
  int ship{-1};
  std::array<double, 3> relative_pose{{0.0, 0.0, 0.0}};
  double noise_xy{0.0};
  double noise_heading{0.0};
};

struct PolarSensorMeta {
  std::string name;
  int ship{-1};
  std::array<double, 3> relative_pose{{0.0, 0.0, 0.0}};
  double range_noise{0.0};
  double angle_noise_deg{0.0};
};

struct CameraSensorMeta {
  std::string name;
  int ship{-1};
  std::array<double, 3> relative_pose{{0.0, 0.0, 0.0}};
  double angle_noise_deg{0.0};
};

struct TrustConfig {
  bool enable{false};
  std::string scaling{"inverse"};   // inverse | inverse_sqrt | linear | off
  double floor{0.001};
  double linear_k{5.0};
  double alpha1{0.9};
  double alpha2{0.99};
  double gate_bad_ratio{0.5};
  double gnss_pos_thresh{5.99};      // chi2_2 @ 95%
  double gnss_hdg_thresh{3.84};      // chi2_1 @ 95%
  double robust_k_mult{3.0};
  bool gnss_veto{false};
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
  TrustConfig trust;
};

// Numeric payload. Indices match SimMeta::{gnss,polar,camera} ordering.
struct SimData3D {
  // (T,)
  std::vector<double> time;
  // (T * M)  per-sub-step dt (s)
  std::vector<double> imu_dt;

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

  // Polar marker detections: (Np * T * Kpm * 3) [range, az, el]; valid (Np*T*Kpm)
  int n_polar{0};
  int kmax_polar_marker{0};
  std::vector<double> polar_marker;
  std::vector<std::uint8_t> polar_marker_valid;
  // Polar shoreline detections: (Np * T * Kps * 3) [range, az, el]; valid (Np*T*Kps)
  int kmax_polar_shoreline{0};
  std::vector<double> polar_shoreline;
  std::vector<std::uint8_t> polar_shoreline_valid;

  // Camera: (Nc * T * Kc * 2) [az, el]; valid (Nc * T * Kc)
  int n_camera{0};
  int kmax_camera{0};
  std::vector<double> camera;
  std::vector<std::uint8_t> camera_valid;

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
  bool polarMarkerValid(int s, int t, int k) const {
    return polar_marker_valid[(s * T + t) * kmax_polar_marker + k] != 0;
  }
  Eigen::Map<const Eigen::Matrix<double, 3, 1>> polarMarkerReading(int s, int t, int k) const {
    return Eigen::Map<const Eigen::Matrix<double, 3, 1>>(
        polar_marker.data() + ((s * T + t) * kmax_polar_marker + k) * 3);
  }
  bool polarShorelineValid(int s, int t, int k) const {
    return polar_shoreline_valid[(s * T + t) * kmax_polar_shoreline + k] != 0;
  }
  Eigen::Map<const Eigen::Matrix<double, 3, 1>> polarShorelineReading(int s, int t, int k) const {
    return Eigen::Map<const Eigen::Matrix<double, 3, 1>>(
        polar_shoreline.data() + ((s * T + t) * kmax_polar_shoreline + k) * 3);
  }
  bool cameraValid(int s, int t, int k) const {
    return camera_valid[(s * T + t) * kmax_camera + k] != 0;
  }
  Eigen::Map<const Eigen::Matrix<double, 2, 1>> cameraReading(int s, int t, int k) const {
    return Eigen::Map<const Eigen::Matrix<double, 2, 1>>(
        camera.data() + ((s * T + t) * kmax_camera + k) * 2);
  }
};

// Throws std::runtime_error on schema mismatch or IO errors.
SimMeta loadSimMeta(const std::string& json_path);
SimData3D loadSimData(const std::string& npz_path, const SimMeta& meta);

}  // namespace parnav
