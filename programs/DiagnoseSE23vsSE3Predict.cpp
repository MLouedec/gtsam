/**
 * @file DiagnoseSE23vsSE3Predict.cpp
 * @brief Lock-step comparison of legacy CombinedImu preintegration vs the
 *        SE_2(3) variant, feeding both the *same* IMU sequence and predicting
 *        from a fixed initial state with zero bias.
 *
 * Used to localise the SE23 attitude jump observed around t≈43 s in the
 * aiding=none simulator runs. With bias-free, noise-free IMU the two
 * preintegrators must produce identical means; the first per-step index at
 * which ||Logmap(R_legacy^T * R_se23)|| exceeds a small threshold identifies
 * the step that introduces the divergence. Everything after that is downstream.
 *
 * Output: CSV with per-update-step diffs (t, rotation geodesic, position,
 * velocity, preintegrated rotation magnitudes, IMU rate magnitude) plus a
 * stdout print of the first divergence crossing.
 */

#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor2.h>
#include <gtsam/navigation/CombinedImuFactorGM.h>
#include <gtsam/navigation/PreintegrationCombinedParamsT.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ManifoldPreintegrationSE23.h>
#include <gtsam/navigation/NavState.h>

#include <Eigen/Core>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal CSV reader — same column layout as SimulationFixedLagSmoother.cpp.
// ---------------------------------------------------------------------------

struct SimData {
  uint64_t N{};
  Eigen::MatrixXd data;

  uint16_t freq() const { return static_cast<uint16_t>(data(0, 30)); }
  Eigen::MatrixXd imu_f() const { return data.block(0, 11, N, 3); }
  Eigen::MatrixXd imu_w() const { return data.block(0, 14, N, 3); }
  Eigen::MatrixXd true_pos() const { return data.block(0, 1, N, 3).transpose(); }
  Eigen::MatrixXd true_vel() const { return data.block(0, 4, N, 3).transpose(); }
  Eigen::MatrixXd true_att() const { return data.block(0, 7, N, 4).transpose(); }
};

static std::optional<SimData> read_csv(const std::string& filename) {
  std::ifstream file(filename);
  if (!file) {
    std::cerr << "Error: cannot open " << filename << "\n";
    return std::nullopt;
  }
  std::string line;
  std::vector<double> values;
  int rows = 0;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) values.push_back(std::stod(cell));
    ++rows;
  }
  SimData sd;
  sd.N = rows;
  sd.data = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic,
                                           Eigen::Dynamic, Eigen::RowMajor>>(
      values.data(), rows, values.size() / rows);
  return sd;
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  // Hardcoded to match SimulationFixedLagSmoother.cpp's default_input_file.
  // If you change the path there, change it here too.
  const std::string input =
      "/Users/ghms/ws/ntnu/parnav_ins_sim/data/"
      "otter_simulation_data_01_100Hz_noisy_biased_aided_at_10Hz_cpp.csv";
  std::string output = "/tmp/se23_vs_se3_predict.csv";
  double duration = 0.0;           // 0 = full data
  double flag_threshold = 1e-6;    // rad

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const std::string& f) {
      if (i + 1 >= argc) {
        std::cerr << f << " requires a value\n";
        std::exit(1);
      }
    };
    if (a == "--output") { need(a); output = argv[++i]; }
    else if (a == "--duration") { need(a); duration = std::stod(argv[++i]); }
    else if (a == "--threshold") { need(a); flag_threshold = std::stod(argv[++i]); }
    else if (a == "-h" || a == "--help") {
      std::cout
          << "Usage: " << argv[0] << " [options]\n"
          << "  (input path is hardcoded; matches SimulationFixedLagSmoother)\n"
          << "  --output <path>       diff CSV (default /tmp/se23_vs_se3_predict.csv)\n"
          << "  --duration <sec>      cap run length (0 = full)\n"
          << "  --threshold <rad>     first-crossing flag threshold (default 1e-6)\n";
      return 0;
    }
    else { std::cerr << "Unknown arg: " << a << "\n"; return 1; }
  }

  auto maybe_sd = read_csv(input);
  if (!maybe_sd) return 1;
  const SimData& sd = *maybe_sd;

  const double dt = 1.0 / sd.freq();
  const uint64_t max_idx =
      (duration > 0.0)
          ? std::min<uint64_t>(
                sd.N, static_cast<uint64_t>(duration / dt) + 1)
          : sd.N;

  printf("Loaded %llu rows @ %u Hz; running %.2f s\n",
         static_cast<unsigned long long>(sd.N), sd.freq(),
         (max_idx - 1) * dt);

  // --- Preintegration params (CombinedImuFactor flavour).
  // Mean propagation ignores covariance, but the params must be valid.
  using BIAS = gtsam::imuBias::ConstantBias;
  auto p = gtsam::PreintegrationCombinedParamsT<BIAS>::MakeSharedD(9.81);
  p->accelerometerCovariance = gtsam::I_3x3 * 1e-8;
  p->gyroscopeCovariance     = gtsam::I_3x3 * 1e-8;
  p->integrationCovariance   = gtsam::I_3x3 * 1e-10;
  p->biasAccCovariance       = gtsam::I_3x3 * 1e-12;
  p->biasOmegaCovariance     = gtsam::I_3x3 * 1e-12;

  BIAS zero_bias;  // identity (zero acc, zero gyro bias)

  using PIMLegacy =
      gtsam::PreintegratedCombinedMeasurementsGMT<
          gtsam::ManifoldPreintegrationT<BIAS>, BIAS>;
  using PIMSE23 =
      gtsam::PreintegratedCombinedMeasurements2T<
          gtsam::ManifoldPreintegrationSE23<BIAS>, BIAS>;

  PIMLegacy pim_legacy(p, zero_bias);
  PIMSE23   pim_se23(p, zero_bias);

  // --- Initial state from ground truth at row 0.
  Eigen::MatrixXd tA = sd.true_att();
  Eigen::MatrixXd tP = sd.true_pos();
  Eigen::MatrixXd tV = sd.true_vel();
  gtsam::Rot3 R0 = gtsam::Rot3::Quaternion(
      tA(0, 0), tA(1, 0), tA(2, 0), tA(3, 0));  // (qx, qy, qz, qw)
  gtsam::Point3 p0(tP(0, 0), tP(1, 0), tP(2, 0));
  gtsam::Vector3 v0(tV(0, 0), tV(1, 0), tV(2, 0));

  gtsam::NavState       state0_nav(gtsam::Pose3(R0, p0), v0);
  gtsam::Se23  state0_ext(R0, v0, p0);

  // --- Output CSV ---
  std::ofstream out(output);
  if (!out) { std::cerr << "cannot open " << output << "\n"; return 1; }
  out << "t,"
         "dR_geodesic_rad,dR_x,dR_y,dR_z,"
         "dp_norm,dp_x,dp_y,dp_z,"
         "dv_norm,dv_x,dv_y,dv_z,"
         "deltaR_legacy_rad,deltaR_se23_rad,"
         "w_norm,f_norm\n";

  Eigen::MatrixXd imu_f = sd.imu_f();
  Eigen::MatrixXd imu_w = sd.imu_w();

  double first_flagged = -1.0;
  double worst_dR = 0.0;
  double worst_dR_t = 0.0;

  for (uint64_t idx = 1; idx < max_idx; ++idx) {
    Eigen::Vector3d f = imu_f.row(idx).transpose();
    Eigen::Vector3d w = imu_w.row(idx).transpose();
    pim_legacy.integrateMeasurement(f, w, dt);
    pim_se23.integrateMeasurement(f, w, dt);

    const bool is_update_step = ((idx + 1) % 10 == 0);
    if (!is_update_step) continue;

    double t = idx * dt;
    gtsam::NavState      ns  = pim_legacy.predict(state0_nav, zero_bias);
    gtsam::Se23 ext = pim_se23.predict(state0_ext, zero_bias);

    gtsam::Rot3 R_L = ns.pose().rotation();
    gtsam::Rot3 R_S = ext.rotation();
    Eigen::Vector3d p_L = ns.pose().translation();
    Eigen::Vector3d p_S = ext.x(1);
    Eigen::Vector3d v_L = ns.v();
    Eigen::Vector3d v_S = ext.x(0);

    Eigen::Vector3d dR_vec = gtsam::Rot3::Logmap(R_L.inverse() * R_S);
    double dR_norm = dR_vec.norm();
    Eigen::Vector3d dp_vec = p_S - p_L;
    Eigen::Vector3d dv_vec = v_S - v_L;

    double deltaRL = gtsam::Rot3::Logmap(pim_legacy.deltaRij()).norm();
    double deltaRS = gtsam::Rot3::Logmap(pim_se23.deltaRij()).norm();

    out << t << ","
        << dR_norm << "," << dR_vec.x() << "," << dR_vec.y() << "," << dR_vec.z() << ","
        << dp_vec.norm() << "," << dp_vec.x() << "," << dp_vec.y() << "," << dp_vec.z() << ","
        << dv_vec.norm() << "," << dv_vec.x() << "," << dv_vec.y() << "," << dv_vec.z() << ","
        << deltaRL << "," << deltaRS << ","
        << w.norm() << "," << f.norm() << "\n";

    if (dR_norm > worst_dR) { worst_dR = dR_norm; worst_dR_t = t; }
    if (first_flagged < 0.0 && dR_norm > flag_threshold) {
      first_flagged = t;
      printf(
          ">>> First crossing |dR|>%.1e at t=%.3f s: "
          "|dR|=%.3e rad, |dp|=%.3e m, |dv|=%.3e m/s\n",
          flag_threshold, t, dR_norm, dp_vec.norm(), dv_vec.norm());
    }
  }

  printf("Worst |dR|=%.3e rad at t=%.3f s\n", worst_dR, worst_dR_t);
  if (first_flagged < 0.0)
    printf("No crossing of %.1e rad in the observed window.\n", flag_threshold);
  printf("Per-step diffs written to %s\n", output.c_str());
  return 0;
}
