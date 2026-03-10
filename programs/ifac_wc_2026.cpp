/**
 * @file ifac_wc_2026.cpp
 * @brief Fixed-lag smoother port of the multirotor PARS/GNSS/compass/baro
 *        experiment (IFAC WC 2026 paper, originally `multirotor_test.cpp`).
 *
 * Runs a fixed-lag smoother on a real multirotor CSV log (IMU, GNSS, PARS,
 * compass/baseline, barometer). Compile-time macros in the original program
 * have been promoted to runtime CLI flags:
 *
 *   --preint {se3|se23}         (default: se3)
 *   --bias {cb|gm}              (default: cb)
 *   --handover {none|angle|angle-baro|angle-range}
 *                               (default: none -> GNSS for the whole run)
 *   --robust {none|gm|tukey}    (default: none)
 *
 * Default configuration is SE3 + CB.
 */

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Unit3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/AttitudeFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor2.h>
#include <gtsam/navigation/CombinedImuFactorGM.h>
#include <gtsam/navigation/PreintegrationCombinedParamsT.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ManifoldPreintegrationSE23.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>

#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "AzimuthFactor.hpp"
#include "BaroFactorKnownPoint.hpp"
#include "CompassFactor.hpp"
#include "ElevationFactor.hpp"
#include "ExtendedPoseAttitudeFactor.hpp"
#include "GPSFactorSE23.hpp"
#include "RangeFactor.hpp"
#include "utils.hpp"

using parnav::deg2rad;
using parnav::rad2deg;
using parnav::ssa;

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

// ============================================================================
// CLI / configuration
// ============================================================================

enum class Handover { None, Angle, AngleBaro, AngleRange };
enum class Robust { None, GemanMcClure, Tukey };

struct Options {
  bool use_se23 = false;
  bool use_gauss_markov = false;
  Handover handover = Handover::None;
  Robust robust = Robust::None;
  double robust_threshold = 0.0;  // auto-filled from scheme default if 0

  std::string input_file =
      "/Users/ghms/ws/ntnu/parnav/parnav-scripts/post_processing/"
      "df_flat_data.csv";
  std::string output_dir =
      "/Users/ghms/ws/ntnu/parnav/parnav-scripts/post_processing/";
  std::string output_prefix = "ifac_wc_2026_";

  // Debug logging: per-update CSV with innovations, predicted & smoothed yaw,
  // bias estimates, and rover/MB flags. Off by default.
  bool debug_log = false;
  std::string debug_log_file;  // auto-filled if empty

  // Tuning knobs exposed for the zigzag sweep driver.
  double attitude_sigma = 0.05;  // rad, per-axis 2D Unit3 sigma
  int compass_lag_ticks = 0;     // shift z_gnss_comp by N IMU ticks (signed)
  double noise_scaling = 1;      // accel ARW inflation factor (legacy default)
  double bias_scaling = 1;       // gyro ARW + both bias-walk inflation
  // 1-pole IIR LPF on raw IMU before integrateMeasurement. <=0 disables.
  double gyro_lpf_hz = 0.0;
  double accel_lpf_hz = 0.0;
  // Rz(theta) applied to baseline_body to correct rover-antenna yaw miscal.
  double baseline_yaw_offset_deg = 0.0;
  // MSL altitude [m] of the NED origin, subtracted inside BaroFactor so that
  // the residual stays bounded. Replace with the surveyed/known field height
  // for each campaign (legacy default 271.7171 was specific to a prior site).
  double baro_origin_msl = 271.8671 - 0.15;
  // Local sea-level reference pressure [kPa]. The 101.29 standard only holds
  // on a standard day; on any other day this must be calibrated to the local
  // QFF/QNH. Negative means "auto-calibrate from the static window using
  // baro_origin_msl as the truth altitude" (default).
  double baro_p0_kpa = -1.0;
  // Measurement-noise sigma [m] on the BaroFactor.
  double baro_sigma = 3.0;
  // Ground-level air temperature [C], used to set the sea-level reference
  // T0 that drives dh/dp in the hypsometric formula. Standard day is 15;
  // cold days compress dh/dp linearly with T0.
  double ground_temp_c = 15.0;
  // Override the handover switch time (seconds). <0 means use the default
  // baked into handover_cfg() for the chosen --handover.
  double switch_time = -1.0;
};

namespace {
void print_usage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " [options]\n"
      << "  --preint {se3|se23}         state parameterisation (default: se3)\n"
      << "  --bias {cb|gm}              bias model (default: cb)\n"
      << "  --handover {none|angle|angle-baro|angle-range}\n"
      << "                               aiding handover policy (default: "
         "none)\n"
      << "  --robust {none|gm|tukey}    robust kernel on PARS factors\n"
      << "  --robust-threshold <v>      kernel threshold (default 1.0 for gm, "
         "4.6851 for tukey)\n"
      << "  --input <path>\n"
      << "  --output-dir <path>\n"
      << "  --output-prefix <str>\n"
      << "  --debug-log [path]          emit per-update debug CSV\n"
      << "                              (default path: "
         "<output-dir>/<prefix>debug.csv)\n"
      << "  --attitude-sigma <v>        Unit3 attitude factor sigma [rad] "
         "(default 0.05)\n"
      << "  --compass-lag-ticks <n>     shift z_gnss_comp index by n IMU ticks "
         "(signed)\n"
      << "  --noise-scaling <s>         accel ARW inflation (default 10)\n"
      << "  --bias-scaling <s>          gyro ARW + bias walk inflation "
         "(default 50)\n"
      << "  --gyro-lpf-hz <f>           1-pole IIR cutoff for gyro [Hz], 0 "
         "disables\n"
      << "  --accel-lpf-hz <f>          1-pole IIR cutoff for accel [Hz], 0 "
         "disables\n"
      << "  --baseline-yaw-offset-deg <d> Rz rotation of baseline_body to "
         "correct\n"
      << "                              rover-antenna yaw miscalibration "
         "(default 0)\n"
      << "  --baro-origin-msl <m>       MSL altitude [m] of NED origin used "
         "inside\n"
      << "                              BaroFactor (default 271.7171, set to "
         "field height)\n"
      << "  --baro-p0-kpa <v>           local sea-level reference pressure "
         "[kPa]\n"
      << "                              (default: auto-calibrate from static "
         "window\n"
      << "                              against --baro-origin-msl)\n"
      << "  --baro-sigma <m>            BaroFactor measurement sigma "
         "(default 1.0)\n"
      << "  --ground-temp-c <C>         ground-level air temperature [C];\n"
      << "                              sets T0_K (default 15 = std day)\n"
      << "  --switch-time <s>           override handover switch time "
         "[seconds]\n"
      << "  -h, --help\n";
}

bool parse_args(int argc, char** argv, Options& o) {
  auto need = [&](int i, const char* f) {
    if (i + 1 >= argc) {
      std::cerr << "Error: " << f << " needs a value\n";
      return false;
    }
    return true;
  };
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (a == "--preint") {
      if (!need(i, "--preint")) return false;
      std::string v = argv[++i];
      if (v == "se3")
        o.use_se23 = false;
      else if (v == "se23")
        o.use_se23 = true;
      else {
        std::cerr << "Unknown --preint " << v << "\n";
        return false;
      }
    } else if (a == "--bias") {
      if (!need(i, "--bias")) return false;
      std::string v = argv[++i];
      if (v == "cb")
        o.use_gauss_markov = false;
      else if (v == "gm")
        o.use_gauss_markov = true;
      else {
        std::cerr << "Unknown --bias " << v << "\n";
        return false;
      }
    } else if (a == "--handover") {
      if (!need(i, "--handover")) return false;
      std::string v = argv[++i];
      if (v == "none")
        o.handover = Handover::None;
      else if (v == "angle")
        o.handover = Handover::Angle;
      else if (v == "angle-baro")
        o.handover = Handover::AngleBaro;
      else if (v == "angle-range")
        o.handover = Handover::AngleRange;
      else {
        std::cerr << "Unknown --handover " << v << "\n";
        return false;
      }
    } else if (a == "--robust") {
      if (!need(i, "--robust")) return false;
      std::string v = argv[++i];
      if (v == "none")
        o.robust = Robust::None;
      else if (v == "gm")
        o.robust = Robust::GemanMcClure;
      else if (v == "tukey")
        o.robust = Robust::Tukey;
      else {
        std::cerr << "Unknown --robust " << v << "\n";
        return false;
      }
    } else if (a == "--robust-threshold") {
      if (!need(i, "--robust-threshold")) return false;
      o.robust_threshold = std::stod(argv[++i]);
    } else if (a == "--input") {
      if (!need(i, "--input")) return false;
      o.input_file = argv[++i];
    } else if (a == "--output-dir") {
      if (!need(i, "--output-dir")) return false;
      o.output_dir = argv[++i];
      if (!o.output_dir.empty() && o.output_dir.back() != '/')
        o.output_dir.push_back('/');
    } else if (a == "--output-prefix") {
      if (!need(i, "--output-prefix")) return false;
      o.output_prefix = argv[++i];
    } else if (a == "--debug-log") {
      o.debug_log = true;
      // Optional path argument: accept it only if the next token isn't a flag.
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        o.debug_log_file = argv[++i];
      }
    } else if (a == "--attitude-sigma") {
      if (!need(i, "--attitude-sigma")) return false;
      o.attitude_sigma = std::stod(argv[++i]);
    } else if (a == "--compass-lag-ticks") {
      if (!need(i, "--compass-lag-ticks")) return false;
      o.compass_lag_ticks = std::stoi(argv[++i]);
    } else if (a == "--noise-scaling") {
      if (!need(i, "--noise-scaling")) return false;
      o.noise_scaling = std::stod(argv[++i]);
    } else if (a == "--bias-scaling") {
      if (!need(i, "--bias-scaling")) return false;
      o.bias_scaling = std::stod(argv[++i]);
    } else if (a == "--gyro-lpf-hz") {
      if (!need(i, "--gyro-lpf-hz")) return false;
      o.gyro_lpf_hz = std::stod(argv[++i]);
    } else if (a == "--accel-lpf-hz") {
      if (!need(i, "--accel-lpf-hz")) return false;
      o.accel_lpf_hz = std::stod(argv[++i]);
    } else if (a == "--baseline-yaw-offset-deg") {
      if (!need(i, "--baseline-yaw-offset-deg")) return false;
      o.baseline_yaw_offset_deg = std::stod(argv[++i]);
    } else if (a == "--baro-origin-msl") {
      if (!need(i, "--baro-origin-msl")) return false;
      o.baro_origin_msl = std::stod(argv[++i]);
    } else if (a == "--baro-p0-kpa") {
      if (!need(i, "--baro-p0-kpa")) return false;
      o.baro_p0_kpa = std::stod(argv[++i]);
    } else if (a == "--baro-sigma") {
      if (!need(i, "--baro-sigma")) return false;
      o.baro_sigma = std::stod(argv[++i]);
    } else if (a == "--ground-temp-c") {
      if (!need(i, "--ground-temp-c")) return false;
      o.ground_temp_c = std::stod(argv[++i]);
    } else if (a == "--switch-time") {
      if (!need(i, "--switch-time")) return false;
      o.switch_time = std::stod(argv[++i]);
    } else {
      std::cerr << "Unknown arg: " << a << "\n";
      print_usage(argv[0]);
      return false;
    }
  }
  if (o.robust_threshold == 0.0) {
    o.robust_threshold = (o.robust == Robust::Tukey) ? 4.6851 : 1.0;
  }
  return true;
}
}  // namespace

// ============================================================================
// CSV parsing (port of parse_multirotor_csv)
// ============================================================================

struct MultirotorData {
  std::vector<double> t;
  std::vector<Eigen::Vector3d> f_m;
  std::vector<Eigen::Vector3d> w_m;
  std::vector<Eigen::Vector3d> z_bt;  // [azi, ele, range]
  std::vector<Eigen::Vector3d> z_bt_ned;
  std::vector<int> bt_meas_idx;
  std::vector<Eigen::Vector3d> z_gnss_ned;
  std::vector<int> gnss_meas_idx;  // legacy (== MB) -- kept for compat
  std::vector<double> yaw;
  std::vector<int> yaw_idx;
  std::vector<Eigen::Vector3d> z_gnss_comp;  // baseline vector in nav frame
  std::vector<double> z_baro;                // pressure [kPa]
  std::vector<int> baro_idx;
  std::vector<double> z_gnss_range;
  // IFAC WC 2026: separate MB (position) and rover (baseline/attitude)
  // measurement indices. Empty when reading a legacy 26-column CSV.
  std::vector<int> gnss_mb_meas_idx;
  std::vector<int> gnss_rover_meas_idx;
};

std::optional<MultirotorData> parse_multirotor_csv(
    const std::string& filename) {
  std::ifstream file(filename);
  if (!file) {
    std::cerr << "Error: Could not open " << filename << "\n";
    return std::nullopt;
  }
  MultirotorData d;
  std::string line;
  std::getline(file, line);  // header
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string cell;
    std::vector<std::string> tok;
    while (std::getline(ss, cell, ',')) tok.push_back(cell);
    // 26: legacy, 28: IFAC WC 2026 (adds gnss_mb_meas_idx, gnss_rover_meas_idx)
    if (tok.size() != 26 && tok.size() != 28) {
      std::cerr << "Unexpected column count: " << tok.size()
                << " (expected 26 or 28)\n";
      return std::nullopt;
    }
    const bool has_split_idx = (tok.size() == 28);
    size_t i = 0;
    d.t.push_back(std::stod(tok[i++]));
    d.f_m.emplace_back(std::stod(tok[i]), std::stod(tok[i + 1]),
                       std::stod(tok[i + 2]));
    i += 3;
    d.w_m.emplace_back(std::stod(tok[i]), std::stod(tok[i + 1]),
                       std::stod(tok[i + 2]));
    i += 3;
    d.z_bt.emplace_back(std::stod(tok[i]), std::stod(tok[i + 1]),
                        std::stod(tok[i + 2]));
    i += 3;
    d.z_bt_ned.emplace_back(std::stod(tok[i]), std::stod(tok[i + 1]),
                            std::stod(tok[i + 2]));
    i += 3;
    d.bt_meas_idx.push_back(std::stoi(tok[i++]));
    d.z_gnss_ned.emplace_back(std::stod(tok[i]), std::stod(tok[i + 1]),
                              std::stod(tok[i + 2]));
    i += 3;
    d.gnss_meas_idx.push_back(std::stoi(tok[i++]));
    d.yaw.push_back(std::stod(tok[i++]));
    d.yaw_idx.push_back(std::stoi(tok[i++]));
    d.z_gnss_comp.emplace_back(std::stod(tok[i]), std::stod(tok[i + 1]),
                               std::stod(tok[i + 2]));
    i += 3;
    d.z_baro.push_back(std::stod(tok[i++]));
    d.baro_idx.push_back(std::stoi(tok[i++]));
    d.z_gnss_range.push_back(std::stod(tok[i++]));
    if (has_split_idx) {
      d.gnss_mb_meas_idx.push_back(std::stoi(tok[i++]));
      d.gnss_rover_meas_idx.push_back(std::stoi(tok[i++]));
    } else {
      // Legacy CSV: both attitude and position shared gnss_meas_idx.
      d.gnss_mb_meas_idx.push_back(d.gnss_meas_idx.back());
      d.gnss_rover_meas_idx.push_back(d.gnss_meas_idx.back());
    }
  }

  return d;
}

// ============================================================================
// Helpers
// ============================================================================

namespace {

gtsam::SharedNoiseModel wrap_robust(const gtsam::SharedNoiseModel& base,
                                    Robust r, double k) {
  switch (r) {
    case Robust::None:
      return base;
    case Robust::GemanMcClure:
      return gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::GemanMcClure::Create(k), base);
    case Robust::Tukey:
      return gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Tukey::Create(k), base);
  }
  return base;
}

struct HandoverCfg {
  double switch_time;
  bool use_baro;
  bool use_range;
};

HandoverCfg handover_cfg(Handover h) {
  switch (h) {
    case Handover::None:
      return {1e9, false, false};
    case Handover::Angle:
      return {200.0, false, false};
    case Handover::AngleBaro:
      return {200.0, true, false};
    case Handover::AngleRange:
      return {200.0, false, true};
  }
  return {1e9, false, false};
}

void write_vec3(const std::string& path,
                const std::vector<Eigen::Vector3d>& v) {
  std::ofstream f(path);
  f << "x,y,z\n";
  for (const auto& x : v) f << x.x() << "," << x.y() << "," << x.z() << "\n";
}
void write_scalar(const std::string& path, const std::vector<double>& v) {
  std::ofstream f(path);
  f << "t\n";
  for (double x : v) f << x << "\n";
}

}  // namespace

// ============================================================================
// Estimation loop
// ============================================================================

template <class BIAS, bool UseSE23>
void run_estimation(const MultirotorData& d, const Options& opts) {
  using PIMLegacy = gtsam::PreintegratedCombinedMeasurementsGMT<
      gtsam::ManifoldPreintegrationT<BIAS>, BIAS>;
  using FactorLegacy = gtsam::CombinedImuFactorGMT<PIMLegacy, BIAS>;
  using PIMSE23 = gtsam::PreintegratedCombinedMeasurements2T<
      gtsam::ManifoldPreintegrationSE23<BIAS>, BIAS>;
  using FactorSE23 = gtsam::CombinedImuFactor2T<PIMSE23, BIAS>;
  using PIM = std::conditional_t<UseSE23, PIMSE23, PIMLegacy>;
  using ImuFactor = std::conditional_t<UseSE23, FactorSE23, FactorLegacy>;
  using PoseParam =
      std::conditional_t<UseSE23, gtsam::Se23, gtsam::Pose3>;

  auto hcfg = handover_cfg(opts.handover);
  if (opts.switch_time >= 0.0) hcfg.switch_time = opts.switch_time;
  const bool use_baro = hcfg.use_baro;
  const bool use_range = hcfg.use_range;

  // --- Sensor parameters (from original program) ---
  const double g0 = 9.80665;
  const double vrw = 0.07;
  const double arw = 0.15;
  const double bias_instability_acc = 0.05;  // milli g
  const double bias_instability_ars = 0.5;   // deg/hour
  const double T_acc = 3600.0;
  const double T_ars = 3600.0;
  const double noise_scaling = opts.noise_scaling;
  const double bias_scaling = opts.bias_scaling;

  const double q_v = std::pow(noise_scaling * vrw / 60.0, 2.0);
  const double q_o = std::pow((noise_scaling * arw / 60.0) * deg2rad(1.0), 2.0);
  const double q_b_v =
      (2.0 / T_acc) *
      std::pow(bias_scaling * bias_instability_acc * (g0 / 1000.0), 2.0);
  const double q_b_o =
      (2.0 / T_ars) *
      std::pow((bias_scaling * bias_instability_ars / 3600.0) * deg2rad(1.0),
               2.0);
  const double q_p = 1e-3;

  // --- Preintegration params ---
  auto p = gtsam::PreintegrationCombinedParamsT<BIAS>::MakeSharedD(g0);
  p->accelerometerCovariance = gtsam::I_3x3 * q_v;
  p->gyroscopeCovariance = gtsam::I_3x3 * q_o;
  p->integrationCovariance = gtsam::I_3x3 * q_p;
  p->biasAccCovariance = gtsam::I_3x3 * q_b_v;
  p->biasOmegaCovariance = gtsam::I_3x3 * q_b_o;

  // --- Measurement noise models ---
  gtsam::Matrix3 R_GNSS_pos = gtsam::I_3x3;
  R_GNSS_pos(0, 0) = std::pow(1.5, 2.0);
  R_GNSS_pos(1, 1) = std::pow(1.5, 2.0);
  R_GNSS_pos(2, 2) = std::pow(3.0, 2.0);
  auto gnss_noise =
      gtsam::noiseModel::Diagonal::Variances(R_GNSS_pos.diagonal());

  // Attitude factor noise: 2D Unit3 residual, σ exposed via --attitude-sigma
  // (default 0.05 rad per axis, matching original SE3 path).
  const double att_s = opts.attitude_sigma;
  auto attitude_noise = gtsam::noiseModel::Diagonal::Variances(
      gtsam::Vector2(att_s * att_s, att_s * att_s));

  auto yaw_noise = gtsam::noiseModel::Isotropic::Sigma(1, 7.0);

  auto baro_noise = gtsam::noiseModel::Isotropic::Sigma(1, opts.baro_sigma);

  // PARS noise models (1-d each; robust kernel applied if requested)
  auto pars_azi_base = gtsam::noiseModel::Isotropic::Sigma(1, deg2rad(5.0));
  auto pars_ele_base = gtsam::noiseModel::Isotropic::Sigma(1, deg2rad(5.0));
  auto pars_range_base = gtsam::noiseModel::Isotropic::Sigma(1, 2.5);
  auto pars_azi_noise =
      wrap_robust(pars_azi_base, opts.robust, opts.robust_threshold);
  auto pars_ele_noise =
      wrap_robust(pars_ele_base, opts.robust, opts.robust_threshold);
  auto pars_range_noise =
      wrap_robust(pars_range_base, opts.robust, opts.robust_threshold);

  // --- PARS geometry (fixed constants from original program) ---
  //  Eigen::Matrix3d R_rn;
  //  R_rn << -0.0369, 0.1402, -0.9894, -0.0456, -0.9893, -0.1384, -0.9983,
  //  0.0401,
  //      0.0429;
  //  Eigen::Vector3d pars_origin(-0.0519, -0.0820, 0.0492);
  Eigen::Matrix3d R_rn;
  R_rn << 0.0297, -0.0824, -0.9962, -0.0379, -0.9960, 0.0813, -0.9988, 0.0353,
      -0.0327;
  Eigen::Vector3d pars_origin(0.0601, 0.1740, 0.0436);

  // --- Compass / baseline geometry ---
  const gtsam::Point3 p_imu_rover_b(0.153, -0.019, -0.302);
  const gtsam::Point3 p_imu_bm_b(-0.310, 0.156, -0.300);
  gtsam::Point3 baseline_body = p_imu_rover_b - p_imu_bm_b;
  baseline_body *= -1;
  if (opts.baseline_yaw_offset_deg != 0.0) {
    const double th = opts.baseline_yaw_offset_deg * M_PI / 180.0;
    baseline_body = gtsam::Rot3::Rz(th).rotate(baseline_body);
    std::fprintf(stderr, "Applied baseline yaw offset: %.3f deg\n",
                 opts.baseline_yaw_offset_deg);
  }

  // --- Initial state (from original program) ---
  gtsam::Rot3 R0 =
      gtsam::Rot3::Quaternion(0.017, 0.0085, -0.0007, 0.9998).normalized();
  gtsam::Point3 p0 = gtsam::Point3::Zero();
  gtsam::Vector3 v0 = gtsam::Vector3::Zero();

  const double A_pos = 2.5;
  const double A_vel = 0.5;
  const double A_att = 0.15;
  // Per-axis prior overrides for yaw and altitude (match original P0 layout
  // in multirotor_test.cpp: P0(2,2)=0.25^2 and P0(5,5)=5^2).
  const double A_yaw = 0.25;
  const double A_pos_z = 5.0;
  const double A_acc_bias = (50.0 * g0 / 1000.0);
  const double A_gyro_bias = deg2rad(360.0 / 3600.0);
  // const double A_gyro_bias = deg2rad(360.0 / 3600.0);

  BIAS prior_bias;
  if constexpr (std::is_same_v<BIAS, gtsam::imuBias::GaussMarkovBias>) {
    prior_bias = gtsam::imuBias::GaussMarkovBias(
        gtsam::Vector3::Zero(), gtsam::Vector3::Zero(), T_acc, T_ars);
  }

  auto bias_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << A_acc_bias, A_acc_bias, A_acc_bias, A_gyro_bias,
       A_gyro_bias, A_gyro_bias)
          .finished());

  // --- Smoother setup ---
  gtsam::ISAM2Params isam_params;
  // GTSAM 4.3.0 default CHOLESKY throws Indeterminate on the combined IMU+bias
  // system; QR is numerically robust (see SimulationFixedLagSmoother).
  isam_params.factorization = gtsam::ISAM2Params::QR;
  isam_params.relinearizeSkip = 1;
  isam_params.relinearizeThreshold = 0.001;
  isam_params.findUnusedFactorSlots = true;
  const double smoother_lag = 5.0;
  gtsam::IncrementalFixedLagSmoother smoother(smoother_lag, isam_params);

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;
  gtsam::FixedLagSmoother::KeyTimestampMap timestamps;

  if constexpr (UseSE23) {
    auto ep_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(9) << A_att, A_att, A_yaw, A_vel, A_vel, A_vel, A_pos,
         A_pos, A_pos_z)
            .finished());
    gtsam::Se23 ext0(R0, v0, p0);
    graph.addPrior<gtsam::Se23>(X(0), ext0, ep_noise);
    values.insert(X(0), ext0);
    timestamps[X(0)] = 0.0;
  } else {
    auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << A_att, A_att, A_yaw, A_pos, A_pos, A_pos_z)
            .finished());
    auto vel_noise = gtsam::noiseModel::Isotropic::Sigma(3, A_vel);
    graph.addPrior<gtsam::Pose3>(X(0), gtsam::Pose3(R0, p0), pose_noise);
    graph.addPrior<gtsam::Vector3>(V(0), v0, vel_noise);
    values.insert(X(0), gtsam::Pose3(R0, p0));
    values.insert(V(0), v0);
    timestamps[X(0)] = 0.0;
    timestamps[V(0)] = 0.0;
  }
  graph.addPrior<BIAS>(B(0), prior_bias, bias_noise);
  values.insert(B(0), prior_bias);
  timestamps[B(0)] = 0.0;
  double baro_p0 = (opts.baro_p0_kpa > 0.0) ? opts.baro_p0_kpa : 101.29;
  const double baro_T0_K =
      parnav::BaroFactorKnownPoint<PoseParam>::T0_from_ground_temp_c(
          opts.ground_temp_c, opts.baro_origin_msl);
  if (use_baro) {
    // Auto-calibrate p0 from the static window: average the first ~40 s of
    // valid baro samples and solve for the local sea-level reference such
    // that height_from_pressure(p_avg, p0, T0_K) == baro_origin_msl. No
    // live baro-bias state is created -- BaroFactorKnownPoint handles the
    // sensor bias as a fixed offset (here zero, since p0 is calibrated).
    // Use only the [20 s, 40 s) slice. The first ~20 s of the dune
    // pressure log carries a sub-sampled startup transient that biases
    // p_avg low by ~3 m of altitude; by 20 s the sensor has settled.
    // The window still ends before the 40 s baro fire gate so the
    // calibration and live samples remain non-overlapping.
    double p_sum = 0.0;
    int p_n = 0;
    for (size_t k = 0; k < d.t.size() && d.t[k] - d.t[0] < 40.0; ++k) {
      if (d.baro_idx[k] != 0 && d.t[k] - d.t[0] >= 20.0) {
        p_sum += d.z_baro[k];
        ++p_n;
      }
    }
    if (opts.baro_p0_kpa > 0.0) {
      printf("Baro init:   p0=%.3f kPa (manual), T0=%.2f K\n", baro_p0,
             baro_T0_K);
    } else if (p_n > 0) {
      const double p_avg = p_sum / p_n;
      baro_p0 = parnav::BaroFactorKnownPoint<PoseParam>::p0_from_known_altitude(
          p_avg, opts.baro_origin_msl, baro_T0_K);
      printf(
          "Baro init:   p_avg=%.3f kPa over %d samples, origin=%.2f m, "
          "T_ground=%.1f C -> T0=%.2f K, p0_local=%.3f kPa\n",
          p_avg, p_n, opts.baro_origin_msl, opts.ground_temp_c, baro_T0_K,
          baro_p0);
    } else {
      printf(
          "Baro init:   no static-window samples; p0=%.3f kPa, T0=%.2f K "
          "(standard)\n",
          baro_p0, baro_T0_K);
    }
  }
  smoother.update(graph, values, timestamps);

  // --- Predict loop ---
  auto preintegrated = std::make_shared<PIM>(p, prior_bias);

  using StateType =
      std::conditional_t<UseSE23, gtsam::Se23, gtsam::NavState>;
  StateType prev_state = [&] {
    if constexpr (UseSE23)
      return gtsam::Se23(R0, v0, p0);
    else
      return gtsam::NavState(gtsam::Pose3(R0, p0), v0);
  }();
  BIAS prev_bias = prior_bias;

  // Result storage
  std::vector<gtsam::Rot3> est_R{R0};
  std::vector<Eigen::Vector3d> est_p{p0};
  std::vector<Eigen::Vector3d> est_v{v0};
  std::vector<Eigen::Vector3d> est_ba{prior_bias.accelerometer()};
  std::vector<Eigen::Vector3d> est_bg{prior_bias.gyroscope()};
  std::vector<Eigen::Vector3d> pos_sigma{Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> vel_sigma{Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> att_sigma{Eigen::Vector3d::Zero()};

  double accumulated_time = 0.0;
  int correction_count = 0;
  const auto start = std::chrono::system_clock::now();
  const auto N = static_cast<int64_t>(d.t.size());

  // --- Debug log (per-update CSV for yaw-zigzag analysis) ---
  // Apply compass_lag_ticks by re-timing the entire rover column in place:
  // both the gating index and the baseline value shift together. A negative
  // lag moves the rover measurement earlier on the IMU time axis (pipeline
  // delay compensation). This avoids pulling from zero-padded rows that
  // would result from shifting only the value.
  std::vector<int> rover_idx_shifted = d.gnss_rover_meas_idx;
  std::vector<Eigen::Vector3d> z_gnss_comp_shifted = d.z_gnss_comp;
  if (opts.compass_lag_ticks != 0) {
    const int64_t lag = opts.compass_lag_ticks;
    std::fill(rover_idx_shifted.begin(), rover_idx_shifted.end(), 0);
    std::fill(z_gnss_comp_shifted.begin(), z_gnss_comp_shifted.end(),
              Eigen::Vector3d::Zero());
    // shifted[i] := original[i - lag] (guarded)
    for (int64_t i = 0; i < N; ++i) {
      const int64_t j = i - lag;
      if (j >= 0 && j < N) {
        rover_idx_shifted[i] = d.gnss_rover_meas_idx[j];
        z_gnss_comp_shifted[i] = d.z_gnss_comp[j];
      }
    }
    printf("Compass lag: %+lld IMU ticks applied\n",
           static_cast<long long>(lag));
  }

  std::ofstream debug_ofs;
  if (opts.debug_log) {
    std::string path =
        opts.debug_log_file.empty()
            ? (opts.output_dir + opts.output_prefix + "debug.csv")
            : opts.debug_log_file;
    debug_ofs.open(path);
    if (!debug_ofs) {
      std::cerr << "Warning: could not open debug log " << path << "\n";
    } else {
      debug_ofs
          << "t,correction_idx,idx,mb_tick,rover_tick,baro_tick,before_"
             "handover,"
          << "yaw_prop,yaw_post,pitch_prop,pitch_post,roll_prop,roll_post,"
          << "att_innov_u,att_innov_v,att_innov_norm,"
          << "z_gnss_comp_x,z_gnss_comp_y,z_gnss_comp_z,"
          << "nPred_x,nPred_y,nPred_z,"
          << "bg_x,bg_y,bg_z,ba_x,ba_y,ba_z,"
          << "baseline_body_x,baseline_body_y,baseline_body_z\n";
      printf("Debug log: %s\n", path.c_str());
    }
  }

  // 1-pole IIR LPF state for gyro/accel (initialised to first sample below).
  // y[n] = (1-alpha) * y[n-1] + alpha * x[n], with alpha = dt / (RC + dt),
  // RC = 1/(2*pi*fc). dt recomputed per sample (allows non-uniform t).
  Eigen::Vector3d gyro_lpf_state = d.w_m[0];
  Eigen::Vector3d accel_lpf_state = d.f_m[0];
  const bool use_gyro_lpf = opts.gyro_lpf_hz > 0.0;
  const bool use_accel_lpf = opts.accel_lpf_hz > 0.0;
  if (use_gyro_lpf || use_accel_lpf) {
    printf("IMU LPF: gyro=%.1f Hz, accel=%.1f Hz\n", opts.gyro_lpf_hz,
           opts.accel_lpf_hz);
  }

  for (int64_t idx = 1; idx < N; ++idx) {
    const double dt = d.t[idx] - d.t[idx - 1];
    accumulated_time += dt;

    Eigen::Vector3d w_in = d.w_m[idx];
    Eigen::Vector3d f_in = d.f_m[idx];
    if (use_gyro_lpf) {
      const double rc = 1.0 / (2.0 * M_PI * opts.gyro_lpf_hz);
      const double a = dt / (rc + dt);
      gyro_lpf_state = (1.0 - a) * gyro_lpf_state + a * w_in;
      w_in = gyro_lpf_state;
    }
    if (use_accel_lpf) {
      const double rc = 1.0 / (2.0 * M_PI * opts.accel_lpf_hz);
      const double a = dt / (rc + dt);
      accel_lpf_state = (1.0 - a) * accel_lpf_state + a * f_in;
      f_in = accel_lpf_state;
    }
    preintegrated->integrateMeasurement(f_in, w_in, dt);

    const bool before_handover = accumulated_time < hcfg.switch_time;
    const bool baro_tick =
        use_baro && d.baro_idx[idx] != 0 && accumulated_time >= 40.0;
    // IFAC WC 2026: split MB (position/RTK) and rover (baseline/attitude)
    // gates so the attitude factor only fires on IMU ticks where a rover
    // compass measurement is actually available. Previously both shared
    // gnss_meas_idx (== MB), which leaked zero-padded baselines into a
    // degenerate Unit3 attitude constraint and caused a yaw zigzag.
    const bool mb_tick = d.gnss_mb_meas_idx[idx] != 0;
    const bool rover_tick = rover_idx_shifted[idx] != 0;
    const bool pre_tick =
        before_handover && (mb_tick || rover_tick || baro_tick);
    const bool post_tick =
        !before_handover && (d.bt_meas_idx[idx] != 0 || baro_tick);

    // --- Log the preintegration-predicted state every IMU tick ---
    // This keeps the output arrays aligned with the IMU time vector. At
    // update ticks we overwrite the last-pushed entry with the smoothed
    // estimate further down.
    {
      gtsam::Rot3 R_pred;
      Eigen::Vector3d p_pred, v_pred;
      if constexpr (UseSE23) {
        gtsam::Se23 prop =
            preintegrated->predict(prev_state, prev_bias);
        R_pred = prop.rotation();
        p_pred = prop.x(1);
        v_pred = prop.x(0);
      } else {
        gtsam::NavState prop = preintegrated->predict(prev_state, prev_bias);
        R_pred = prop.pose().rotation();
        p_pred = prop.pose().translation();
        v_pred = prop.v();
      }
      est_R.push_back(R_pred);
      est_p.push_back(p_pred);
      est_v.push_back(v_pred);
      est_ba.push_back(prev_bias.accelerometer());
      est_bg.push_back(prev_bias.gyroscope());
      // Between updates we don't recompute marginals; carry the last-known
      // sigma forward (approx OK for plotting — the update overwrites it).
      pos_sigma.push_back(pos_sigma.back());
      vel_sigma.push_back(vel_sigma.back());
      att_sigma.push_back(att_sigma.back());
    }

    if (!pre_tick && !post_tick) continue;

    correction_count++;
    const double t_now = d.t[idx];

    graph.resize(0);
    values.clear();
    timestamps.clear();

    // --- IMU factor ---
    if constexpr (UseSE23) {
      graph.add(ImuFactor(X(correction_count - 1), X(correction_count),
                          B(correction_count - 1), B(correction_count),
                          *preintegrated));
    } else {
      graph.add(ImuFactor(X(correction_count - 1), V(correction_count - 1),
                          X(correction_count), V(correction_count),
                          B(correction_count - 1), B(correction_count),
                          *preintegrated));
    }

    // --- Predict and insert initial values ---
    gtsam::Rot3 R_prop;  // kept for debug log
    if constexpr (UseSE23) {
      gtsam::Se23 prop = preintegrated->predict(prev_state, prev_bias);
      R_prop = prop.rotation();
      values.insert(X(correction_count), prop);
      values.insert(B(correction_count), prev_bias);
      timestamps[X(correction_count)] = t_now;
      timestamps[B(correction_count)] = t_now;
    } else {
      gtsam::NavState prop = preintegrated->predict(prev_state, prev_bias);
      R_prop = prop.pose().rotation();
      values.insert(X(correction_count), prop.pose());
      values.insert(V(correction_count), prop.v());
      values.insert(B(correction_count), prev_bias);
      timestamps[X(correction_count)] = t_now;
      timestamps[V(correction_count)] = t_now;
      timestamps[B(correction_count)] = t_now;
    }

    // --- Aiding factors ---
    if (pre_tick) {
      // GPS position factor -- gated on the MB (RTK) index only.
      if (mb_tick) {
        if constexpr (UseSE23) {
          graph.add(parnav::GPSFactorSE23(X(correction_count),
                                          d.z_gnss_ned[idx], gnss_noise));
        } else {
          graph.add(gtsam::GPSFactor(X(correction_count), d.z_gnss_ned[idx],
                                     gnss_noise));
        }
      }
      // Attitude factor: 2D Unit3 residual -- gated on the rover
      // (NAV_RELPOSNED baseline) index only. This prevents zero-padded
      // baselines on MB-only ticks from injecting a degenerate Unit3
      // constraint (previously caused a yaw zigzag).
      if (rover_tick) {
        const auto& z = z_gnss_comp_shifted[idx];
        if (z.norm() > 1e-3) {
          const gtsam::Unit3 nZ_meas(z_gnss_comp_shifted[idx]);
          const gtsam::Unit3 bRef_body(baseline_body);
          if constexpr (UseSE23) {
            graph.add(parnav::ExtendedPoseAttitudeFactor(
                X(correction_count), nZ_meas, attitude_noise, bRef_body));
          } else {
            graph.add(gtsam::AttitudeFactor<gtsam::Pose3>(X(correction_count), nZ_meas,
                                                 attitude_noise, bRef_body));
          }
        }
      }
      if (baro_tick) {
        graph.add(parnav::BaroFactorKnownPoint<PoseParam>(
            X(correction_count), d.z_baro[idx], /*p_bias=*/0.0, baro_noise,
            opts.baro_origin_msl, baro_p0, baro_T0_K));
      }
    } else if (post_tick) {
      if (d.bt_meas_idx[idx] != 0) {
        const double azi = d.z_bt[idx](0);
        const double ele = d.z_bt[idx](1);
        const double range = d.z_bt[idx](2);

        // std::cout << "Azi " << azi << "\n";
        // std::cout << "Ele " << ele << "\n";
        // std::cout << "Ran " << range << "\n";
        // std::cin.get();

        graph.add(parnav::AzimuthFactor<PoseParam>(
            X(correction_count), pars_azi_noise, azi, -pars_origin, R_rn));
        graph.add(parnav::ElevationFactor<PoseParam>(
            X(correction_count), pars_ele_noise, ele, -pars_origin, R_rn));
        if (use_range) {
          graph.add(parnav::RangeFactor<PoseParam>(X(correction_count),
                                                   pars_range_noise, range,
                                                   -pars_origin, R_rn));
        }
      }

      if (baro_tick) {
        graph.add(parnav::BaroFactorKnownPoint<PoseParam>(
            X(correction_count), d.z_baro[idx], /*p_bias=*/0.0, baro_noise,
            opts.baro_origin_msl, baro_p0, baro_T0_K));
      }
    }

    // --- Update smoother ---
    smoother.update(graph, values, timestamps);

    // --- Extract results ---
    gtsam::Values result = smoother.calculateEstimate();
    gtsam::Rot3 R_est;
    gtsam::Point3 p_est;
    gtsam::Vector3 v_est;
    if constexpr (UseSE23) {
      auto ext = result.at<gtsam::Se23>(X(correction_count));
      prev_state = ext;
      R_est = ext.rotation();
      p_est = ext.x(1);
      v_est = ext.x(0);
    } else {
      auto pose = result.at<gtsam::Pose3>(X(correction_count));
      auto vel = result.at<gtsam::Vector3>(V(correction_count));
      prev_state = gtsam::NavState(pose, vel);
      R_est = pose.rotation();
      p_est = pose.translation();
      v_est = vel;
    }
    prev_bias = result.at<BIAS>(B(correction_count));

    preintegrated->resetIntegrationAndSetBias(prev_bias);

    // Overwrite the last-logged (predicted) entry with the smoothed estimate.
    est_R.back() = R_est;
    est_p.back() = p_est;
    est_v.back() = v_est;
    est_ba.back() = prev_bias.accelerometer();
    est_bg.back() = prev_bias.gyroscope();

    // Marginal sigmas
    Eigen::Vector3d sp = Eigen::Vector3d::Zero();
    Eigen::Vector3d sv = Eigen::Vector3d::Zero();
    Eigen::Vector3d sa = Eigen::Vector3d::Zero();
    try {
      if constexpr (UseSE23) {
        gtsam::Matrix C = smoother.marginalCovariance(X(correction_count));
        sa << std::sqrt(C(0, 0)), std::sqrt(C(1, 1)), std::sqrt(C(2, 2));
        sv << std::sqrt(C(3, 3)), std::sqrt(C(4, 4)), std::sqrt(C(5, 5));
        sp << std::sqrt(C(6, 6)), std::sqrt(C(7, 7)), std::sqrt(C(8, 8));
      } else {
        gtsam::Matrix Cp = smoother.marginalCovariance(X(correction_count));
        gtsam::Matrix Cv = smoother.marginalCovariance(V(correction_count));
        sa << std::sqrt(Cp(0, 0)), std::sqrt(Cp(1, 1)), std::sqrt(Cp(2, 2));
        sp << std::sqrt(Cp(3, 3)), std::sqrt(Cp(4, 4)), std::sqrt(Cp(5, 5));
        sv << std::sqrt(Cv(0, 0)), std::sqrt(Cv(1, 1)), std::sqrt(Cv(2, 2));
      }
    } catch (const std::exception&) {
      // fall through with zeros
    }
    pos_sigma.back() = sp;
    vel_sigma.back() = sv;
    att_sigma.back() = sa;

    // --- Debug log row ---
    if (debug_ofs.is_open()) {
      // Propagated attitude (before smoother) and smoothed attitude.
      const double yaw_prop = R_prop.yaw();
      const double pitch_prop = R_prop.pitch();
      const double roll_prop = R_prop.roll();
      const double yaw_post = R_est.yaw();
      const double pitch_post = R_est.pitch();
      const double roll_post = R_est.roll();

      // Attitude innovation (2D Unit3 error) at rover ticks only. Computed
      // from the *propagated* rotation so it reflects what the factor saw
      // before the update.
      double inn_u = std::numeric_limits<double>::quiet_NaN();
      double inn_v = std::numeric_limits<double>::quiet_NaN();
      double inn_n = std::numeric_limits<double>::quiet_NaN();
      Eigen::Vector3d nPred =
          Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
      if (rover_tick && before_handover) {
        const gtsam::Unit3 nZ_meas(z_gnss_comp_shifted[idx]);
        nPred = R_prop.matrix() * baseline_body.normalized();
        const gtsam::Unit3 nPred_u(nPred);
        const gtsam::Vector2 e = nZ_meas.errorVector(nPred_u);
        inn_u = e(0);
        inn_v = e(1);
        inn_n = e.norm();
      }

      const auto& bg = prev_bias.gyroscope();
      const auto& ba = prev_bias.accelerometer();
      // Log the *shifted* baseline so z_gnss_comp matches what the factor
      // actually consumed (identical to d.z_gnss_comp when lag==0).
      const auto& zc = z_gnss_comp_shifted[idx];

      debug_ofs << t_now << ',' << correction_count << ',' << idx << ','
                << (mb_tick ? 1 : 0) << ',' << (rover_tick ? 1 : 0) << ','
                << (baro_tick ? 1 : 0) << ',' << (before_handover ? 1 : 0)
                << ',' << yaw_prop << ',' << yaw_post << ',' << pitch_prop
                << ',' << pitch_post << ',' << roll_prop << ',' << roll_post
                << ',' << inn_u << ',' << inn_v << ',' << inn_n << ',' << zc.x()
                << ',' << zc.y() << ',' << zc.z() << ',' << nPred.x() << ','
                << nPred.y() << ',' << nPred.z() << ',' << bg.x() << ','
                << bg.y() << ',' << bg.z() << ',' << ba.x() << ',' << ba.y()
                << ',' << ba.z() << ',' << baseline_body.x() << ','
                << baseline_body.y() << ',' << baseline_body.z() << '\n';
    }

    if (correction_count % 100 == 0) {
      printf("step %lld  t=%.2f  p=[%.2f %.2f %.2f]\n",
             static_cast<long long>(idx), t_now, p_est.x(), p_est.y(),
             p_est.z());
    }
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  printf("Final position: [%.4f, %.4f, %.4f]\n", est_p.back().x(),
         est_p.back().y(), est_p.back().z());
  printf("Elapsed: %.3f s | data horizon: %.1f s\n", elapsed.count(),
         accumulated_time);

  // --- Save ---
  const std::string pre = opts.output_dir + opts.output_prefix;
  std::vector<Eigen::Vector3d> att_rpy;
  att_rpy.reserve(est_R.size());
  for (const auto& R : est_R)
    att_rpy.emplace_back(R.roll(), R.pitch(), R.yaw());
  write_scalar(pre + "time.csv", d.t);
  write_vec3(pre + "pos.csv", est_p);
  write_vec3(pre + "vel.csv", est_v);
  write_vec3(pre + "att.csv", att_rpy);
  write_vec3(pre + "acc.csv", est_ba);
  write_vec3(pre + "gyro.csv", est_bg);
  write_vec3(pre + "pos_std.csv", pos_sigma);
  write_vec3(pre + "vel_std.csv", vel_sigma);
  write_vec3(pre + "att_std.csv", att_sigma);
  printf("Saved results to %s*\n", pre.c_str());
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  Options opts;
  if (!parse_args(argc, argv, opts)) return 1;

  // Build a tag for the output prefix so runs don't stomp on each other.
  const char* pre_tag = opts.use_se23 ? "se23" : "se3";
  const char* bias_tag = opts.use_gauss_markov ? "gm" : "cb";
  const char* ho_tag = opts.handover == Handover::None    ? "no_handover"
                       : opts.handover == Handover::Angle ? "angle_handover"
                       : opts.handover == Handover::AngleBaro
                           ? "angle_baro_handover"
                           : "angle_range_handover";
  const char* rb_tag = opts.robust == Robust::None           ? "no_robust"
                       : opts.robust == Robust::GemanMcClure ? "gm"
                                                             : "tukey";
  opts.output_prefix +=
      std::string(pre_tag) + "_" + bias_tag + "_" + ho_tag + "_" + rb_tag + "_";

  printf("Input:       %s\n", opts.input_file.c_str());
  printf("Output:      %s%s*\n", opts.output_dir.c_str(),
         opts.output_prefix.c_str());
  printf("State:       %s\n",
         opts.use_se23 ? "ExtendedPose3 (SE23)" : "Pose3 (SE3)");
  printf("Bias:        %s\n",
         opts.use_gauss_markov ? "Gauss-Markov" : "Constant");
  printf("Handover:    %s\n", ho_tag);
  printf("Robust:      %s (k=%.4f)\n", rb_tag, opts.robust_threshold);
  printf(
      "Baro:        origin_msl=%.2f m, sigma=%.2f m, T_ground=%.1f C, p0=%s\n",
      opts.baro_origin_msl, opts.baro_sigma, opts.ground_temp_c,
      opts.baro_p0_kpa > 0.0 ? "manual" : "auto");

  auto data = parse_multirotor_csv(opts.input_file);
  if (!data) return 1;
  printf("Loaded %zu samples\n", data->t.size());

  using GM = gtsam::imuBias::GaussMarkovBias;
  using CB = gtsam::imuBias::ConstantBias;

  try {
    if (opts.use_gauss_markov) {
      if (opts.use_se23)
        run_estimation<GM, true>(*data, opts);
      else
        run_estimation<GM, false>(*data, opts);
    } else {
      if (opts.use_se23)
        run_estimation<CB, true>(*data, opts);
      else
        run_estimation<CB, false>(*data, opts);
    }
  } catch (const gtsam::IndeterminateSystemException& e) {
    std::cerr << "IndeterminateSystemException: " << e.what() << "\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 2;
  }
  return 0;
}
