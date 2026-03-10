/**
 * @file run_multirotor_cs_df.cpp
 * @brief Fixed-lag smoother port of ins_fgo/programs/run_multirotor_cs_df.cpp,
 *        rewritten in the IFAC WC 2026 style (raw GTSAM factors, no parnav
 *        Parsers, and the pressure-based parnav::BaroFactor with origin_msl
 *        and p0 calibration).
 *
 * Paths and data layout are unchanged:
 *   <base>/{time,imu,rtk,df,cs,baro,uwb1..5}.csv
 * Default base = ieee_fusion_cs_df_data/aiding_proper.
 *
 * Compile-time switches from the original have been promoted to CLI flags:
 *
 *   --aiding {proper|rtk_range|cs_downsampled}     (default: proper)
 *   --handover {none|angle-range|uwb|angle-baro}   (default: none -> RTK)
 *   --robust {none|gm|tukey}                        (default: none)
 *   --switch-time <s>                              (default: 400)
 *
 * Robust kernel detail (matches original parnav hard-coded behaviour): with
 * --robust gm the Geman-McClure threshold is 0.9 for azimuth and range but
 * 3.0 for elevation. Tukey uses 4.6851 across all three.
 */

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/CombinedImuFactorGM.h>
#include <gtsam/navigation/PreintegrationCombinedParamsT.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <Eigen/Core>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "AzimuthFactor.hpp"
#include "BaroFactor.hpp"
#include "ElevationFactor.hpp"
#include "RangeFactor.hpp"
#include "utils.hpp"

using parnav::deg2rad;
using parnav::rad2deg;

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::D;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

// ============================================================================
// CLI
// ============================================================================

enum class Aiding { Proper, RtkRange, CsDownsampled };
enum class Handover { None, AngleRange, Uwb, AngleBaro };
enum class Robust { None, GemanMcClure, Tukey };

struct Options {
  Aiding aiding = Aiding::Proper;
  Handover handover = Handover::None;
  Robust robust = Robust::None;
  double switch_time = 400.0;
  bool use_music = false;

  std::string base_path = "/Users/ghms/ws/ntnu/parnav/ieee_fusion_cs_df_data";
  std::string output_dir =
      "/Users/ghms/ws/ntnu/parnav/ieee_fusion_cs_df_results_root_new/";
  std::string output_prefix = "multirotor_cs_df_";

  // Baro
  double baro_origin_msl = 0.0;  // dataset origin altitude; override per run
  double baro_p0_kpa = -1.0;     // auto-calibrate when <0
  double baro_bias_sigma = 1.0;
  double baro_sigma = 1.0;
  // Ground-level air temperature in °C; drives T0_K used by heightOut.
  // Defaults to 15 (standard day). Cold-day flights need this to recover
  // the actual dh/dp slope.
  double ground_temp_c = 15.0;
  // If true, the baro residual is wrapped in the same robust kernel as the
  // PARS / UWB factors. Default is true: baro joins BLE and UWB under the
  // M-estimator. Pass --baro-robust none to reproduce the original parnav
  // cs_df behaviour where the baro was trusted as-is.
  bool baro_robust = true;

  // IMU noise scaling. Acc and gyro are scaled independently because the
  // post-takeoff vibration spectrum is very different on the two channels.
  // `--noise-scaling` (back-compat) still sets BOTH if neither dedicated
  // flag is passed.
  double acc_noise_scaling = 33.0;
  double gyro_noise_scaling = 100.0;
  double bias_scaling = 5.0;
};

static void print_usage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " [options]\n"
      << "  --aiding {proper|rtk_range|cs_downsampled}\n"
      << "  --handover {none|angle-range|uwb|angle-baro}\n"
      << "  --robust {none|gm|tukey}\n"
      << "  --switch-time <s>\n"
      << "  --base-path <path>          (override default base)\n"
      << "  --output-dir <path>\n"
      << "  --output-prefix <str>\n"
      << "  --baro-origin-msl <m>\n"
      << "  --baro-p0-kpa <v>           (negative -> auto-calibrate)\n"
      << "  --baro-bias-sigma <m>\n"
      << "  --baro-sigma <m>\n"
      << "  --ground-temp-c <C>         sea-level T0 follows from ground\n"
      << "                              temperature (default 15 = std day)\n"
      << "  --baro-robust {match|none}  (default: match; none reproduces the\n"
      << "                               original parnav behaviour where baro\n"
      << "                               had no robust kernel)\n"
      << "  --noise-scaling <s>         (sets both acc and gyro)\n"
      << "  --acc-noise-scaling <s>     (default 33)\n"
      << "  --gyro-noise-scaling <s>    (default 100)\n"
      << "  --bias-scaling <s>          (default 50)\n"
      << "  --use-music                 (use _root dataset variant)\n"
      << "  -h, --help\n";
}

static bool parse_args(int argc, char** argv, Options& o) {
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
    } else if (a == "--aiding") {
      if (!need(i, "--aiding")) return false;
      std::string v = argv[++i];
      if (v == "proper")
        o.aiding = Aiding::Proper;
      else if (v == "rtk_range")
        o.aiding = Aiding::RtkRange;
      else if (v == "cs_downsampled")
        o.aiding = Aiding::CsDownsampled;
      else {
        std::cerr << "Unknown --aiding " << v << "\n";
        return false;
      }
    } else if (a == "--handover") {
      if (!need(i, "--handover")) return false;
      std::string v = argv[++i];
      if (v == "none")
        o.handover = Handover::None;
      else if (v == "angle-range")
        o.handover = Handover::AngleRange;
      else if (v == "uwb")
        o.handover = Handover::Uwb;
      else if (v == "angle-baro")
        o.handover = Handover::AngleBaro;
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
    } else if (a == "--switch-time") {
      if (!need(i, "--switch-time")) return false;
      o.switch_time = std::stod(argv[++i]);
    } else if (a == "--base-path") {
      if (!need(i, "--base-path")) return false;
      o.base_path = argv[++i];
    } else if (a == "--output-dir") {
      if (!need(i, "--output-dir")) return false;
      o.output_dir = argv[++i];
      if (!o.output_dir.empty() && o.output_dir.back() != '/')
        o.output_dir.push_back('/');
    } else if (a == "--output-prefix") {
      if (!need(i, "--output-prefix")) return false;
      o.output_prefix = argv[++i];
    } else if (a == "--baro-origin-msl") {
      if (!need(i, "--baro-origin-msl")) return false;
      o.baro_origin_msl = std::stod(argv[++i]);
    } else if (a == "--baro-p0-kpa") {
      if (!need(i, "--baro-p0-kpa")) return false;
      o.baro_p0_kpa = std::stod(argv[++i]);
    } else if (a == "--baro-bias-sigma") {
      if (!need(i, "--baro-bias-sigma")) return false;
      o.baro_bias_sigma = std::stod(argv[++i]);
    } else if (a == "--baro-sigma") {
      if (!need(i, "--baro-sigma")) return false;
      o.baro_sigma = std::stod(argv[++i]);
    } else if (a == "--ground-temp-c") {
      if (!need(i, "--ground-temp-c")) return false;
      o.ground_temp_c = std::stod(argv[++i]);
    } else if (a == "--baro-robust") {
      if (!need(i, "--baro-robust")) return false;
      std::string v = argv[++i];
      if (v == "match")
        o.baro_robust = true;
      else if (v == "none")
        o.baro_robust = false;
      else {
        std::cerr << "Unknown --baro-robust " << v << "\n";
        return false;
      }
    } else if (a == "--noise-scaling") {
      if (!need(i, "--noise-scaling")) return false;
      const double v = std::stod(argv[++i]);
      o.acc_noise_scaling = v;
      o.gyro_noise_scaling = v;
    } else if (a == "--acc-noise-scaling") {
      if (!need(i, "--acc-noise-scaling")) return false;
      o.acc_noise_scaling = std::stod(argv[++i]);
    } else if (a == "--gyro-noise-scaling") {
      if (!need(i, "--gyro-noise-scaling")) return false;
      o.gyro_noise_scaling = std::stod(argv[++i]);
    } else if (a == "--bias-scaling") {
      if (!need(i, "--bias-scaling")) return false;
      o.bias_scaling = std::stod(argv[++i]);
    } else if (a == "--use-music") {
      o.use_music = true;
    } else {
      std::cerr << "Unknown arg: " << a << "\n";
      print_usage(argv[0]);
      return false;
    }
  }
  return true;
}

// ============================================================================
// CSV helpers (per-sensor files)
// ============================================================================

namespace {

std::string join(const std::string& folder, const std::string& file) {
  if (folder.empty()) return file;
  if (folder.back() == '/' || folder.back() == '\\') return folder + file;
  return folder + "/" + file;
}

std::vector<std::vector<double>> read_csv(const std::string& path) {
  std::ifstream f(path);
  std::vector<std::vector<double>> rows;
  if (!f) {
    std::cerr << "Could not open " << path << "\n";
    return rows;
  }
  std::string line;
  std::getline(f, line);  // header
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string cell;
    std::vector<double> r;
    while (std::getline(ss, cell, ',')) {
      try {
        r.push_back(std::stod(cell));
      } catch (...) {
        r.push_back(0.0);
      }
    }
    rows.push_back(std::move(r));
  }
  return rows;
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

}  // namespace

// ============================================================================
// Data
// ============================================================================

struct Data {
  std::vector<double> t;
  std::vector<Eigen::Vector3d> f_m, w_m;
  std::vector<int> rtk_idx;
  std::vector<Eigen::Vector3d> z_rtk;  // RTK position (NED)
  std::vector<int> df_idx;
  std::vector<double> df_azi, df_ele;
  std::vector<int> cs_idx;
  std::vector<double> cs_range;
  std::vector<int> baro_idx;
  std::vector<double> z_baro;  // pressure [kPa]
  static constexpr int NUM_UWB = 5;
  std::array<std::vector<int>, NUM_UWB> uwb_idx;
  std::array<std::vector<double>, NUM_UWB> uwb_range;
};

static std::optional<Data> load_data(const std::string& base_path) {
  Data d;
  // time
  for (const auto& r : read_csv(join(base_path, "time.csv")))
    if (!r.empty()) d.t.push_back(r[0]);
  if (d.t.empty()) {
    std::cerr << "Empty time.csv at " << base_path << "\n";
    return std::nullopt;
  }
  const size_t N = d.t.size();

  // imu
  auto imu = read_csv(join(base_path, "imu.csv"));
  d.f_m.resize(N, Eigen::Vector3d::Zero());
  d.w_m.resize(N, Eigen::Vector3d::Zero());
  for (size_t i = 0; i < std::min(N, imu.size()); ++i) {
    if (imu[i].size() >= 7) {
      d.f_m[i] << imu[i][1], imu[i][2], imu[i][3];
      d.w_m[i] << imu[i][4], imu[i][5], imu[i][6];
    }
  }

  // rtk: idx, px, py, pz, ax, ay, az (we only need idx + position)
  d.rtk_idx.assign(N, 0);
  d.z_rtk.assign(N, Eigen::Vector3d::Zero());
  auto rtk = read_csv(join(base_path, "rtk.csv"));
  for (size_t i = 0; i < std::min(N, rtk.size()); ++i) {
    if (rtk[i].size() >= 4 && rtk[i][0] != 0.0) {
      d.rtk_idx[i] = 1;
      d.z_rtk[i] << rtk[i][1], rtk[i][2], rtk[i][3];
    }
  }

  // df: idx, azi, ele
  d.df_idx.assign(N, 0);
  d.df_azi.assign(N, 0.0);
  d.df_ele.assign(N, 0.0);
  auto df = read_csv(join(base_path, "df.csv"));
  for (size_t i = 0; i < std::min(N, df.size()); ++i) {
    if (df[i].size() >= 3 && df[i][0] != 0.0) {
      d.df_idx[i] = 1;
      d.df_azi[i] = df[i][1];
      d.df_ele[i] = df[i][2];
    }
  }

  // cs: idx, range
  d.cs_idx.assign(N, 0);
  d.cs_range.assign(N, 0.0);
  auto cs = read_csv(join(base_path, "cs.csv"));
  for (size_t i = 0; i < std::min(N, cs.size()); ++i) {
    if (cs[i].size() >= 2 && cs[i][0] != 0.0) {
      d.cs_idx[i] = 1;
      d.cs_range[i] = cs[i][1];
    }
  }

  // baro: idx, pressure
  d.baro_idx.assign(N, 0);
  d.z_baro.assign(N, std::numeric_limits<double>::quiet_NaN());
  auto baro = read_csv(join(base_path, "baro.csv"));
  for (size_t i = 0; i < std::min(N, baro.size()); ++i) {
    if (baro[i].size() >= 2 && baro[i][0] != 0.0) {
      d.baro_idx[i] = 1;
      d.z_baro[i] = baro[i][1];
    }
  }

  // uwb1..5
  for (int u = 0; u < Data::NUM_UWB; ++u) {
    d.uwb_idx[u].assign(N, 0);
    d.uwb_range[u].assign(N, 0.0);
    auto uwb =
        read_csv(join(base_path, "uwb" + std::to_string(u + 1) + ".csv"));
    for (size_t i = 0; i < std::min(N, uwb.size()); ++i) {
      if (uwb[i].size() >= 2 && uwb[i][0] != 0.0) {
        d.uwb_idx[u][i] = 1;
        d.uwb_range[u][i] = uwb[i][1];
      }
    }
  }
  return d;
}

// ============================================================================
// Estimation
// ============================================================================

template <class BIAS>
void run_estimation(const Data& d, const Options& opts) {
  using BiasT = BIAS;
  using PIM = gtsam::PreintegratedCombinedMeasurementsGMT<
      gtsam::ManifoldPreintegrationT<BiasT>, BiasT>;
  using ImuFactor = gtsam::CombinedImuFactorGMT<PIM, BiasT>;
  using PoseParam = gtsam::Pose3;

  // --- IMU noise (matches original program) ---
  const double g0 = 9.80665;
  const double vrw = 0.07;
  const double arw = 0.15;
  const double bias_instability_acc = 0.05;  // milli g
  const double bias_instability_ars = 0.5;   // deg/hour
  const double T_acc = 3600.0, T_ars = 3600.0;
  const double q_v = std::pow(opts.acc_noise_scaling * vrw / 60.0, 2.0);
  const double q_o =
      std::pow((opts.gyro_noise_scaling * arw / 60.0) * deg2rad(1.0), 2.0);
  const double q_b_v =
      (2.0 / T_acc) *
      std::pow(opts.bias_scaling * bias_instability_acc * (g0 / 1000.0), 2.0);
  const double q_b_o =
      (2.0 / T_ars) *
      std::pow(
          (opts.bias_scaling * bias_instability_ars / 3600.0) * deg2rad(1.0),
          2.0);
  const double q_p = 1e-40;

  auto p = gtsam::PreintegrationCombinedParamsT<BiasT>::MakeSharedD(g0);
  p->accelerometerCovariance = gtsam::I_3x3 * q_v;
  p->gyroscopeCovariance = gtsam::I_3x3 * q_o;
  p->integrationCovariance = gtsam::I_3x3 * q_p;
  p->biasAccCovariance = gtsam::I_3x3 * q_b_v;
  p->biasOmegaCovariance = gtsam::I_3x3 * q_b_o;

  // --- Measurement noise ---
  gtsam::Matrix3 R_GNSS_pos = gtsam::I_3x3;
  R_GNSS_pos(0, 0) = 1.5 * 1.5;
  R_GNSS_pos(1, 1) = 1.5 * 1.5;
  R_GNSS_pos(2, 2) = 3.0 * 3.0;
  auto gnss_noise =
      gtsam::noiseModel::Diagonal::Variances(R_GNSS_pos.diagonal());

  auto baro_base = gtsam::noiseModel::Isotropic::Sigma(1, opts.baro_sigma);
  auto baro_bias_noise =
      gtsam::noiseModel::Isotropic::Sigma(1, opts.baro_bias_sigma);

  // PARS / BT noise (original: 5 deg az/el, 1.5 m range; here we keep 2.5 m
  // range to match ifac defaults — overridable via robust threshold logic).
  auto pars_azi_base = gtsam::noiseModel::Isotropic::Sigma(1, deg2rad(5.0));
  auto pars_ele_base = gtsam::noiseModel::Isotropic::Sigma(1, deg2rad(5.0));
  auto pars_range_base = gtsam::noiseModel::Isotropic::Sigma(1, 3.5);

  // Geman-McClure: 0.9 for azimuth/range, 3.0 for elevation (matches the
  // original parnav PARSParser hard-coded behaviour). Tukey: 4.6851 uniform.
  const double k_az = (opts.robust == Robust::Tukey) ? 4.6851 : 0.9;
  const double k_el = (opts.robust == Robust::Tukey) ? 4.6851 : 3.0;
  const double k_ra = (opts.robust == Robust::Tukey) ? 4.6851 : 0.9;
  auto pars_azi_noise = wrap_robust(pars_azi_base, opts.robust, k_az);
  auto pars_ele_noise = wrap_robust(pars_ele_base, opts.robust, k_el);
  auto pars_range_noise = wrap_robust(pars_range_base, opts.robust, k_ra);

  // UWB and barometer: same robust kernel as PARS, GmC k=0.9 / Tukey 4.6851.
  auto uwb_base = gtsam::noiseModel::Isotropic::Sigma(1, 1.5);
  const double k_uwb = (opts.robust == Robust::Tukey) ? 4.6851 : 0.9;
  const double k_baro = (opts.robust == Robust::Tukey) ? 4.6851 : 0.9;
  auto uwb_noise = wrap_robust(uwb_base, opts.robust, k_uwb);
  auto baro_noise = opts.baro_robust
                        ? wrap_robust(baro_base, opts.robust, k_baro)
                        : gtsam::SharedNoiseModel(baro_base);

  // --- PARS / radio geometry (NED <- radio) ---
  Eigen::Matrix3d R_rn;
  R_rn << -0.0369, 0.1402, -0.9894, -0.0456, -0.9893, -0.1384, -0.9983, 0.0401,
      0.0429;
  Eigen::Vector3d t_rn(-0.0519, -0.0820, 0.0492);
  if (opts.use_music) {
    R_rn << -0.0369, 0.1382, -0.9897, -0.0446, -0.9896, -0.1366, -0.9983,
        0.0391, 0.0426;
    t_rn << -0.0380, -0.1358, 0.0471;
  }

  // UWB anchors in NED. The factor takes the landmark in nav frame, so we use
  // the anchor position directly (no negation like the original parser did,
  // since the parser had its own sign convention).
  const std::array<gtsam::Point3, Data::NUM_UWB> uwb_anchors = {
      gtsam::Point3(0.000, 0.000, -0.000),
      gtsam::Point3(-13.020, 11.667, 0.228),
      gtsam::Point3(-29.094, 7.693, 0.603),
      gtsam::Point3(-19.017, -28.132, 0.675),
      gtsam::Point3(0.000, -23.243, 0.233)};

  // --- Initial state ---
  gtsam::Rot3 R0 = gtsam::Rot3::Identity();
  gtsam::Point3 p0 = gtsam::Point3::Zero();
  gtsam::Vector3 v0 = gtsam::Vector3::Zero();
  const double A_pos = 2.5, A_vel = 0.5, A_att_rp = deg2rad(180.0);
  const double A_pos_z = 5.0;
  const double A_yaw = deg2rad(180.0);
  const double A_acc_bias = 0.1;
  const double A_gyro_bias = deg2rad(0.5);

  BiasT prior_bias;
  if constexpr (std::is_same_v<BiasT, gtsam::imuBias::GaussMarkovBias>) {
    prior_bias = gtsam::imuBias::GaussMarkovBias(
        gtsam::Vector3::Zero(), gtsam::Vector3::Zero(), T_acc, T_ars);
  }
  auto bias_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << A_acc_bias, A_acc_bias, A_acc_bias, A_gyro_bias,
       A_gyro_bias, A_gyro_bias)
          .finished());

  // --- Smoother ---
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

  auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << A_att_rp, A_att_rp, A_yaw, A_pos, A_pos, A_pos_z)
          .finished());
  auto vel_noise = gtsam::noiseModel::Isotropic::Sigma(3, A_vel);
  graph.addPrior<gtsam::Pose3>(X(0), gtsam::Pose3(R0, p0), pose_noise);
  graph.addPrior<gtsam::Vector3>(V(0), v0, vel_noise);
  graph.addPrior<BiasT>(B(0), prior_bias, bias_noise);
  values.insert(X(0), gtsam::Pose3(R0, p0));
  values.insert(V(0), v0);
  values.insert(B(0), prior_bias);
  timestamps[X(0)] = 0.0;
  timestamps[V(0)] = 0.0;
  timestamps[B(0)] = 0.0;

  const bool perform_handover = (opts.handover != Handover::None);
  const bool use_uwb_mode = (opts.handover == Handover::Uwb);
  const bool use_baro_mode = (opts.handover == Handover::AngleBaro);

  // Auto-calibrate baro p0 from the first 100 s of data, when the drone is
  // assumed to be on the ground and the static assumption holds. Baro
  // measurements themselves fire from t0 onwards.
  double baro_p0 = (opts.baro_p0_kpa > 0.0) ? opts.baro_p0_kpa : 101.29;
  const double baro_T0_K = parnav::BaroFactor<PoseParam>::T0_from_ground_temp_c(
      opts.ground_temp_c, opts.baro_origin_msl);
  if (use_baro_mode) {
    double p_sum = 0.0;
    int p_n = 0;
    for (size_t k = 0; k < d.t.size() && d.t[k] - d.t[0] < 100.0; ++k) {
      if (d.baro_idx[k] != 0) {
        p_sum += d.z_baro[k];
        ++p_n;
      }
    }
    if (opts.baro_p0_kpa <= 0.0 && p_n > 0) {
      const double p_avg = p_sum / p_n;
      baro_p0 = parnav::BaroFactor<PoseParam>::p0_from_known_altitude(
          p_avg, opts.baro_origin_msl, baro_T0_K);
      printf(
          "Baro init: p_avg=%.3f kPa over %d samples, origin=%.2f m, "
          "T_ground=%.1f C -> T0=%.2f K, p0_local=%.3f kPa\n",
          p_avg, p_n, opts.baro_origin_msl, opts.ground_temp_c, baro_T0_K,
          baro_p0);
    } else {
      printf("Baro init: p0=%.3f kPa (manual or no samples), T0=%.2f K\n",
             baro_p0, baro_T0_K);
    }
    graph.addPrior<double>(D(0), 0.0, baro_bias_noise);
    values.insert(D(0), 0.0);
    timestamps[D(0)] = 0.0;
  }

  smoother.update(graph, values, timestamps);

  // --- Loop ---
  auto preintegrated = std::make_shared<PIM>(p, prior_bias);
  gtsam::NavState prev_state(gtsam::Pose3(R0, p0), v0);
  BiasT prev_bias = prior_bias;
  double prev_baro_bias = 0.0;

  std::vector<gtsam::Rot3> est_R{R0};
  std::vector<Eigen::Vector3d> est_p{p0};
  std::vector<Eigen::Vector3d> est_v{v0};
  std::vector<Eigen::Vector3d> est_ba{prior_bias.accelerometer()};
  std::vector<Eigen::Vector3d> est_bg{prior_bias.gyroscope()};
  std::vector<double> est_baro_bias{0.0};
  std::vector<Eigen::Vector3d> pos_sigma{Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> vel_sigma{Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> att_sigma{Eigen::Vector3d::Zero()};

  // Variances (NOT sigmas) of the last successful smoother marginal. Between
  // corrections we add the diagonal of the preintegrated noise covariance
  // accumulated since that correction, so the reported sigma actually grows
  // while we coast on IMU instead of staying flat.
  Eigen::Vector3d last_pos_var = Eigen::Vector3d::Zero();
  Eigen::Vector3d last_vel_var = Eigen::Vector3d::Zero();
  Eigen::Vector3d last_att_var = Eigen::Vector3d::Zero();

  // Preintegrated covariance layout for the legacy SE3 CombinedImuFactor:
  //   [theta(3), rho(3), nu(3), bias_acc(3), bias_gyro(3)]
  // (See gtsam/navigation/CombinedImuFactor.h.) The propagated marginal is
  // approximately `last_*_var + preintMeasCov diagonal for the same block`.
  auto propagated_sigmas = [&](Eigen::Vector3d& pos_s, Eigen::Vector3d& vel_s,
                               Eigen::Vector3d& att_s) {
    const auto C = preintegrated->preintMeasCov();
    const Eigen::Vector3d att_v = C.template block<3, 3>(0, 0).diagonal();
    const Eigen::Vector3d pos_v = C.template block<3, 3>(3, 3).diagonal();
    const Eigen::Vector3d vel_v = C.template block<3, 3>(6, 6).diagonal();
    for (int k = 0; k < 3; ++k) {
      pos_s[k] = std::sqrt(last_pos_var[k] + pos_v[k]);
      vel_s[k] = std::sqrt(last_vel_var[k] + vel_v[k]);
      att_s[k] = std::sqrt(last_att_var[k] + att_v[k]);
    }
  };

  double accumulated_time = 0.0;
  int correction_count = 0;
  const auto start = std::chrono::system_clock::now();
  const auto N = static_cast<int64_t>(d.t.size());

  bool aborted_early = false;
  for (int64_t idx = 1; idx < N; ++idx) try {
      const double dt = d.t[idx] - d.t[idx - 1];
      accumulated_time += dt;
      const double t_now = d.t[idx];
      preintegrated->integrateMeasurement(d.f_m[idx], d.w_m[idx], dt);

      bool any_uwb_present = false;
      if (use_uwb_mode) {
        for (int u = 0; u < Data::NUM_UWB; ++u)
          if (d.uwb_idx[u][idx]) {
            any_uwb_present = true;
            break;
          }
      }

      bool use_gnss = false, use_bt = false, use_uwb = false, use_baro = false;
      if (perform_handover) {
        if (accumulated_time < opts.switch_time) {
          use_gnss = d.rtk_idx[idx] != 0;
          if (use_baro_mode) use_baro = d.baro_idx[idx] != 0;
        } else {
          if (use_uwb_mode) {
            use_uwb = any_uwb_present;
          } else if (use_baro_mode) {
            use_bt = (d.df_idx[idx] || d.cs_idx[idx]);
            use_baro = d.baro_idx[idx] != 0;
          } else {
            // angle-range
            use_bt = (d.df_idx[idx] || d.cs_idx[idx]);
          }
        }
      } else {
        use_gnss = d.rtk_idx[idx] != 0;
      }

      const bool has_meas = use_gnss || use_bt || use_uwb || use_baro;
      if (!has_meas) {
        // log propagated state with propagated uncertainty
        gtsam::NavState prop = preintegrated->predict(prev_state, prev_bias);
        est_R.push_back(prop.pose().rotation());
        est_p.push_back(prop.pose().translation());
        est_v.push_back(prop.v());
        est_ba.push_back(prev_bias.accelerometer());
        est_bg.push_back(prev_bias.gyroscope());
        est_baro_bias.push_back(prev_baro_bias);
        Eigen::Vector3d ps, vs, as;
        propagated_sigmas(ps, vs, as);
        pos_sigma.push_back(ps);
        vel_sigma.push_back(vs);
        att_sigma.push_back(as);
        continue;
      }
      correction_count++;

      graph.resize(0);
      values.clear();
      timestamps.clear();

      graph.add(ImuFactor(X(correction_count - 1), V(correction_count - 1),
                          X(correction_count), V(correction_count),
                          B(correction_count - 1), B(correction_count),
                          *preintegrated));

      gtsam::NavState prop = preintegrated->predict(prev_state, prev_bias);
      values.insert(X(correction_count), prop.pose());
      values.insert(V(correction_count), prop.v());
      values.insert(B(correction_count), prev_bias);
      timestamps[X(correction_count)] = t_now;
      timestamps[V(correction_count)] = t_now;
      timestamps[B(correction_count)] = t_now;
      if (use_baro_mode) timestamps[D(0)] = t_now;

      if (use_gnss) {
        graph.add(
            gtsam::GPSFactor(X(correction_count), d.z_rtk[idx], gnss_noise));
        printf("[t=%.3f c=%d] +GNSS pos [%.2f %.2f %.2f]\n", t_now,
               correction_count, d.z_rtk[idx].x(), d.z_rtk[idx].y(),
               d.z_rtk[idx].z());
      }
      if (use_bt) {
        if (d.df_idx[idx]) {
          graph.add(parnav::AzimuthFactor<PoseParam>(
              X(correction_count), pars_azi_noise, d.df_azi[idx], t_rn, R_rn));
          graph.add(parnav::ElevationFactor<PoseParam>(
              X(correction_count), pars_ele_noise, d.df_ele[idx], t_rn, R_rn));
          printf("[t=%.3f c=%d] +DF az=%.2f deg el=%.2f deg\n", t_now,
                 correction_count, rad2deg(d.df_azi[idx]),
                 rad2deg(d.df_ele[idx]));
        }
        if (d.cs_idx[idx]) {
          graph.add(parnav::RangeFactor<PoseParam>(
              X(correction_count), pars_range_noise, d.cs_range[idx], t_rn,
              R_rn));
          printf("[t=%.3f c=%d] +CS range=%.2f m\n", t_now, correction_count,
                 d.cs_range[idx]);
        }
      }
      if (use_uwb) {
        for (int u = 0; u < Data::NUM_UWB; ++u) {
          if (!d.uwb_idx[u][idx]) continue;
          graph.add(parnav::RangeFactor<PoseParam>(
              X(correction_count), uwb_noise, d.uwb_range[u][idx],
              uwb_anchors[u]));
          printf("[t=%.3f c=%d] +UWB%d range=%.2f m\n", t_now, correction_count,
                 u + 1, d.uwb_range[u][idx]);
        }
      }
      if (use_baro) {
        graph.add(parnav::BaroFactor<PoseParam>(
            X(correction_count), D(0), d.z_baro[idx], baro_noise,
            opts.baro_origin_msl, baro_p0, baro_T0_K));
        printf("[t=%.3f c=%d] +Baro p=%.3f kPa (T0=%.2f K)\n", t_now,
               correction_count, d.z_baro[idx], baro_T0_K);
      }

      printf("[t=%.3f c=%d] optimize\n", t_now, correction_count);
      smoother.update(graph, values, timestamps);

      gtsam::Values result = smoother.calculateEstimate();
      auto pose = result.at<gtsam::Pose3>(X(correction_count));
      auto vel = result.at<gtsam::Vector3>(V(correction_count));
      prev_state = gtsam::NavState(pose, vel);
      prev_bias = result.at<BiasT>(B(correction_count));
      if (use_baro_mode) prev_baro_bias = result.at<double>(D(0));
      preintegrated->resetIntegrationAndSetBias(prev_bias);

      est_R.push_back(pose.rotation());
      est_p.push_back(pose.translation());
      est_v.push_back(vel);
      est_ba.push_back(prev_bias.accelerometer());
      est_bg.push_back(prev_bias.gyroscope());
      est_baro_bias.push_back(prev_baro_bias);

      Eigen::Vector3d sp = Eigen::Vector3d::Zero();
      Eigen::Vector3d sv = Eigen::Vector3d::Zero();
      Eigen::Vector3d sa = Eigen::Vector3d::Zero();
      try {
        gtsam::Matrix Cp = smoother.marginalCovariance(X(correction_count));
        gtsam::Matrix Cv = smoother.marginalCovariance(V(correction_count));
        sa << std::sqrt(Cp(0, 0)), std::sqrt(Cp(1, 1)), std::sqrt(Cp(2, 2));
        sp << std::sqrt(Cp(3, 3)), std::sqrt(Cp(4, 4)), std::sqrt(Cp(5, 5));
        sv << std::sqrt(Cv(0, 0)), std::sqrt(Cv(1, 1)), std::sqrt(Cv(2, 2));
        // Snapshot the post-update marginal as the anchor for IMU-coast
        // propagation between corrections.
        last_att_var << Cp(0, 0), Cp(1, 1), Cp(2, 2);
        last_pos_var << Cp(3, 3), Cp(4, 4), Cp(5, 5);
        last_vel_var << Cv(0, 0), Cv(1, 1), Cv(2, 2);
      } catch (const std::exception&) {
      }
      pos_sigma.push_back(sp);
      vel_sigma.push_back(sv);
      att_sigma.push_back(sa);

      if (correction_count % 100 == 0) {
        printf("step %lld  t=%.2f  p=[%.2f %.2f %.2f]\n",
               static_cast<long long>(idx), t_now, pose.translation().x(),
               pose.translation().y(), pose.translation().z());
      }
    } catch (const gtsam::IndeterminateSystemException& e) {
      fprintf(stderr,
              "Smoother went indeterminate at idx=%lld t=%.2f: %s\n"
              "Switching to IMU-only coasting for the remainder of the run.\n",
              static_cast<long long>(idx), d.t[idx], e.what());
      // Re-base preintegration on the last good state so coasting starts clean.
      preintegrated->resetIntegrationAndSetBias(prev_bias);
      // Log a propagated entry for this tick so output length still matches.
      gtsam::NavState prop = preintegrated->predict(prev_state, prev_bias);
      est_R.push_back(prop.pose().rotation());
      est_p.push_back(prop.pose().translation());
      est_v.push_back(prop.v());
      est_ba.push_back(prev_bias.accelerometer());
      est_bg.push_back(prev_bias.gyroscope());
      est_baro_bias.push_back(prev_baro_bias);
      Eigen::Vector3d ps, vs, as;
      propagated_sigmas(ps, vs, as);
      pos_sigma.push_back(ps);
      vel_sigma.push_back(vs);
      att_sigma.push_back(as);
      continue;
    } catch (const std::exception& e) {
      fprintf(stderr, "Exception at idx=%lld t=%.2f: %s\n",
              static_cast<long long>(idx), d.t[idx], e.what());
      aborted_early = true;
      break;
    }

  // If we bailed out early, pad the result vectors so they line up with d.t
  // by repeating the last estimate (keeps the MATLAB plotter happy).
  if (aborted_early) {
    while (static_cast<int64_t>(est_p.size()) < N) {
      est_R.push_back(est_R.back());
      est_p.push_back(est_p.back());
      est_v.push_back(est_v.back());
      est_ba.push_back(est_ba.back());
      est_bg.push_back(est_bg.back());
      est_baro_bias.push_back(est_baro_bias.back());
      pos_sigma.push_back(pos_sigma.back());
      vel_sigma.push_back(vel_sigma.back());
      att_sigma.push_back(att_sigma.back());
    }
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  printf("Final position: [%.4f, %.4f, %.4f]\n", est_p.back().x(),
         est_p.back().y(), est_p.back().z());
  printf("Elapsed: %.3f s | horizon: %.1f s\n", elapsed.count(),
         accumulated_time);

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
  if (use_baro_mode) write_scalar(pre + "baro.csv", est_baro_bias);
  printf("Saved outputs to %s*\n", pre.c_str());
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  Options opts;
  if (!parse_args(argc, argv, opts)) return 1;

  // Resolve base_path / output prefix to match the original program layout.
  if (opts.use_music) {
    opts.base_path += "_root";
    opts.output_dir =
        "/Users/ghms/ws/ntnu/parnav/ieee_fusion_cs_df_results_root/";
  }
  switch (opts.aiding) {
    case Aiding::Proper:
      opts.base_path += "/aiding_proper";
      opts.output_prefix += "aiding_proper_";
      break;
    case Aiding::RtkRange:
      opts.base_path += "/aiding_rtk_range";
      opts.output_prefix += "aiding_rtk_range_";
      break;
    case Aiding::CsDownsampled:
      opts.base_path += "/aiding_cs_downsampled";
      opts.output_prefix += "aiding_cs_downsampled_";
      break;
  }
  const char* rb_tag = opts.robust == Robust::None           ? "no_robust_"
                       : opts.robust == Robust::GemanMcClure ? "geman_mcclure_"
                                                             : "tukey_m_";
  const char* ho_tag = opts.handover == Handover::None         ? "rtk_"
                       : opts.handover == Handover::AngleRange ? "bt_"
                       : opts.handover == Handover::Uwb        ? "uwb_"
                                                               : "baro_";
  opts.output_prefix += rb_tag;
  opts.output_prefix += ho_tag;

  printf("Base:        %s\n", opts.base_path.c_str());
  printf("Output:      %s%s*\n", opts.output_dir.c_str(),
         opts.output_prefix.c_str());
  printf("Handover:    %s switch_time=%.1f s\n", ho_tag, opts.switch_time);
  printf("Robust:      %s\n", rb_tag);

  auto data = load_data(opts.base_path);
  if (!data) return 1;
  printf("Loaded %zu samples\n", data->t.size());

  try {
    run_estimation<gtsam::imuBias::ConstantBias>(*data, opts);
  } catch (const gtsam::IndeterminateSystemException& e) {
    std::cerr << "IndeterminateSystemException: " << e.what() << "\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 2;
  }
  return 0;
}
