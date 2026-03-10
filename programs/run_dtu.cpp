/**
 * @file run_dtu.cpp
 * @brief DTU land_harbour fixed-lag smoother (GNSS + IMU, optional radio DF/CS),
 *        templated on the four {SE3, SE23} x {ConstantBias, GaussMarkovBias}
 *        combinations. Adapted from run_bledar.cpp.
 *
 * Dataset: the receiver is on a STATIONARY land platform; the signal is
 * transmitted from a box on the MOVING otter, which also carries the (Ouster-
 * derived, reconstructed) IMU and its own GNSS. Fed by the CSV folder written by
 * parnav-ros-modules/matlab/export_gtsam_csv.m (dense, IMU-rate, row-aligned).
 *
 *   --aiding {gnss|pars}       gnss = otter GNSS position (GPS factor)
 *                              pars = DF azimuth/elevation + CS range (radio)
 *   --preint {se3|se23}        (default se3)
 *   --bias   {cb|gm}           (default cb)
 *   --robust {none|gm|tukey}   (default none)
 *
 * DIFFERENCES vs run_bledar:
 *   - GNSS is DGPS (~metre), NOT RTK -> looser position noise (see gnss_noise).
 *   - Radio->NED geometry R_rn = R_radio_ned from combined_flight_analysis.m
 *     (t_rn = 0; we only estimated rotation). DF/CS are OFF in the default export
 *     (header-only df/cs.csv), so run gnss aiding until PARS is enabled there.
 *   - IMU noise defaults are INHERITED from the STIM300 tuning and are almost
 *     certainly wrong for the Ouster-derived IMU -- revisit via --acc/gyro-noise.
 *
 * Truth for the error columns is the GNSS position interpolated to the estimator
 * timesteps; with --aiding gnss that truth IS the aiding source, so pos_err is
 * ~0 by construction (a real validation must hold GNSS out). Results ->
 * <output-dir>dtu_{cb,gm}{,_se23}.csv in the plot_3sigma.py column layout.
 */

#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor2.h>
#include <gtsam/navigation/CombinedImuFactorGM.h>
#include <gtsam/navigation/PreintegrationCombinedParamsT.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ManifoldPreintegration.h>
#include <gtsam/navigation/ManifoldPreintegrationSE23.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "AzimuthFactor.hpp"
#include "ElevationFactor.hpp"
#include "RangeFactor.hpp"

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

static inline double deg2rad(double d) { return d * M_PI / 180.0; }

// ============================================================================
// Options
// ============================================================================

enum class Aiding { Gnss, Pars, None };  // None = IMU-only dead reckoning
enum class Robust { None, GemanMcClure, Tukey };

struct Options {
  bool use_se23 = false;
  bool use_gm = false;
  Aiding aiding = Aiding::Gnss;
  Robust robust = Robust::None;
  double switch_time = 400.0;  // pars: bootstrap with GNSS until this [s]
  double dr_start = 60.0;      // aiding none: bootstrap GNSS until this [s], then dead-reckon
  double dr_horizon = INFINITY;  // aiding none: end the run this long after dr_start ([s])
  bool df_use_elevation = false;  // pars: use DF elevation. OFF for DTU — the near-horizon
                                  // planar array has unusable elevation (multipath) that
                                  // blows up the vertical state. Azimuth + CS range only.

  std::string base_path =
      "/Users/ghms/ws/dtu/data_analysis/gtsam_dtu";
  std::string output_dir =
      "/Users/ghms/ws/dtu/data_analysis/gtsam_results/";

  double acc_noise_scaling = 33.0;
  double gyro_noise_scaling = 100.0;
  double bias_scaling = 1.0;  // datasheet-true stationary bias sigma

  // Gauss-Markov bias correlation times [s], independent per channel (only used
  // for --bias gm; ConstantBias keeps a fixed reference). PLACEHOLDER defaults
  // pending the consistency/RMSE tau sweep -- the STIM300's true correlation
  // times are long and not resolvable from the 224 s flight-static window, so
  // set these via --bias-tau-{acc,gyro}.
  double bias_tau_acc = 3600.0;
  double bias_tau_gyro = 3600.0;

  // SE_2(3)-only knobs (ignored for --preint se3).
  gtsam::SE23CovarianceMethod cov_method =
      gtsam::SE23CovarianceMethod::Brossard;
  gtsam::SE23IncrementModel increment =
      gtsam::SE23IncrementModel::SimpleGlobalAcc;
};

static void print_usage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " [options]\n"
      << "  --preint {se3|se23}         state parameterisation (default se3)\n"
      << "  --bias   {cb|gm}            bias model (default cb)\n"
      << "  --bias-tau-acc <s>         GM accel-bias correlation time "
         "(default 3600)\n"
      << "  --bias-tau-gyro <s>        GM gyro-bias correlation time "
         "(default 3600); gm only\n"
      << "  --aiding {gnss|pars|none}   aiding source (default gnss);\n"
      << "                              none = IMU-only dead reckoning\n"
      << "  --dr-start <s>              none: GNSS bootstrap until <s>, then\n"
      << "                              IMU-only dead reckoning (default 60)\n"
      << "  --dr-horizon <s>            none: end the run <s> after dr-start\n"
      << "                              (default inf = to end of data)\n"
      << "  --switch-time <s>           pars: GNSS bootstrap until <s>\n"
      << "                              then switch to pars (default 400)\n"
      << "  --robust {none|gm|tukey}    robust kernel (default none)\n"
      << "  --covmethod {brossard|ours|vanloan}  se23 process-noise method\n"
      << "                              (default brossard)\n"
      << "  --increment {simple|full}   se23 increment model (default simple)\n"
      << "  --base-path <path>          per-sensor CSV folder\n"
      << "  --output-dir <path>         result CSV directory\n"
      << "  --acc-noise-scaling <s>     (default 33)\n"
      << "  --gyro-noise-scaling <s>    (default 100)\n"
      << "  --bias-scaling <s>          inflate datasheet bias sigma "
         "(default 1)\n"
      << "  -h, --help\n";
}

static bool parse_args(int argc, char** argv, Options& o) {
  auto need = [&](int i, const char* flag) {
    if (i + 1 >= argc) {
      std::cerr << "Error: " << flag << " requires a value\n";
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
      if (v == "se23") o.use_se23 = true;
      else if (v == "se3" || v == "legacy") o.use_se23 = false;
      else { std::cerr << "Unknown --preint " << v << "\n"; return false; }
    } else if (a == "--bias") {
      if (!need(i, "--bias")) return false;
      std::string v = argv[++i];
      if (v == "gm") o.use_gm = true;
      else if (v == "cb") o.use_gm = false;
      else { std::cerr << "Unknown --bias " << v << "\n"; return false; }
    } else if (a == "--bias-tau-acc") {
      if (!need(i, "--bias-tau-acc")) return false;
      o.bias_tau_acc = std::stod(argv[++i]);
    } else if (a == "--bias-tau-gyro") {
      if (!need(i, "--bias-tau-gyro")) return false;
      o.bias_tau_gyro = std::stod(argv[++i]);
    } else if (a == "--aiding") {
      if (!need(i, "--aiding")) return false;
      std::string v = argv[++i];
      if (v == "gnss") o.aiding = Aiding::Gnss;
      else if (v == "pars") o.aiding = Aiding::Pars;
      else if (v == "none") o.aiding = Aiding::None;
      else { std::cerr << "Unknown --aiding " << v << "\n"; return false; }
    } else if (a == "--switch-time") {
      if (!need(i, "--switch-time")) return false;
      o.switch_time = std::stod(argv[++i]);
    } else if (a == "--dr-start") {
      if (!need(i, "--dr-start")) return false;
      o.dr_start = std::stod(argv[++i]);
    } else if (a == "--dr-horizon") {
      if (!need(i, "--dr-horizon")) return false;
      o.dr_horizon = std::stod(argv[++i]);
    } else if (a == "--df-elevation") {
      o.df_use_elevation = true;
    } else if (a == "--robust") {
      if (!need(i, "--robust")) return false;
      std::string v = argv[++i];
      if (v == "none") o.robust = Robust::None;
      else if (v == "gm") o.robust = Robust::GemanMcClure;
      else if (v == "tukey") o.robust = Robust::Tukey;
      else { std::cerr << "Unknown --robust " << v << "\n"; return false; }
    } else if (a == "--covmethod") {
      if (!need(i, "--covmethod")) return false;
      std::string v = argv[++i];
      if (v == "brossard") o.cov_method = gtsam::SE23CovarianceMethod::Brossard;
      else if (v == "ours") o.cov_method = gtsam::SE23CovarianceMethod::Ours;
      else if (v == "vanloan")
        o.cov_method = gtsam::SE23CovarianceMethod::VanLoan;
      else { std::cerr << "Unknown --covmethod " << v << "\n"; return false; }
    } else if (a == "--increment") {
      if (!need(i, "--increment")) return false;
      std::string v = argv[++i];
      if (v == "simple")
        o.increment = gtsam::SE23IncrementModel::SimpleGlobalAcc;
      else if (v == "full")
        o.increment = gtsam::SE23IncrementModel::ConstantBodyImu;
      else { std::cerr << "Unknown --increment " << v << "\n"; return false; }
    } else if (a == "--base-path") {
      if (!need(i, "--base-path")) return false;
      o.base_path = argv[++i];
    } else if (a == "--output-dir") {
      if (!need(i, "--output-dir")) return false;
      o.output_dir = argv[++i];
      if (!o.output_dir.empty() && o.output_dir.back() != '/')
        o.output_dir.push_back('/');
    } else if (a == "--acc-noise-scaling") {
      if (!need(i, "--acc-noise-scaling")) return false;
      o.acc_noise_scaling = std::stod(argv[++i]);
    } else if (a == "--gyro-noise-scaling") {
      if (!need(i, "--gyro-noise-scaling")) return false;
      o.gyro_noise_scaling = std::stod(argv[++i]);
    } else if (a == "--bias-scaling") {
      if (!need(i, "--bias-scaling")) return false;
      o.bias_scaling = std::stod(argv[++i]);
    } else {
      std::cerr << "Unknown arg: " << a << "\n";
      print_usage(argv[0]);
      return false;
    }
  }
  return true;
}

// ============================================================================
// CSV helpers
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
};

static std::optional<Data> load_data(const std::string& base_path) {
  Data d;
  for (const auto& r : read_csv(join(base_path, "time.csv")))
    if (!r.empty()) d.t.push_back(r[0]);
  if (d.t.empty()) {
    std::cerr << "Empty time.csv at " << base_path << "\n";
    return std::nullopt;
  }
  const size_t N = d.t.size();

  auto imu = read_csv(join(base_path, "imu.csv"));
  d.f_m.resize(N, Eigen::Vector3d::Zero());
  d.w_m.resize(N, Eigen::Vector3d::Zero());
  for (size_t i = 0; i < std::min(N, imu.size()); ++i)
    if (imu[i].size() >= 7) {
      d.f_m[i] << imu[i][1], imu[i][2], imu[i][3];
      d.w_m[i] << imu[i][4], imu[i][5], imu[i][6];
    }

  d.rtk_idx.assign(N, 0);
  d.z_rtk.assign(N, Eigen::Vector3d::Zero());
  auto rtk = read_csv(join(base_path, "rtk.csv"));
  for (size_t i = 0; i < std::min(N, rtk.size()); ++i)
    if (rtk[i].size() >= 4 && rtk[i][0] != 0.0) {
      d.rtk_idx[i] = 1;
      d.z_rtk[i] << rtk[i][1], rtk[i][2], rtk[i][3];
    }

  d.df_idx.assign(N, 0);
  d.df_azi.assign(N, 0.0);
  d.df_ele.assign(N, 0.0);
  auto df = read_csv(join(base_path, "df.csv"));
  for (size_t i = 0; i < std::min(N, df.size()); ++i)
    if (df[i].size() >= 3 && df[i][0] != 0.0) {
      d.df_idx[i] = 1;
      d.df_azi[i] = df[i][1];
      d.df_ele[i] = df[i][2];
    }

  d.cs_idx.assign(N, 0);
  d.cs_range.assign(N, 0.0);
  auto cs = read_csv(join(base_path, "cs.csv"));
  for (size_t i = 0; i < std::min(N, cs.size()); ++i)
    if (cs[i].size() >= 2 && cs[i][0] != 0.0) {
      d.cs_idx[i] = 1;
      d.cs_range[i] = cs[i][1];
    }

  return d;
}

// ============================================================================
// GPS factor on ExtendedPose3 (position-only)
// ============================================================================

class GPSFactorSE23 : public gtsam::NoiseModelFactorN<gtsam::Se23> {
  gtsam::Point3 measured_;

 public:
  GPSFactorSE23(gtsam::Key key, const gtsam::Point3& measured,
                const gtsam::SharedNoiseModel& model)
      : gtsam::NoiseModelFactorN<gtsam::Se23>(model, key),
        measured_(measured) {}

  gtsam::Vector evaluateError(const gtsam::Se23& X,
                              gtsam::OptionalMatrixType H) const override {
    if (H) {
      gtsam::Matrix H_p(3, 9);
      gtsam::Point3 p = X.x(1, H_p);
      *H = H_p;
      return p - measured_;
    }
    return X.x(1) - measured_;
  }
};

// Linear interpolation of the RTK position (truth) to an arbitrary time.
static Eigen::Vector3d interp_rtk(const std::vector<double>& rt,
                                  const std::vector<Eigen::Vector3d>& rp,
                                  double tq) {
  if (rt.empty()) return Eigen::Vector3d::Zero();
  if (tq <= rt.front()) return rp.front();
  if (tq >= rt.back()) return rp.back();
  auto it = std::upper_bound(rt.begin(), rt.end(), tq);
  size_t j = static_cast<size_t>(it - rt.begin());
  double a = (tq - rt[j - 1]) / (rt[j] - rt[j - 1]);
  return (1.0 - a) * rp[j - 1] + a * rp[j];
}

// ============================================================================
// Estimation (templated on bias + preintegrator family)
// ============================================================================

template <class BIAS, bool UseSE23>
void run_estimation(const Data& d, const Options& opts) {
  using PIMLegacy = gtsam::PreintegratedCombinedMeasurementsGMT<
      gtsam::ManifoldPreintegrationT<BIAS>, BIAS>;
  using FactorLegacy = gtsam::CombinedImuFactorGMT<PIMLegacy, BIAS>;
  using PIMSE23 = gtsam::PreintegratedCombinedMeasurements2T<
      gtsam::ManifoldPreintegrationSE23<BIAS>, BIAS>;
  using FactorSE23 = gtsam::CombinedImuFactor2T<PIMSE23, BIAS>;
  using PIM = std::conditional_t<UseSE23, PIMSE23, PIMLegacy>;
  using StateType = std::conditional_t<UseSE23, gtsam::Se23,
                                       gtsam::NavState>;
  using PoseParam = std::conditional_t<UseSE23, gtsam::Se23,
                                       gtsam::Pose3>;

  // --- IMU noise ---
  const double g0 = 9.80665;
  const double vrw = 0.07, arw = 0.15;

  // GM bias correlation times [s], INDEPENDENT per channel. Tunable for
  // GaussMarkovBias (--bias-tau-{acc,gyro}); fixed 3600 s reference for
  // ConstantBias (so tuning tau does not move the CB baseline). These feed BOTH
  // q_b (= 2 sigma^2 / tau) and the GaussMarkovBias constructor.
  constexpr bool kIsGM = std::is_same_v<BIAS, gtsam::imuBias::GaussMarkovBias>;
  const double T_acc = kIsGM ? opts.bias_tau_acc : 3600.0;
  const double T_ars = kIsGM ? opts.bias_tau_gyro : 3600.0;

  // Stationary bias variance from the STIM300 datasheet Allan floor:
  //   B = ASD_min / 0.664 ,  sigma_b^2 = 2 * B^2 * ln(2) / pi   (bias instab.).
  // ASD_min: gyro 0.325 deg/h (Fig 6-5), accel 0.04 mg (Fig 6-13, 10g unit).
  const double mg = g0 * 1e-3;
  const double B_gyro = 0.325 / 0.664;  // deg/h
  const double B_acc = 0.04 / 0.664;    // mg
  const double sig2_bg = 2.0 * B_gyro * B_gyro * std::log(2.0) / M_PI *
                         std::pow(deg2rad(1.0 / 3600.0), 2.0);  // (rad/s)^2
  const double sig2_ba = 2.0 * B_acc * B_acc * std::log(2.0) / M_PI *
                         (mg * mg);  // (m/s^2)^2

  const double q_v = std::pow(opts.acc_noise_scaling * vrw / 60.0, 2.0);
  const double q_o =
      std::pow((opts.gyro_noise_scaling * arw / 60.0) * deg2rad(1.0), 2.0);
  // Driving PSD q = 2 sigma^2 / tau; bias_scaling inflates the stationary sigma
  // (so variance scales by bias_scaling^2). With the datasheet sigma above, the
  // physically-true run is --bias-scaling 1.
  const double q_b_v =
      (2.0 / T_acc) * std::pow(opts.bias_scaling, 2.0) * sig2_ba;
  const double q_b_o =
      (2.0 / T_ars) * std::pow(opts.bias_scaling, 2.0) * sig2_bg;

  auto p = gtsam::PreintegrationCombinedParamsT<BIAS>::MakeSharedD(g0);
  p->accelerometerCovariance = gtsam::I_3x3 * q_v;
  p->gyroscopeCovariance = gtsam::I_3x3 * q_o;
  p->integrationCovariance = gtsam::I_3x3 * 1e-8;
  p->biasAccCovariance = gtsam::I_3x3 * q_b_v;
  p->biasOmegaCovariance = gtsam::I_3x3 * q_b_o;

  // --- Measurement noise ---
  // DTU GNSS is DGPS (~metre), not RTK. The otter position is a differenced
  // baseline (common-mode error cancels) but still ~metre-level, so use a looser
  // position sigma than run_bledar's RTK values. Tune against the data.
  gtsam::Matrix3 R_GNSS_pos = gtsam::I_3x3;
  R_GNSS_pos(0, 0) = 2.0 * 2.0;
  R_GNSS_pos(1, 1) = 2.0 * 2.0;
  R_GNSS_pos(2, 2) = 4.0 * 4.0;
  auto gnss_noise = gtsam::noiseModel::Diagonal::Variances(R_GNSS_pos.diagonal());

  auto pars_azi_base = gtsam::noiseModel::Isotropic::Sigma(1, deg2rad(5.0));
  auto pars_ele_base = gtsam::noiseModel::Isotropic::Sigma(1, deg2rad(5.0));
  auto pars_range_base = gtsam::noiseModel::Isotropic::Sigma(1, 3.5);

  const double k_az = (opts.robust == Robust::Tukey) ? 4.6851 : 0.9;
  const double k_el = (opts.robust == Robust::Tukey) ? 4.6851 : 3.0;
  const double k_ra = (opts.robust == Robust::Tukey) ? 4.6851 : 0.9;
  auto pars_azi_noise = wrap_robust(pars_azi_base, opts.robust, k_az);
  auto pars_ele_noise = wrap_robust(pars_ele_base, opts.robust, k_el);
  auto pars_range_noise = wrap_robust(pars_range_base, opts.robust, k_ra);

  // --- PARS / radio geometry (NED <- radio) ---
  // R_rn = R_radio_ned from parnav-ros-modules/matlab/combined_flight_analysis.m
  // (maps radio-frame directions -> NED; t_rn=0, only rotation was estimated).
  // Confirmed physical mount: bearing 150.8 deg, tilt -0.1 deg, roll 136.2 deg
  // (= ~45 deg mod the square array's 90 deg symmetry); free==constrained fit.
  Eigen::Matrix3d R_rn;
  R_rn << 0.3543, -0.8725, -0.3363, 0.6294, 0.4885, -0.6043, 0.6916, 0.0024,
      0.7223;
  Eigen::Vector3d t_rn(0.0, 0.0, 0.0);

  // --- Initial state ---
  gtsam::Rot3 R0 = gtsam::Rot3::Identity();
  gtsam::Point3 p0 = gtsam::Point3::Zero();
  gtsam::Vector3 v0 = gtsam::Vector3::Zero();
  gtsam::Pose3 pose0(R0, p0);
  const double A_pos = 2.5, A_vel = 0.5, A_att_rp = deg2rad(180.0);
  const double A_pos_z = 5.0, A_yaw = deg2rad(180.0);
  const double A_acc_bias = 0.1, A_gyro_bias = deg2rad(0.5);

  BIAS prior_bias;
  if constexpr (std::is_same_v<BIAS, gtsam::imuBias::GaussMarkovBias>) {
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

  if constexpr (UseSE23) {
    auto epose_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(9) << A_att_rp, A_att_rp, A_yaw, A_vel, A_vel, A_vel,
         A_pos, A_pos, A_pos_z)
            .finished());
    gtsam::Se23 ext0(R0, v0, p0);
    graph.addPrior<gtsam::Se23>(X(0), ext0, epose_noise);
    graph.addPrior<BIAS>(B(0), prior_bias, bias_noise);
    values.insert(X(0), ext0);
    values.insert(B(0), prior_bias);
    timestamps[X(0)] = 0.0;
    timestamps[B(0)] = 0.0;
  } else {
    auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << A_att_rp, A_att_rp, A_yaw, A_pos, A_pos, A_pos_z)
            .finished());
    auto vel_noise = gtsam::noiseModel::Isotropic::Sigma(3, A_vel);
    graph.addPrior<gtsam::Pose3>(X(0), pose0, pose_noise);
    graph.addPrior<gtsam::Vector3>(V(0), v0, vel_noise);
    graph.addPrior<BIAS>(B(0), prior_bias, bias_noise);
    values.insert(X(0), pose0);
    values.insert(V(0), v0);
    values.insert(B(0), prior_bias);
    timestamps[X(0)] = 0.0;
    timestamps[V(0)] = 0.0;
    timestamps[B(0)] = 0.0;
  }
  smoother.update(graph, values, timestamps);

  // --- RTK truth samples (for the position error) ---
  std::vector<double> rtk_t;
  std::vector<Eigen::Vector3d> rtk_p;
  for (size_t i = 0; i < d.t.size(); ++i)
    if (d.rtk_idx[i]) {
      rtk_t.push_back(d.t[i]);
      rtk_p.push_back(d.z_rtk[i]);
    }

  // --- Loop ---
  std::shared_ptr<PIM> preintegrated;
  if constexpr (UseSE23) {
    preintegrated = std::make_shared<PIM>(
        p, prior_bias, Eigen::Matrix<double, 15, 15>::Zero(), opts.increment,
        opts.cov_method);
  } else {
    preintegrated = std::make_shared<PIM>(p, prior_bias);
  }
  StateType prev_state = [&] {
    if constexpr (UseSE23) return gtsam::Se23(R0, v0, p0);
    else return gtsam::NavState(pose0, v0);
  }();
  BIAS prev_bias = prior_bias;

  std::vector<gtsam::Rot3> est_R{R0};
  std::vector<Eigen::Vector3d> est_p{p0}, est_v{v0};
  std::vector<Eigen::Vector3d> pos_sigma{Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> vel_sigma{Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> att_sigma{Eigen::Vector3d::Zero()};

  // Propagated variance anchors (grow sigma while coasting on IMU).
  Eigen::Vector3d last_pos_var = Eigen::Vector3d::Zero();
  Eigen::Vector3d last_vel_var = Eigen::Vector3d::Zero();
  Eigen::Vector3d last_att_var = Eigen::Vector3d::Zero();

  // preintMeasCov diagonal block layout: legacy [theta,pos,vel]; se23
  // [theta,nu(vel),rho(pos)].
  auto propagated_sigmas = [&](Eigen::Vector3d& ps, Eigen::Vector3d& vs,
                               Eigen::Vector3d& as) {
    const auto C = preintegrated->preintMeasCov();
    const Eigen::Vector3d att_v = C.template block<3, 3>(0, 0).diagonal();
    Eigen::Vector3d pos_v, vel_v;
    if constexpr (UseSE23) {
      vel_v = C.template block<3, 3>(3, 3).diagonal();
      pos_v = C.template block<3, 3>(6, 6).diagonal();
    } else {
      pos_v = C.template block<3, 3>(3, 3).diagonal();
      vel_v = C.template block<3, 3>(6, 6).diagonal();
    }
    for (int k = 0; k < 3; ++k) {
      ps[k] = std::sqrt(last_pos_var[k] + pos_v[k]);
      vs[k] = std::sqrt(last_vel_var[k] + vel_v[k]);
      as[k] = std::sqrt(last_att_var[k] + att_v[k]);
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

      // Dead-reckoning horizon: in aiding=none, end the run dr_horizon seconds
      // after aiding stops, for a clean fixed-length DR segment (default: to end).
      if (opts.aiding == Aiding::None &&
          accumulated_time > opts.dr_start + opts.dr_horizon)
        break;

      // Which measurements fire this tick.
      //   gnss : position aiding whenever a fix is present.
      //   pars : bootstrap GNSS until switch_time, then DF/CS.
      //   none : bootstrap GNSS until dr_start, then NO aiding — the has_meas
      //          =false branch below coasts on the IMU (predict + growing
      //          preintMeasCov), i.e. dead reckoning with propagated uncertainty.
      bool use_gnss = false, use_bt = false;
      if (opts.aiding == Aiding::Gnss) {
        use_gnss = d.rtk_idx[idx] != 0;
      } else if (opts.aiding == Aiding::None) {
        use_gnss = (accumulated_time < opts.dr_start) && (d.rtk_idx[idx] != 0);
      } else {  // Aiding::Pars
        if (accumulated_time < opts.switch_time)
          use_gnss = d.rtk_idx[idx] != 0;
        else
          use_bt = (d.df_idx[idx] || d.cs_idx[idx]);
      }

      const bool has_meas = use_gnss || use_bt;
      if (!has_meas) {
        // Coast: log the predicted state with propagated uncertainty.
        Eigen::Vector3d ps, vs, as;
        propagated_sigmas(ps, vs, as);
        if constexpr (UseSE23) {
          gtsam::Se23 prop =
              preintegrated->predict(prev_state, prev_bias);
          est_R.push_back(prop.rotation());
          est_p.push_back(prop.x(1));
          est_v.push_back(prop.x(0));
        } else {
          gtsam::NavState prop = preintegrated->predict(prev_state, prev_bias);
          est_R.push_back(prop.pose().rotation());
          est_p.push_back(prop.pose().translation());
          est_v.push_back(prop.v());
        }
        pos_sigma.push_back(ps);
        vel_sigma.push_back(vs);
        att_sigma.push_back(as);
        continue;
      }
      correction_count++;

      graph.resize(0);
      values.clear();
      timestamps.clear();

      // IMU factor
      if constexpr (UseSE23) {
        graph.add(FactorSE23(X(correction_count - 1), X(correction_count),
                             B(correction_count - 1), B(correction_count),
                             *preintegrated));
      } else {
        graph.add(FactorLegacy(X(correction_count - 1), V(correction_count - 1),
                              X(correction_count), V(correction_count),
                              B(correction_count - 1), B(correction_count),
                              *preintegrated));
      }

      // Predict + insert
      if constexpr (UseSE23) {
        gtsam::Se23 prop =
            preintegrated->predict(prev_state, prev_bias);
        values.insert(X(correction_count), prop);
        values.insert(B(correction_count), prev_bias);
        timestamps[X(correction_count)] = t_now;
        timestamps[B(correction_count)] = t_now;
      } else {
        gtsam::NavState prop = preintegrated->predict(prev_state, prev_bias);
        values.insert(X(correction_count), prop.pose());
        values.insert(V(correction_count), prop.v());
        values.insert(B(correction_count), prev_bias);
        timestamps[X(correction_count)] = t_now;
        timestamps[V(correction_count)] = t_now;
        timestamps[B(correction_count)] = t_now;
      }

      // Aiding factors
      if (use_gnss) {
        if constexpr (UseSE23)
          graph.add(GPSFactorSE23(X(correction_count), d.z_rtk[idx], gnss_noise));
        else
          graph.add(gtsam::GPSFactor(X(correction_count), d.z_rtk[idx],
                                     gnss_noise));
      }
      if (use_bt) {
        if (d.df_idx[idx]) {
          graph.add(parnav::AzimuthFactor<PoseParam>(
              X(correction_count), pars_azi_noise, d.df_azi[idx], t_rn, R_rn));
          if (opts.df_use_elevation)
            graph.add(parnav::ElevationFactor<PoseParam>(
                X(correction_count), pars_ele_noise, d.df_ele[idx], t_rn, R_rn));
        }
        if (d.cs_idx[idx]) {
          graph.add(parnav::RangeFactor<PoseParam>(
              X(correction_count), pars_range_noise, d.cs_range[idx], t_rn,
              R_rn));
        }
      }

      smoother.update(graph, values, timestamps);
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
        prev_state =
            gtsam::NavState(result.at<gtsam::Pose3>(X(correction_count)),
                            result.at<gtsam::Vector3>(V(correction_count)));
        R_est = prev_state.pose().rotation();
        p_est = prev_state.pose().translation();
        v_est = prev_state.v();
      }
      prev_bias = result.at<BIAS>(B(correction_count));
      preintegrated->resetIntegrationAndSetBias(prev_bias);

      est_R.push_back(R_est);
      est_p.push_back(p_est);
      est_v.push_back(v_est);

      Eigen::Vector3d sp = Eigen::Vector3d::Zero();
      Eigen::Vector3d sv = Eigen::Vector3d::Zero();
      Eigen::Vector3d sa = Eigen::Vector3d::Zero();
      try {
        gtsam::Matrix Cx = smoother.marginalCovariance(X(correction_count));
        if constexpr (UseSE23) {
          // ExtendedPose3 9x9: [theta(0-2), nu/vel(3-5), rho/pos(6-8)]
          sa << std::sqrt(Cx(0, 0)), std::sqrt(Cx(1, 1)), std::sqrt(Cx(2, 2));
          sv << std::sqrt(Cx(3, 3)), std::sqrt(Cx(4, 4)), std::sqrt(Cx(5, 5));
          sp << std::sqrt(Cx(6, 6)), std::sqrt(Cx(7, 7)), std::sqrt(Cx(8, 8));
          last_att_var << Cx(0, 0), Cx(1, 1), Cx(2, 2);
          last_vel_var << Cx(3, 3), Cx(4, 4), Cx(5, 5);
          last_pos_var << Cx(6, 6), Cx(7, 7), Cx(8, 8);
        } else {
          gtsam::Matrix Cv = smoother.marginalCovariance(V(correction_count));
          sa << std::sqrt(Cx(0, 0)), std::sqrt(Cx(1, 1)), std::sqrt(Cx(2, 2));
          sp << std::sqrt(Cx(3, 3)), std::sqrt(Cx(4, 4)), std::sqrt(Cx(5, 5));
          sv << std::sqrt(Cv(0, 0)), std::sqrt(Cv(1, 1)), std::sqrt(Cv(2, 2));
          last_att_var << Cx(0, 0), Cx(1, 1), Cx(2, 2);
          last_pos_var << Cx(3, 3), Cx(4, 4), Cx(5, 5);
          last_vel_var << Cv(0, 0), Cv(1, 1), Cv(2, 2);
        }
      } catch (const std::exception&) {
      }
      pos_sigma.push_back(sp);
      vel_sigma.push_back(sv);
      att_sigma.push_back(sa);

      if (correction_count % 100 == 0)
        printf("step %lld t=%.2f p=[%.2f %.2f %.2f]\n",
               static_cast<long long>(idx), t_now, p_est.x(), p_est.y(),
               p_est.z());
    } catch (const gtsam::IndeterminateSystemException& e) {
      fprintf(stderr, "Indeterminate at idx=%lld t=%.2f: %s (coasting)\n",
              static_cast<long long>(idx), d.t[idx], e.what());
      // This correction did NOT commit: X(correction_count) was never added to
      // the smoother. Roll the counter back so the next correction reconnects
      // from the last committed state, and DO NOT reset the preintegration —
      // keep accumulating so that correction's IMU factor spans the full
      // interval. i.e. treat an indeterminate update exactly like a coast step.
      correction_count--;
      Eigen::Vector3d ps, vs, as;
      propagated_sigmas(ps, vs, as);
      if constexpr (UseSE23) {
        gtsam::Se23 prop =
            preintegrated->predict(prev_state, prev_bias);
        est_R.push_back(prop.rotation());
        est_p.push_back(prop.x(1));
        est_v.push_back(prop.x(0));
      } else {
        gtsam::NavState prop = preintegrated->predict(prev_state, prev_bias);
        est_R.push_back(prop.pose().rotation());
        est_p.push_back(prop.pose().translation());
        est_v.push_back(prop.v());
      }
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

  if (aborted_early)
    while (static_cast<int64_t>(est_p.size()) < N) {
      est_R.push_back(est_R.back());
      est_p.push_back(est_p.back());
      est_v.push_back(est_v.back());
      pos_sigma.push_back(pos_sigma.back());
      vel_sigma.push_back(vel_sigma.back());
      att_sigma.push_back(att_sigma.back());
    }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  printf("Final position: [%.4f, %.4f, %.4f]\n", est_p.back().x(),
         est_p.back().y(), est_p.back().z());
  printf("Elapsed: %.3f s | horizon: %.1f s | corrections: %d\n",
         elapsed.count(), accumulated_time, correction_count);

  // --- Write result CSV in the plot_3sigma.py column layout ---
  // Filename carries the aiding + bias + state tags so runs don't clobber each
  // other, e.g. dtu_gnss_cb.csv, dtu_none_cb.csv, dtu_pars_cb_se23.csv.
  const char* aiding_tag = opts.aiding == Aiding::Gnss   ? "gnss"
                           : opts.aiding == Aiding::Pars ? "pars"
                                                         : "none";
  const std::string bias_tag =
      std::is_same_v<BIAS, gtsam::imuBias::GaussMarkovBias> ? "gm" : "cb";
  const std::string se23_tag = UseSE23 ? "_se23" : "";
  const std::string out_file = opts.output_dir + "dtu_" + aiding_tag + "_" +
                               bias_tag + se23_tag + ".csv";
  std::ofstream out(out_file);
  if (!out) {
    std::cerr << "Could not open output " << out_file << "\n";
    return;
  }
  out << "t,pos_n,pos_e,pos_d,vel_n,vel_e,vel_d,roll,pitch,yaw,"
      << "pos_err_n,pos_err_e,pos_err_d,vel_err_n,vel_err_e,vel_err_d,"
      << "att_err_roll,att_err_pitch,att_err_yaw,"
      << "acc_bias_err_x,acc_bias_err_y,acc_bias_err_z,"
      << "gyro_bias_err_x,gyro_bias_err_y,gyro_bias_err_z,"
      << "sig3_roll,sig3_pitch,sig3_yaw,"
      << "sig3_pos_n,sig3_pos_e,sig3_pos_d,"
      << "sig3_vel_n,sig3_vel_e,sig3_vel_d,"
      << "sig3_ab_x,sig3_ab_y,sig3_ab_z,"
      << "sig3_gb_x,sig3_gb_y,sig3_gb_z\n";

  const size_t n_rows = std::min(est_p.size(), static_cast<size_t>(N));
  for (size_t i = 0; i < n_rows; ++i) {
    const auto& ep = est_p[i];
    const auto& ev = est_v[i];
    const auto rpy = Eigen::Vector3d(est_R[i].roll(), est_R[i].pitch(),
                                     est_R[i].yaw());
    // Position error vs the RTK truth interpolated to this timestamp.
    const Eigen::Vector3d rtk = interp_rtk(rtk_t, rtk_p, d.t[i]);
    const Eigen::Vector3d pe = ep - rtk;
    const auto& s3p = pos_sigma[i];
    const auto& s3v = vel_sigma[i];
    const auto& s3a = att_sigma[i];
    char buf[1024];
    snprintf(buf, sizeof(buf),
             "%.6f,"
             "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
             "%.6f,%.6f,%.6f,0,0,0,"    // pos_err, vel_err(0)
             "0,0,0,"                    // att_err(0)
             "0,0,0,0,0,0,"              // bias_err(0)
             "%.8f,%.8f,%.8f,"           // sig3 att
             "%.8f,%.8f,%.8f,"           // sig3 pos
             "%.8f,%.8f,%.8f,"           // sig3 vel
             "0,0,0,0,0,0\n",            // sig3 bias(0)
             d.t[i], ep.x(), ep.y(), ep.z(), ev.x(), ev.y(), ev.z(), rpy.x(),
             rpy.y(), rpy.z(), pe.x(), pe.y(), pe.z(), 3.0 * s3a.x(),
             3.0 * s3a.y(), 3.0 * s3a.z(), 3.0 * s3p.x(), 3.0 * s3p.y(),
             3.0 * s3p.z(), 3.0 * s3v.x(), 3.0 * s3v.y(), 3.0 * s3v.z());
    out << buf;
  }
  printf("Saved results to %s\n", out_file.c_str());
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  Options opts;
  if (!parse_args(argc, argv, opts)) return 1;

  const char* aiding_name = opts.aiding == Aiding::Gnss   ? "GNSS (DGPS)"
                            : opts.aiding == Aiding::Pars ? "PARS (DF+CS)"
                                                          : "NONE (dead-reckon)";
  if (opts.aiding == Aiding::None)
    printf("Dead reckoning: GNSS bootstrap until %.0f s, then IMU-only%s\n",
           opts.dr_start,
           std::isinf(opts.dr_horizon)
               ? " to end of data"
               : (" for " + std::to_string((int)opts.dr_horizon) + " s").c_str());
  printf("Base:    %s\n", opts.base_path.c_str());
  printf("Mode:    %s | %s | aiding=%s\n",
         opts.use_se23 ? "SE_2(3)" : "SE(3)", opts.use_gm ? "GM bias" : "CB bias",
         aiding_name);

  auto data = load_data(opts.base_path);
  if (!data) return 1;
  printf("Loaded %zu samples\n", data->t.size());

  try {
    using CB = gtsam::imuBias::ConstantBias;
    using GM = gtsam::imuBias::GaussMarkovBias;
    if (opts.use_se23 && opts.use_gm)
      run_estimation<GM, true>(*data, opts);
    else if (opts.use_se23)
      run_estimation<CB, true>(*data, opts);
    else if (opts.use_gm)
      run_estimation<GM, false>(*data, opts);
    else
      run_estimation<CB, false>(*data, opts);
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 2;
  }
  return 0;
}
