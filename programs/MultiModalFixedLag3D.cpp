/**
 * @file MultiModalFixedLag3D.cpp
 * @brief 3D fixed-lag smoother for the multi-modale-simulator export.
 *
 * Consumes `simulation_results_3d.npz` + `simulation_results_3d.meta.json`
 * produced by `f_export_3d.py`. The simulator is planar; we lift it into
 * SO(3)/R^3 and pin (z, roll, pitch) via heterogeneous Pose3 priors so the
 * standard GTSAM CombinedImuFactor + GPSFactor pipeline can run unchanged.
 *
 * Out of scope: data association for polar/camera detections, robust kernels,
 * trust model. This is a port-validation harness, not a port of the Python
 * factor-graph frontend.
 */

#include "CompassFactor.hpp"
#include "MarkerAssociation.hpp"
#include "MarkerAzimuthFactor.hpp"
#include "MultiModalSimLoader.hpp"
#include "RangeFactor.hpp"
#include "SelfTrust.hpp"
#include "utils.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

namespace {

struct Args {
  std::string sim_data;
  std::string meta;
  int ship_index = 0;
  double lag = 5.0;
  std::string out_csv = "multimodal_estimates.csv";
  std::string trust_csv;              // empty -> derived from out_csv
  std::string robust_pos = "none";   // none | huber | tukey | gmc
  double robust_pos_k = 1.345;
  std::string robust_yaw = "none";
  double robust_yaw_k = 1.345;
  std::string robust_polar = "gmc";  // default: robustify marker factors.
                                     // Mis-associations/jamming produce outlier
                                     // detections; GMC downweights them so good
                                     // markers still anchor pose (critical during
                                     // GNSS blackout). --robust-polar none to off.
  double robust_polar_k = 1.345;
  int use_landmarks = 1;              // 1 on (default), 0 off
  int use_shoreline = 0;              // 0 off (default): shoreline association
                                      // is weak and corrupts yaw; opt in with
                                      // --shoreline once the frontend is fixed.
  int use_camera = 0;                 // 0 off (default): bearing-only camera
                                      // factors; opt in with --camera.
  double assoc_gate_chi2 = 5.99;      // 2-DoF χ² @ 95%
  // Trust overrides (sentinel: empty / NaN means "use sidecar value")
  int trust_enable = -1;              // -1 keep, 0 force off, 1 force on
  std::string trust_scaling;
  double trust_floor = std::nan("");
  double trust_alpha1 = std::nan("");
  double trust_alpha2 = std::nan("");
  double trust_pos_thresh = std::nan("");
  double trust_hdg_thresh = std::nan("");
  double trust_robust_k_mult = std::nan("");
  int trust_gnss_veto = -1;
};

Args parseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc)
        throw std::runtime_error("missing value for " + k);
      return argv[++i];
    };
    if (k == "--sim-data") a.sim_data = next();
    else if (k == "--meta") a.meta = next();
    else if (k == "--ship-index") a.ship_index = std::stoi(next());
    else if (k == "--lag") a.lag = std::stod(next());
    else if (k == "--out") a.out_csv = next();
    else if (k == "--robust-pos") a.robust_pos = next();
    else if (k == "--robust-pos-k") a.robust_pos_k = std::stod(next());
    else if (k == "--robust-yaw") a.robust_yaw = next();
    else if (k == "--robust-yaw-k") a.robust_yaw_k = std::stod(next());
    else if (k == "--trust-csv") a.trust_csv = next();
    else if (k == "--trust-enable") a.trust_enable = 1;
    else if (k == "--no-trust") a.trust_enable = 0;
    else if (k == "--trust-scaling") a.trust_scaling = next();
    else if (k == "--trust-floor") a.trust_floor = std::stod(next());
    else if (k == "--trust-alpha1") a.trust_alpha1 = std::stod(next());
    else if (k == "--trust-alpha2") a.trust_alpha2 = std::stod(next());
    else if (k == "--trust-gnss-pos-thresh") a.trust_pos_thresh = std::stod(next());
    else if (k == "--trust-gnss-hdg-thresh") a.trust_hdg_thresh = std::stod(next());
    else if (k == "--trust-robust-k-mult") a.trust_robust_k_mult = std::stod(next());
    else if (k == "--trust-gnss-veto") a.trust_gnss_veto = 1;
    else if (k == "--no-trust-gnss-veto") a.trust_gnss_veto = 0;
    else if (k == "--robust-polar") a.robust_polar = next();
    else if (k == "--robust-polar-k") a.robust_polar_k = std::stod(next());
    else if (k == "--no-landmarks") a.use_landmarks = 0;
    else if (k == "--landmarks") a.use_landmarks = 1;
    else if (k == "--no-shoreline") a.use_shoreline = 0;
    else if (k == "--shoreline") a.use_shoreline = 1;
    else if (k == "--no-camera") a.use_camera = 0;
    else if (k == "--camera") a.use_camera = 1;
    else if (k == "--assoc-gate-chi2") a.assoc_gate_chi2 = std::stod(next());
    else if (k == "-h" || k == "--help") {
      std::cout
          << "MultiModalFixedLag3D --sim-data PATH --meta PATH "
          << "[--ship-index 0] [--lag 5.0] [--out estimates.csv]\n"
          << "  [--robust-pos none|huber|tukey|gmc] [--robust-pos-k K]\n"
          << "  [--robust-yaw none|huber|tukey|gmc] [--robust-yaw-k K]\n"
          << "  [--trust-enable | --no-trust] [--trust-csv PATH]\n"
          << "  [--trust-scaling inverse|inverse_sqrt|linear|off]\n"
          << "  [--trust-floor F] [--trust-alpha1 A] [--trust-alpha2 A]\n"
          << "  [--trust-gnss-pos-thresh CHI2] [--trust-gnss-hdg-thresh CHI2]\n"
          << "  [--trust-robust-k-mult K] [--trust-gnss-veto | --no-trust-gnss-veto]\n"
          << "  [--landmarks | --no-landmarks] [--shoreline | --no-shoreline]\n"
          << "  [--camera | --no-camera] [--assoc-gate-chi2 5.99]\n"
          << "  [--robust-polar none|huber|tukey|gmc] [--robust-polar-k K]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown flag: " + k);
    }
  }
  if (a.sim_data.empty() || a.meta.empty())
    throw std::runtime_error("--sim-data and --meta are required");
  return a;
}

gtsam::SharedNoiseModel wrapRobust(const std::string& kind, double k,
                                   const gtsam::SharedNoiseModel& base) {
  if (kind == "none" || kind.empty()) return base;
  using gtsam::noiseModel::mEstimator::GemanMcClure;
  using gtsam::noiseModel::mEstimator::Huber;
  using gtsam::noiseModel::mEstimator::Tukey;
  using gtsam::noiseModel::Robust;
  if (kind == "huber") return Robust::Create(Huber::Create(k), base);
  if (kind == "tukey") return Robust::Create(Tukey::Create(k), base);
  if (kind == "gmc")   return Robust::Create(GemanMcClure::Create(k), base);
  throw std::runtime_error("unknown robust kernel: " + kind);
}


double wrapPi(double a) {
  constexpr double pi = 3.14159265358979323846;
  while (a > pi) a -= 2 * pi;
  while (a < -pi) a += 2 * pi;
  return a;
}

// Apply CLI overrides on top of the YAML-derived TrustConfig from the sidecar.
parnav::TrustConfig effectiveTrust(const parnav::TrustConfig& base,
                                   const Args& a) {
  parnav::TrustConfig t = base;
  if (a.trust_enable == 0) t.enable = false;
  if (a.trust_enable == 1) t.enable = true;
  if (!a.trust_scaling.empty()) t.scaling = a.trust_scaling;
  if (!std::isnan(a.trust_floor)) t.floor = a.trust_floor;
  if (!std::isnan(a.trust_alpha1)) t.alpha1 = a.trust_alpha1;
  if (!std::isnan(a.trust_alpha2)) t.alpha2 = a.trust_alpha2;
  if (!std::isnan(a.trust_pos_thresh)) t.gnss_pos_thresh = a.trust_pos_thresh;
  if (!std::isnan(a.trust_hdg_thresh)) t.gnss_hdg_thresh = a.trust_hdg_thresh;
  if (!std::isnan(a.trust_robust_k_mult))
    t.robust_k_mult = a.trust_robust_k_mult;
  if (a.trust_gnss_veto == 0) t.gnss_veto = false;
  if (a.trust_gnss_veto == 1) t.gnss_veto = true;
  return t;
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parseArgs(argc, argv);

  parnav::SimMeta meta = parnav::loadSimMeta(args.meta);
  parnav::SimData3D d = parnav::loadSimData(args.sim_data, meta);

  if (args.ship_index < 0 || args.ship_index >= d.n_ships)
    throw std::runtime_error("ship-index out of range");
  const int ship = args.ship_index;
  const int T = d.T;
  const double g0 = meta.gravity;

  // ---- Noise models ----
  // GNSS xy/yaw pulled from sensor metadata; z held tight (z=0 is exact in the
  // lifted scenario). Position and yaw are fully decoupled — separate base
  // models, separate (optional) robust wrappers, separate validity gates.
  double sigma_xy = 0.5;
  double sigma_yaw = 0.05;
  for (const auto& g : meta.gnss) {
    if (g.ship == ship || g.ship == -1) {
      if (g.noise_xy > 0.0) sigma_xy = g.noise_xy;
      if (g.noise_heading > 0.0) sigma_yaw = g.noise_heading;
      break;
    }
  }
  auto gnss_pos_base = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(3) << sigma_xy, sigma_xy, 0.05).finished());
  auto gnss_yaw_base = gtsam::noiseModel::Isotropic::Sigma(1, sigma_yaw);

  // ---- Trust model ----
  parnav::TrustConfig trust_cfg = effectiveTrust(meta.trust, args);
  parnav::TrustScalingCfg scale_cfg{trust_cfg.scaling, trust_cfg.floor,
                                    trust_cfg.linear_k};
  parnav::SelfTrust trust(trust_cfg.alpha1, trust_cfg.alpha2);

  // Per-GNSS-sensor trust keys for this ship.
  struct GnssKey {
    int sensor_idx;
    std::string pos_key;
    std::string hdg_key;
  };
  std::vector<GnssKey> gnss_keys;
  for (int s = 0; s < d.n_gnss; ++s) {
    const auto& gm = meta.gnss[s];
    if (gm.ship != ship && gm.ship != -1) continue;
    gnss_keys.push_back({s, gm.name + "_POS", gm.name + "_HDG"});
  }

  // Per-Polar-sensor cached params + trust key.
  struct PolarSensor {
    int sensor_idx;
    std::string key;
    double yaw_offset_rad;
    double sigma_range;
    double sigma_az_rad;
  };
  std::vector<PolarSensor> polar_sensors;
  for (int s = 0; s < d.n_polar; ++s) {
    const auto& pm = meta.polar[s];
    // Strict per-ship match: a land-mounted polar sensor (ship == -1)
    // observes from a stationary land position, so its detections cannot be
    // tied to *this* ship's pose via MarkerAzimuthFactor/RangeFactor without
    // a separate model. Excluded here.
    if (pm.ship != ship) continue;
    polar_sensors.push_back({s, pm.name,
                             parnav::deg2rad(pm.relative_pose[2]),
                             pm.range_noise,
                             parnav::deg2rad(pm.angle_noise_deg)});
  }

  // Per-Camera-sensor cached params + trust key (bearing-only, no range).
  struct CameraSensor {
    int sensor_idx;
    std::string key;
    double yaw_offset_rad;
    double sigma_az_rad;
  };
  std::vector<CameraSensor> camera_sensors;
  for (int s = 0; s < d.n_camera; ++s) {
    const auto& cm = meta.camera[s];
    if (cm.ship != ship) continue;   // land cameras excluded (see polar note)
    camera_sensors.push_back({s, cm.name,
                              parnav::deg2rad(cm.relative_pose[2]),
                              parnav::deg2rad(cm.angle_noise_deg)});
  }

  auto build_gnss_pos_noise = [&](double trust_value) {
    double k = trust_cfg.enable ? parnav::trustScale(trust_value, scale_cfg)
                                : 1.0;
    auto base = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(3) << k * sigma_xy, k * sigma_xy, 0.05).finished());
    return wrapRobust(args.robust_pos, args.robust_pos_k, base);
  };
  auto build_gnss_yaw_noise = [&](double trust_value) {
    double k = trust_cfg.enable ? parnav::trustScale(trust_value, scale_cfg)
                                : 1.0;
    auto base = gtsam::noiseModel::Isotropic::Sigma(1, k * sigma_yaw);
    return wrapRobust(args.robust_yaw, args.robust_yaw_k, base);
  };

  // Heterogeneous Pose3 prior pinning (z, roll, pitch) tightly while leaving
  // (x, y, yaw) loose. Order is (rx, ry, rz, tx, ty, tz) per GTSAM's Pose3
  // tangent convention.
  const double sig_loose = 1e6;
  const double sig_planar = 1e-3;
  auto planar_pin = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << sig_planar, sig_planar, sig_loose,
       sig_loose, sig_loose, sig_planar).finished());

  // Initial-state prior. GT never enters the FGO: pose0 is seeded from the
  // first GNSS fix, so the position/yaw prior is set at GNSS confidence while
  // (z, roll, pitch) stay pinned to the planar manifold. Velocity is a rough
  // first-difference of GNSS, kept loose so it does not over-constrain.
  auto pose_prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << 1e-3, 1e-3, sigma_yaw, sigma_xy, sigma_xy, 1e-3)
          .finished());
  auto vel_prior_noise = gtsam::noiseModel::Isotropic::Sigma(3, 2.0);
  auto bias_prior_noise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);

  // ---- IMU preintegration params ----
  double sigma_acc = 0.05;       // m/s^2/sqrt(Hz) -- tunable; sim is noiseless
  double sigma_gyro = 0.005;     // rad/s/sqrt(Hz)
  double sigma_b_acc = 1e-4;
  double sigma_b_gyro = 1e-5;

  auto p = gtsam::PreintegrationCombinedParams::MakeSharedU(g0);
  p->accelerometerCovariance = gtsam::I_3x3 * sigma_acc * sigma_acc;
  p->gyroscopeCovariance = gtsam::I_3x3 * sigma_gyro * sigma_gyro;
  p->integrationCovariance = gtsam::I_3x3 * 1e-10;
  p->biasAccCovariance = gtsam::I_3x3 * sigma_b_acc * sigma_b_acc;
  p->biasOmegaCovariance = gtsam::I_3x3 * sigma_b_gyro * sigma_b_gyro;

  // ---- Fixed-lag smoother ----
  gtsam::ISAM2Params isam_params;
  // GTSAM 4.3.0 default CHOLESKY throws Indeterminate on the combined IMU+bias
  // system; QR is numerically robust (see SimulationFixedLagSmoother).
  isam_params.factorization = gtsam::ISAM2Params::QR;
  isam_params.relinearizeSkip = 1;
  isam_params.relinearizeThreshold = 0.01;
  isam_params.findUnusedFactorSlots = true;
  gtsam::IncrementalFixedLagSmoother smoother(args.lag, isam_params);

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;
  gtsam::FixedLagSmoother::KeyTimestampMap timestamps;

  // ---- t=0: priors + GNSS-derived initialization (no GT in the FGO) ----
  const int s0 = gnss_keys.empty() ? -1 : gnss_keys.front().sensor_idx;
  double x0 = 0.0, y0 = 0.0, yaw0 = 0.0;
  if (s0 >= 0) {
    for (int t = 0; t < T; ++t)
      if (d.gnssPosValid(s0, t)) { auto r = d.gnssPos(s0, t); x0 = r[0]; y0 = r[1]; break; }
    for (int t = 0; t < T; ++t)
      if (d.gnssYawValid(s0, t)) { yaw0 = d.gnssYaw(s0, t); break; }
  }
  gtsam::Pose3 pose0(gtsam::Rot3::Yaw(yaw0), gtsam::Point3(x0, y0, 0.0));

  // Velocity seed: first-difference of the first two valid GNSS fixes.
  gtsam::Vector3 v0(0.0, 0.0, 0.0);
  if (s0 >= 0) {
    int ta = -1, tb = -1;
    for (int t = 0; t < T; ++t)
      if (d.gnssPosValid(s0, t)) { if (ta < 0) ta = t; else { tb = t; break; } }
    if (ta >= 0 && tb > ta) {
      auto pa = d.gnssPos(s0, ta);
      auto pb = d.gnssPos(s0, tb);
      const double dtp = d.time[tb] - d.time[ta];
      if (dtp > 0.0)
        v0 = gtsam::Vector3((pb[0] - pa[0]) / dtp, (pb[1] - pa[1]) / dtp, 0.0);
    }
  }
  gtsam::imuBias::ConstantBias bias0;

  graph.addPrior<gtsam::Pose3>(X(0), pose0, pose_prior_noise);
  graph.addPrior<gtsam::Vector3>(V(0), v0, vel_prior_noise);
  graph.addPrior<gtsam::imuBias::ConstantBias>(B(0), bias0, bias_prior_noise);
  values.insert(X(0), pose0);
  values.insert(V(0), v0);
  values.insert(B(0), bias0);
  timestamps[X(0)] = d.time[0];
  timestamps[V(0)] = d.time[0];
  timestamps[B(0)] = d.time[0];

  smoother.update(graph, values, timestamps);
  graph.resize(0);
  values.clear();
  timestamps.clear();

  auto pim = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(p, bias0);

  gtsam::NavState prev_state(pose0, v0);
  gtsam::imuBias::ConstantBias prev_bias = bias0;

  // ---- Output CSV ----
  std::ofstream out(args.out_csv);
  out << "t,x,y,z,qw,qx,qy,qz,vx,vy,vz,gt_x,gt_y,gt_yaw\n";

  auto write_row = [&](double t, const gtsam::Pose3& P, const gtsam::Vector3& v,
                       int gt_t) {
    auto q = P.rotation().toQuaternion();
    auto pt = P.translation();
    auto gp = d.gtPose(ship, gt_t);
    double gt_yaw = 2.0 * std::atan2(gp[6], gp[3]);
    out << t << ',' << pt.x() << ',' << pt.y() << ',' << pt.z() << ','
        << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z() << ','
        << v.x() << ',' << v.y() << ',' << v.z() << ','
        << gp[0] << ',' << gp[1] << ',' << gt_yaw << '\n';
  };
  write_row(d.time[0], pose0, v0, 0);

  // ---- Main loop ----
  for (int t = 1; t < T; ++t) {
    // Preintegrate the high-rate IMU sub-samples covering (t-1, t].
    for (int k = 0; k < d.M; ++k) {
      auto s = d.imuSub(ship, t, k);
      gtsam::Vector3 omega(s[0], s[1], s[2]);
      gtsam::Vector3 accel(s[3], s[4], s[5]);
      pim->integrateMeasurement(accel, omega, d.imuSubDt(t, k));
    }

    // CombinedImuFactor between (X,V,B)_{t-1} and (X,V,B)_t
    graph.add(gtsam::CombinedImuFactor(X(t - 1), V(t - 1), X(t), V(t),
                                       B(t - 1), B(t), *pim));

    // Predict initial estimate.
    gtsam::NavState pred = pim->predict(prev_state, prev_bias);
    values.insert(X(t), pred.pose());
    values.insert(V(t), pred.velocity());
    values.insert(B(t), prev_bias);

    // Planar pin: heterogeneous Pose3 prior anchored at predicted (x, y, yaw)
    // with z=0, roll=pitch=0. Rebuild a planar pose from the prediction.
    {
      gtsam::Vector3 rpy = pred.pose().rotation().rpy();
      gtsam::Rot3 R_planar = gtsam::Rot3::Yaw(rpy.z());
      gtsam::Pose3 planar_target(
          R_planar,
          gtsam::Point3(pred.pose().x(), pred.pose().y(), 0.0));
      graph.addPrior<gtsam::Pose3>(X(t), planar_target, planar_pin);
    }

    // GNSS aiding — position and yaw are independent factors with independent
    // validity masks and independent (robust) noise models. With trust on,
    // a per-step innovation pre-check votes good/bad into SelfTrust; the
    // post-vote trust scales the base sigma BEFORE the robust kernel wraps.
    //
    // Predictor sigma = smoother's posterior marginal at X(t-1) (captures
    // everything the smoother has already learned: landmarks, planar pin,
    // GNSS history, IMU history) + the IMU preintegration cov over this
    // step. Using only the preint cov underestimates predictor uncertainty
    // by ~100x because it ignores accumulated state uncertainty.
    Eigen::Matrix<double, 6, 6> marg_prev =
        Eigen::Matrix<double, 6, 6>::Zero();
    try {
      marg_prev = smoother.marginalCovariance(X(t - 1));
    } catch (const std::exception&) {
      // First-step bootstrap: no marginal yet → fall back to preint only.
    }
    const double sigma_marg_xy = std::sqrt(
        0.5 * (std::max(marg_prev(3, 3), 0.0) +
               std::max(marg_prev(4, 4), 0.0)));
    const double sigma_marg_yaw =
        std::sqrt(std::max(marg_prev(2, 2), 0.0));

    Eigen::Matrix<double, 15, 15> pim_cov = pim->preintMeasCov();
    // Convention: rows 0..2 attitude, 3..5 position, 6..8 velocity.
    const double sigma_pim_yaw = std::sqrt(std::max(pim_cov(2, 2), 0.0));
    const double sigma_pim_xy = std::sqrt(
        0.5 * (std::max(pim_cov(3, 3), 0.0) +
               std::max(pim_cov(4, 4), 0.0)));

    const double sigma_pred_xy = std::sqrt(
        sigma_marg_xy * sigma_marg_xy + sigma_pim_xy * sigma_pim_xy);
    const double sigma_pred_yaw = std::sqrt(
        sigma_marg_yaw * sigma_marg_yaw + sigma_pim_yaw * sigma_pim_yaw);

    std::set<std::string> sensors_seen_this_step;
    for (const auto& gk : gnss_keys) {
      const int s = gk.sensor_idx;
      const bool has_pos = d.gnssPosValid(s, t);
      const bool has_yaw = d.gnssYawValid(s, t);

      bool veto_pos = false;
      if (trust_cfg.enable && (has_pos || has_yaw)) {
        // Innovation against the IMU prediction (decoupled from the iSAM2
        // linearization seed by construction since `pred` is built from the
        // last smoothed state).
        const double pred_x = pred.pose().x();
        const double pred_y = pred.pose().y();
        const double pred_yaw = pred.pose().rotation().rpy().z();

        const bool robust_pos_on = (args.robust_pos != "none" &&
                                    !args.robust_pos.empty());
        const bool robust_yaw_on = (args.robust_yaw != "none" &&
                                    !args.robust_yaw.empty());

        if (has_pos) {
          auto r = d.gnssPos(s, t);
          double dx = r[0] - pred_x;
          double dy = r[1] - pred_y;
          double sx = std::sqrt(sigma_xy * sigma_xy +
                                sigma_pred_xy * sigma_pred_xy);
          double maha_pos = (dx * dx + dy * dy) / (sx * sx);
          bool pos_bad;
          if (robust_pos_on) {
            double cutoff = args.robust_pos_k * trust_cfg.robust_k_mult;
            pos_bad = std::sqrt(maha_pos) > cutoff;
          } else {
            pos_bad = maha_pos > trust_cfg.gnss_pos_thresh;
          }
          trust.update(gk.pos_key, !pos_bad);
          sensors_seen_this_step.insert(gk.pos_key);
          if (trust_cfg.gnss_veto && pos_bad) veto_pos = true;
        }
        if (has_yaw) {
          double dtheta = wrapPi(d.gnssYaw(s, t) - pred_yaw);
          double sth = std::sqrt(sigma_yaw * sigma_yaw +
                                 sigma_pred_yaw * sigma_pred_yaw);
          double whitened = std::abs(dtheta / sth);
          bool yaw_bad;
          if (robust_yaw_on) {
            double cutoff = args.robust_yaw_k * trust_cfg.robust_k_mult;
            yaw_bad = whitened > cutoff;
          } else {
            yaw_bad = (whitened * whitened) > trust_cfg.gnss_hdg_thresh;
          }
          trust.update(gk.hdg_key, !yaw_bad);
          sensors_seen_this_step.insert(gk.hdg_key);
        }
      }

      if (has_pos && !veto_pos) {
        auto r = d.gnssPos(s, t);
        graph.add(gtsam::GPSFactor(X(t),
                                   gtsam::Point3(r[0], r[1], r[2]),
                                   build_gnss_pos_noise(trust.get(gk.pos_key))));
      }
      if (has_yaw) {
        graph.add(parnav::CompassFactor<gtsam::Pose3>(
            X(t), d.gnssYaw(s, t),
            build_gnss_yaw_noise(trust.get(gk.hdg_key))));
      }
    }

    // ---- Polar landmarks + shoreline ----
    // Two parallel pathways per polar sensor:
    //   - Marker detections (polar_marker) associate to known point landmarks
    //     and add RangeFactor + MarkerAzimuthFactor against the matched marker.
    //   - Shoreline detections (polar_shoreline) associate to the nearest
    //     known shoreline segment and snap to the closest point on it; the
    //     same RangeFactor + MarkerAzimuthFactor pair is then used against the
    //     snapped point (treated as a "virtual marker").
    // Per-sensor accept/reject counts from BOTH pathways are summed into a
    // single per-step trust vote, matching the Python pipeline's one-vote-per
    // -sensor convention.
    if (args.use_landmarks) {
      auto build_polar_noises = [&](const PolarSensor& ps) {
        double scale = 1.0;
        if (trust_cfg.enable) {
          scale = parnav::trustScale(trust.get(ps.key), scale_cfg);
        }
        auto range_base = gtsam::noiseModel::Isotropic::Sigma(
            1, scale * ps.sigma_range);
        auto az_base = gtsam::noiseModel::Isotropic::Sigma(
            1, scale * ps.sigma_az_rad);
        return std::pair<gtsam::SharedNoiseModel, gtsam::SharedNoiseModel>{
            wrapRobust(args.robust_polar, args.robust_polar_k, range_base),
            wrapRobust(args.robust_polar, args.robust_polar_k, az_base)};
      };

      for (const auto& ps : polar_sensors) {
        int n_total = 0;
        int n_rejected = 0;

        // Marker pathway.
        if (d.n_markers > 0) {
          for (int k = 0; k < d.kmax_polar_marker; ++k) {
            if (!d.polarMarkerValid(ps.sensor_idx, t, k)) continue;
            auto rd = d.polarMarkerReading(ps.sensor_idx, t, k);
            const double range_m = rd[0];
            const double az_rad = rd[1];
            ++n_total;

            parnav::AssocResult ar = parnav::associateMarker(
                pred.pose(), ps.yaw_offset_rad, range_m, az_rad,
                ps.sigma_range, ps.sigma_az_rad, d.markers, d.n_markers,
                args.assoc_gate_chi2);
            if (ar.marker_idx < 0) {
              ++n_rejected;
              continue;
            }
            auto [range_noise, az_noise] = build_polar_noises(ps);
            graph.add(parnav::RangeFactor<gtsam::Pose3>(
                X(t), range_noise, range_m, ar.marker_world));
            graph.add(parnav::MarkerAzimuthFactor<gtsam::Pose3>(
                X(t), az_noise, az_rad, ar.marker_world, ps.yaw_offset_rad));
          }
        }

        // Shoreline pathway (off by default; see Args::use_shoreline).
        if (args.use_shoreline && d.n_shoreline > 0) {
          for (int k = 0; k < d.kmax_polar_shoreline; ++k) {
            if (!d.polarShorelineValid(ps.sensor_idx, t, k)) continue;
            auto rd = d.polarShorelineReading(ps.sensor_idx, t, k);
            const double range_m = rd[0];
            const double az_rad = rd[1];
            ++n_total;

            parnav::AssocResult ar = parnav::associateShoreline(
                pred.pose(), ps.yaw_offset_rad, range_m, az_rad,
                ps.sigma_range, ps.sigma_az_rad, d.shoreline, d.n_shoreline,
                args.assoc_gate_chi2);
            if (ar.marker_idx < 0) {
              ++n_rejected;
              continue;
            }
            auto [range_noise, az_noise] = build_polar_noises(ps);
            graph.add(parnav::RangeFactor<gtsam::Pose3>(
                X(t), range_noise, range_m, ar.marker_world));
            graph.add(parnav::MarkerAzimuthFactor<gtsam::Pose3>(
                X(t), az_noise, az_rad, ar.marker_world, ps.yaw_offset_rad));
          }
        }

        if (trust_cfg.enable && n_total > 0) {
          const double bad_ratio = static_cast<double>(n_rejected) / n_total;
          const bool is_bad = bad_ratio >= trust_cfg.gate_bad_ratio;
          trust.update(ps.key, !is_bad);
          sensors_seen_this_step.insert(ps.key);
        }
      }
    }

    // ---- Camera bearing-only factors (off by default; --camera) ----
    // Cameras measure azimuth to a landmark with no range, so each detection
    // is bearing-only associated to a known marker and adds a single
    // MarkerAzimuthFactor. One trust vote per sensor per step, mirroring polar.
    if (args.use_camera && d.n_markers > 0) {
      for (const auto& cs : camera_sensors) {
        int n_total = 0, n_rejected = 0;
        for (int k = 0; k < d.kmax_camera; ++k) {
          if (!d.cameraValid(cs.sensor_idx, t, k)) continue;
          const double az_rad = d.cameraReading(cs.sensor_idx, t, k)[0];
          ++n_total;

          parnav::AssocResult ar = parnav::associateMarkerBearing(
              pred.pose(), cs.yaw_offset_rad, az_rad, cs.sigma_az_rad,
              d.markers, d.n_markers, args.assoc_gate_chi2);
          if (ar.marker_idx < 0) { ++n_rejected; continue; }

          double scale = trust_cfg.enable
                             ? parnav::trustScale(trust.get(cs.key), scale_cfg)
                             : 1.0;
          auto az_base = gtsam::noiseModel::Isotropic::Sigma(
              1, scale * cs.sigma_az_rad);
          graph.add(parnav::MarkerAzimuthFactor<gtsam::Pose3>(
              X(t), wrapRobust(args.robust_polar, args.robust_polar_k, az_base),
              az_rad, ar.marker_world, cs.yaw_offset_rad));
        }
        if (trust_cfg.enable && n_total > 0) {
          const bool is_bad =
              static_cast<double>(n_rejected) / n_total >= trust_cfg.gate_bad_ratio;
          trust.update(cs.key, !is_bad);
          sensors_seen_this_step.insert(cs.key);
        }
      }
    }

    // Decay forgetting for known sensors that didn't fire this step so trust
    // drifts back toward the prior.
    if (trust_cfg.enable) {
      for (const auto& gk : gnss_keys) {
        if (!sensors_seen_this_step.count(gk.pos_key)) trust.decay(gk.pos_key);
        if (!sensors_seen_this_step.count(gk.hdg_key)) trust.decay(gk.hdg_key);
      }
      for (const auto& ps : polar_sensors) {
        if (!sensors_seen_this_step.count(ps.key)) trust.decay(ps.key);
      }
      for (const auto& cs : camera_sensors) {
        if (!sensors_seen_this_step.count(cs.key)) trust.decay(cs.key);
      }
      trust.record();
    }

    timestamps[X(t)] = d.time[t];
    timestamps[V(t)] = d.time[t];
    timestamps[B(t)] = d.time[t];

    try {
      smoother.update(graph, values, timestamps);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "smoother.update failed at t=%d: %s\n", t, e.what());
      throw;
    }

    auto est_pose = smoother.calculateEstimate<gtsam::Pose3>(X(t));
    auto est_vel = smoother.calculateEstimate<gtsam::Vector3>(V(t));
    auto est_bias =
        smoother.calculateEstimate<gtsam::imuBias::ConstantBias>(B(t));

    write_row(d.time[t], est_pose, est_vel, t);

    prev_state = gtsam::NavState(est_pose, est_vel);
    prev_bias = est_bias;
    pim->resetIntegrationAndSetBias(prev_bias);

    graph.resize(0);
    values.clear();
    timestamps.clear();
  }

  out.close();
  std::printf("Wrote %s (T=%d, ship=%d, lag=%.2fs)\n",
              args.out_csv.c_str(), T, ship, args.lag);

  if (trust_cfg.enable) {
    std::string trust_path = args.trust_csv;
    if (trust_path.empty()) {
      // Default: <out_csv stem>_trust.csv
      auto dot = args.out_csv.find_last_of('.');
      trust_path = (dot == std::string::npos)
                       ? args.out_csv + "_trust.csv"
                       : args.out_csv.substr(0, dot) + "_trust.csv";
    }
    std::ofstream tout(trust_path);
    tout << "t,sensor,trust\n";
    const auto& hist = trust.history();
    for (std::size_t k = 0; k < hist.size(); ++k) {
      const double tk = d.time[k + 1];  // history starts at first stepped t
      for (const auto& kv : hist[k]) {
        tout << tk << ',' << kv.first << ',' << kv.second << '\n';
      }
    }
    tout.close();
    std::printf("Wrote %s (trust history, %zu steps)\n", trust_path.c_str(),
                hist.size());
  }
  return 0;
}
