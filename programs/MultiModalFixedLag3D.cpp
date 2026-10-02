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

#include "AzimuthFactor.hpp"
#include "CompassFactor.hpp"
#include "GatingWindow.hpp"
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
#include <gtsam/slam/BetweenFactor.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
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
  std::string robust_odom = "none";
  double robust_odom_k = 1.345;
  int use_odom = 0;                   // 0 off (default): relative-pose
                                      // odometry factors; opt in with --odom.
  std::string robust_range = "none";
  double robust_range_k = 1.345;
  int use_range = 0;                  // 0 off (default): range-only
                                      // factors; opt in with --range.
  // Association gate override (sentinel: NaN means "use sidecar value",
  // meta.gating.assoc_gate_chi2, default 5.99 = 2-DoF χ² @ 95%).
  double assoc_gate_chi2 = std::nan("");
  // Trust overrides (sentinel: empty / NaN means "use sidecar value")
  int trust_enable = -1;              // -1 keep, 0 force off, 1 force on
  std::string trust_scaling;
  double trust_floor = std::nan("");
  double trust_alpha1 = std::nan("");
  double trust_alpha2 = std::nan("");
  double trust_forget_good = std::nan("");  // continuous-time forgetting
  double trust_forget_bad = std::nan("");
  double trust_pos_thresh = std::nan("");
  double trust_hdg_thresh = std::nan("");
  double trust_robust_k_mult = std::nan("");
  int trust_gnss_veto = -1;
  // Gating window override (sentinel: NaN means "use sidecar value").
  double gate_window_s = std::nan("");
  // Distributed-agent sensor subset (empty = no filtering, all sensors —
  // fully backward compatible with pre-agents invocations).
  std::string agent_id;
  // ALARM1 feedback: path to a `sensor,start,duration` CSV (episodes where
  // a sensor was isolated as faulty by the federated LOO agents). Empty =
  // no rejection (default, backward compatible). Parsed once in main() into
  // reject_windows below.
  std::string reject_episodes_csv;
  std::map<std::string, std::vector<std::pair<double, double>>> reject_windows;
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
    else if (k == "--trust-enable") a.trust_enable = 1;
    else if (k == "--no-trust") a.trust_enable = 0;
    else if (k == "--trust-scaling") a.trust_scaling = next();
    else if (k == "--trust-floor") a.trust_floor = std::stod(next());
    else if (k == "--trust-alpha1") a.trust_alpha1 = std::stod(next());
    else if (k == "--trust-alpha2") a.trust_alpha2 = std::stod(next());
    else if (k == "--trust-forget-good")
      a.trust_forget_good = std::stod(next());
    else if (k == "--trust-forget-bad") a.trust_forget_bad = std::stod(next());
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
    else if (k == "--robust-odom") a.robust_odom = next();
    else if (k == "--robust-odom-k") a.robust_odom_k = std::stod(next());
    else if (k == "--no-odom") a.use_odom = 0;
    else if (k == "--odom") a.use_odom = 1;
    else if (k == "--robust-range") a.robust_range = next();
    else if (k == "--robust-range-k") a.robust_range_k = std::stod(next());
    else if (k == "--no-range") a.use_range = 0;
    else if (k == "--range") a.use_range = 1;
    else if (k == "--assoc-gate-chi2") a.assoc_gate_chi2 = std::stod(next());
    else if (k == "--gate-window-s") a.gate_window_s = std::stod(next());
    else if (k == "--agent-id") a.agent_id = next();
    else if (k == "--reject-episodes") a.reject_episodes_csv = next();
    else if (k == "-h" || k == "--help") {
      std::cout
          << "MultiModalFixedLag3D --sim-data PATH --meta PATH "
          << "[--ship-index 0] [--lag 5.0] [--out estimates.csv]\n"
          << "  [--robust-pos none|huber|tukey|gmc] [--robust-pos-k K]\n"
          << "  [--robust-yaw none|huber|tukey|gmc] [--robust-yaw-k K]\n"
          << "  [--trust-enable | --no-trust]\n"
          << "  [--trust-scaling inverse|inverse_sqrt|linear|off]\n"
          << "  [--trust-floor F] [--trust-alpha1 A] [--trust-alpha2 A]\n"
          << "  [--trust-forget-good F] [--trust-forget-bad F]\n"
          << "  [--trust-gnss-pos-thresh CHI2] [--trust-gnss-hdg-thresh CHI2]\n"
          << "  [--trust-robust-k-mult K] [--trust-gnss-veto | --no-trust-gnss-veto]\n"
          << "  [--landmarks | --no-landmarks] [--shoreline | --no-shoreline]\n"
          << "  [--camera | --no-camera] [--assoc-gate-chi2 CHI2]\n"
          << "  [--robust-polar none|huber|tukey|gmc] [--robust-polar-k K]\n"
          << "  [--odom | --no-odom] [--robust-odom none|huber|tukey|gmc]\n"
          << "  [--robust-odom-k K]\n"
          << "  [--range | --no-range] [--robust-range none|huber|tukey|gmc]\n"
          << "  [--robust-range-k K]\n"
          << "  [--gate-window-s SECONDS]\n"
          << "  [--agent-id NAME] [--reject-episodes PATH]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown flag: " + k);
    }
  }
  if (a.sim_data.empty() || a.meta.empty())
    throw std::runtime_error("--sim-data and --meta are required");
  return a;
}

// ALARM1 feedback: `sensor,start,duration` CSV (as written by
// i_fault_isolation.py's reject_episodes_trust*.csv) -> sensor name ->
// list of [start, start+duration) reject windows.
std::map<std::string, std::vector<std::pair<double, double>>>
loadRejectEpisodes(const std::string& path) {
  std::map<std::string, std::vector<std::pair<double, double>>> out;
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open --reject-episodes file: " + path);
  std::string line;
  std::getline(f, line);  // header
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string sensor, start_s, duration_s;
    std::getline(ss, sensor, ',');
    std::getline(ss, start_s, ',');
    std::getline(ss, duration_s, ',');
    const double start = std::stod(start_s);
    const double duration = std::stod(duration_s);
    out[sensor].emplace_back(start, start + duration);
  }
  return out;
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

// Apply the per-agent override (from the YAML `agents:` block, if set), then
// CLI overrides, on top of the YAML-derived TrustConfig from the sidecar.
// Precedence: CLI explicit flag > per-agent YAML override > global sidecar
// value -- same override order every other CLI flag in this file already
// follows relative to the sidecar.
parnav::TrustConfig effectiveTrust(const parnav::TrustConfig& base,
                                   const Args& a,
                                   const parnav::AgentMeta& agent) {
  parnav::TrustConfig t = base;
  if (agent.trust_enable.has_value()) t.enable = *agent.trust_enable;
  if (a.trust_enable == 0) t.enable = false;
  if (a.trust_enable == 1) t.enable = true;
  if (!a.trust_scaling.empty()) t.scaling = a.trust_scaling;
  if (!std::isnan(a.trust_floor)) t.floor = a.trust_floor;
  if (!std::isnan(a.trust_alpha1)) t.alpha1 = a.trust_alpha1;
  if (!std::isnan(a.trust_alpha2)) t.alpha2 = a.trust_alpha2;
  // Either forgetting value (CLI or YAML) switches to continuous-time mode;
  // the one left unset defaults to 1% per nominal measurement period.
  if (!std::isnan(a.trust_forget_good)) t.forget_good = a.trust_forget_good;
  if (!std::isnan(a.trust_forget_bad)) t.forget_bad = a.trust_forget_bad;
  if (t.forget_good >= 0.0 || t.forget_bad >= 0.0) {
    if (t.forget_good < 0.0) t.forget_good = 0.01;
    if (t.forget_bad < 0.0) t.forget_bad = 0.01;
  }
  if (!std::isnan(a.trust_pos_thresh)) t.gnss_pos_thresh = a.trust_pos_thresh;
  if (!std::isnan(a.trust_hdg_thresh)) t.gnss_hdg_thresh = a.trust_hdg_thresh;
  if (!std::isnan(a.trust_robust_k_mult))
    t.robust_k_mult = a.trust_robust_k_mult;
  if (a.trust_gnss_veto == 0) t.gnss_veto = false;
  if (a.trust_gnss_veto == 1) t.gnss_veto = true;
  return t;
}

}  // namespace

// Run one agent's estimator end-to-end: build its graph from the sensor
// subset in `agent`, smooth, and write all CSVs next to `out_csv`. Every
// piece of state here (smoother, graph, trust/gating, output streams)
// is local, so agents are fully independent — main() just calls this in a
// loop, one agent per iteration.
void runAgent(const parnav::SimData3D& d, const parnav::SimMeta& meta,
              const Args& args, const parnav::AgentMeta& agent,
              const std::string& out_csv) {
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
  parnav::TrustConfig trust_cfg = effectiveTrust(meta.trust, args, agent);

  // trust_cfg.enable == false makes this agent fully passive: robust kernels
  // forced off for every sensor, regardless of --robust-pos/--robust-yaw/
  // --robust-polar. This is the one place in this file where the agent's
  // resolved setting wins over an explicit CLI flag -- LOO agents used for
  // fault isolation must ingest a fault's data uncorrected (no per-agent
  // outlier rejection), otherwise ALARM1's cross-agent divergence test is
  // itself confounded by gating. The GNSS trust veto/pre-check is already
  // skipped when trust_cfg.enable is false (guarded below); this just closes
  // the remaining gap (robust kernels aren't gated on trust_cfg.enable
  // upstream).
  const std::string robust_pos_kind =
      trust_cfg.enable ? args.robust_pos : "none";
  const std::string robust_yaw_kind =
      trust_cfg.enable ? args.robust_yaw : "none";
  const std::string robust_polar_kind =
      trust_cfg.enable ? args.robust_polar : "none";
  const std::string robust_odom_kind =
      trust_cfg.enable ? args.robust_odom : "none";
  const std::string robust_range_kind =
      trust_cfg.enable ? args.robust_range : "none";

  parnav::TrustScalingCfg scale_cfg{trust_cfg.scaling, trust_cfg.floor,
                                    trust_cfg.linear_k};
  parnav::SelfTrust trust(trust_cfg.alpha1, trust_cfg.alpha2);
  const bool trust_continuous =
      trust_cfg.forget_good >= 0.0 && trust_cfg.forget_bad >= 0.0;
  if (trust_continuous) {
    trust.setForgetting(trust_cfg.forget_good, trust_cfg.forget_bad);
    trust.setDefaultRate(1.0 / meta.dt_nominal);
  }

  // ---- Gating pass-ratio (windowed, independent of the trust EWMA) ----
  const double gate_window_s = std::isnan(args.gate_window_s)
                                   ? meta.gating.window_s
                                   : args.gate_window_s;
  parnav::GatingWindow gating(gate_window_s);
  const double assoc_gate_chi2 = std::isnan(args.assoc_gate_chi2)
                                      ? meta.gating.assoc_gate_chi2
                                      : args.assoc_gate_chi2;

  // ---- Distributed-agent sensor subset ----
  // Empty allowlist = no filtering (agent with `sensors: null`/[] = all).
  std::set<std::string> agent_allowlist(agent.sensors.begin(),
                                        agent.sensors.end());
  // Warn on stale/typo'd sensor names: the filters below silently ignore a
  // name that isn't in the sim data, so surface it here rather than letting
  // an agent quietly run with fewer sensors than intended.
  if (!agent_allowlist.empty()) {
    std::set<std::string> known;
    for (const auto& g : meta.gnss) known.insert(g.name);
    for (const auto& p : meta.polar) known.insert(p.name);
    for (const auto& c : meta.camera) known.insert(c.name);
    for (const auto& o : meta.odom) known.insert(o.name);
    for (const auto& r : meta.range) known.insert(r.name);
    for (const auto& n : agent.sensors)
      if (!known.count(n))
        std::fprintf(stderr,
                     "[agent %s] warning: sensor '%s' not in sim data "
                     "(ignored)\n",
                     agent.id.c_str(), n.c_str());
  }
  // Stamped on every output row so per-agent runs can be told apart/joined.
  const std::string agent_col = agent.id;

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
    if (!agent_allowlist.empty() && !agent_allowlist.count(gm.name)) continue;
    gnss_keys.push_back({s, gm.name + "_POS", gm.name + "_HDG"});
  }

  // Per-Polar-sensor cached params + trust key.
  struct PolarSensor {
    int sensor_idx;
    std::string key;
    double yaw_offset_rad;
    double sigma_range;
    double sigma_az_rad;
    // Ship-tracking sensor (e.g. Bluetooth): tracks a ship's live position
    // directly rather than static markers. world_pos only meaningful then.
    bool is_ship_tracker{false};
    // Land-mounted (ship == -1), not a full ship-tracker: its raw detection
    // stream mixes static-marker sightings with the ship's own live
    // position with no per-detection tag, so it goes through the hybrid
    // marker-or-ship association instead of the plain marker pathway.
    // world_pos meaningful whenever is_ship_tracker || is_land.
    bool is_land{false};
    gtsam::Point3 world_pos{0.0, 0.0, 0.0};
  };
  std::vector<PolarSensor> polar_sensors;
  for (int s = 0; s < d.n_polar; ++s) {
    const auto& pm = meta.polar[s];
    const bool is_land = pm.ship == -1;
    const bool is_ship_tracker = !pm.detect_ship_ids.empty();
    const bool tracks_this_ship =
        is_ship_tracker &&
        std::find(pm.detect_ship_ids.begin(), pm.detect_ship_ids.end(),
                  ship) != pm.detect_ship_ids.end();
    // Include: this ship's own sensor; a land sensor that's a ship-tracker
    // for *this* ship (Bluetooth-style, unambiguous); or a land sensor
    // that's not a ship-tracker at all (hybrid marker-or-ship, below).
    // Excluded: a land sensor that's a ship-tracker for a *different* ship
    // (irrelevant here), and any other ship's own sensor.
    if (pm.ship != ship) {
      if (!is_land) continue;
      if (is_ship_tracker && !tracks_this_ship) continue;
    }
    if (!agent_allowlist.empty() && !agent_allowlist.count(pm.name)) continue;
    polar_sensors.push_back({s, pm.name,
                             parnav::deg2rad(pm.relative_pose[2]),
                             pm.range_noise,
                             parnav::deg2rad(pm.angle_noise_deg),
                             is_ship_tracker,
                             is_land,
                             gtsam::Point3(pm.relative_pose[0],
                                           pm.relative_pose[1], 0.0)});
  }

  // Per-Camera-sensor cached params + trust key (bearing-only, no range).
  struct CameraSensor {
    int sensor_idx;
    std::string key;
    double yaw_offset_rad;
    double sigma_az_rad;
    // Land-mounted (ship == -1): hybrid marker-or-ship association, same
    // reasoning as PolarSensor::is_land. world_pos meaningful when is_land.
    bool is_land{false};
    gtsam::Point3 world_pos{0.0, 0.0, 0.0};
  };
  std::vector<CameraSensor> camera_sensors;
  for (int s = 0; s < d.n_camera; ++s) {
    const auto& cm = meta.camera[s];
    if (cm.ship != ship && cm.ship != -1) continue;  // other ships' cameras excluded
    if (!agent_allowlist.empty() && !agent_allowlist.count(cm.name)) continue;
    camera_sensors.push_back({s, cm.name,
                              parnav::deg2rad(cm.relative_pose[2]),
                              parnav::deg2rad(cm.angle_noise_deg),
                              cm.ship == -1,
                              gtsam::Point3(cm.relative_pose[0],
                                            cm.relative_pose[1], 0.0)});
  }

  // Per-Odometry-sensor cached params + trust key. Always ship-mounted (a
  // relative-pose transform is inherently about this ship's own motion).
  struct OdomSensor {
    int sensor_idx;
    std::string key;
    double yaw_offset_rad;
    double sigma_pos;
    double sigma_yaw;
  };
  std::vector<OdomSensor> odom_sensors;
  for (int s = 0; s < d.n_odom; ++s) {
    const auto& om = meta.odom[s];
    if (om.ship != ship) continue;
    if (!agent_allowlist.empty() && !agent_allowlist.count(om.name)) continue;
    odom_sensors.push_back({s, om.name, parnav::deg2rad(om.relative_pose[2]),
                            om.noise_pos, om.noise_yaw});
  }

  // Per-Range-sensor cached params + trust key (range-only, no bearing).
  // Mirrors PolarSensor's is_ship_tracker/is_land split exactly, minus the
  // azimuth-related fields.
  struct RangeSensor {
    int sensor_idx;
    std::string key;
    double sigma_range;
    bool is_ship_tracker{false};
    bool is_land{false};
    gtsam::Point3 world_pos{0.0, 0.0, 0.0};
  };
  std::vector<RangeSensor> range_sensors;
  for (int s = 0; s < d.n_range; ++s) {
    const auto& rm = meta.range[s];
    const bool is_land = rm.ship == -1;
    const bool is_ship_tracker = !rm.detect_ship_ids.empty();
    const bool tracks_this_ship =
        is_ship_tracker &&
        std::find(rm.detect_ship_ids.begin(), rm.detect_ship_ids.end(),
                  ship) != rm.detect_ship_ids.end();
    if (rm.ship != ship) {
      if (!is_land) continue;
      if (is_ship_tracker && !tracks_this_ship) continue;
    }
    if (!agent_allowlist.empty() && !agent_allowlist.count(rm.name)) continue;
    range_sensors.push_back({s, rm.name, rm.range_noise, is_ship_tracker,
                             is_land,
                             gtsam::Point3(rm.relative_pose[0],
                                           rm.relative_pose[1], 0.0)});
  }

  // (s, t) select the per-fix dynamic std when the npz carries gnss_pos_std
  // (real-data exports); falls back to the static per-sensor sigma_xy
  // per-axis when the array is absent or a specific entry is <= 0 (older/
  // synthetic exports have no such array at all).
  auto build_gnss_pos_noise = [&](int s, int t, double trust_value) {
    double k = trust_cfg.enable ? parnav::trustScale(trust_value, scale_cfg)
                                : 1.0;
    double sx = sigma_xy, sy = sigma_xy;
    if (d.hasGnssPosStd()) {
      auto sd = d.gnssPosStd(s, t);
      if (sd[0] > 0.0) sx = sd[0];
      if (sd[1] > 0.0) sy = sd[1];
    }
    auto base = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(3) << k * sx, k * sy, 0.05).finished());
    return wrapRobust(robust_pos_kind, args.robust_pos_k, base);
  };
  auto build_gnss_yaw_noise = [&](double trust_value) {
    double k = trust_cfg.enable ? parnav::trustScale(trust_value, scale_cfg)
                                : 1.0;
    auto base = gtsam::noiseModel::Isotropic::Sigma(1, k * sigma_yaw);
    return wrapRobust(robust_yaw_kind, args.robust_yaw_k, base);
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
  double sigma_acc = meta.imu.sigma_acc;    // m/s^2/sqrt(Hz)
  double sigma_gyro = meta.imu.sigma_gyro;  // rad/s/sqrt(Hz)
  double sigma_b_acc = meta.imu.sigma_b_acc;
  double sigma_b_gyro = meta.imu.sigma_b_gyro;

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
  // Shared GNSS seed: initialise the t=0 prior from the ship's first valid
  // GNSS fix regardless of this agent's sensor subset, so a GNSS-excluded
  // agent (e.g. no_gnss) still starts at the right pose instead of the
  // origin. One-time handshake only — no GPSFactor from an excluded GNSS
  // ever enters the graph, since the main loop still respects the allowlist.
  int s0 = -1;
  for (int s = 0; s < d.n_gnss; ++s) {
    const auto& gm = meta.gnss[s];
    if (gm.ship == ship || gm.ship == -1) { s0 = s; break; }
  }

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

  // Zero-order hold for IMU dropout substeps: real captures have short
  // (~0.4-0.8s) full-window gyro/accel outages (comms/buffer hiccups), too
  // frequent to just skip without leaving CombinedImuFactor degenerate
  // (zero integrated time -> no constraint on V(t)/B(t)). For a slow
  // harbour vessel, "no acceleration, constant angular velocity" during a
  // sub-second gap is a reasonable coast assumption, so held (last-valid)
  // measurements are integrated in place of the missing ones, at the
  // nominal per-substep dt (the real dt is zero-padded for these slots).
  gtsam::Vector3 last_valid_omega(0.0, 0.0, 0.0);
  gtsam::Vector3 last_valid_accel(0.0, 0.0, g0);
  const double dt_substep_nominal = meta.dt_nominal / d.M;

  // ---- Output CSV ----
  std::ofstream out(out_csv);
  out << "t,x,y,z,qw,qx,qy,qz,vx,vy,vz,gt_x,gt_y,gt_yaw,"
         "cov_xx,cov_yy,cov_yawyaw,cov_xy,cov_xyaw,cov_yyaw,agent_id\n";

  // Residual/edge-share diagnostics — raw per-step, no windowing here
  // (windowing is done downstream in Python). Written unconditionally
  // (independent of trust_cfg.enable): these are meant to stand alone from
  // the trust EWMA, per plan.
  auto out_stem = [&]() {
    auto dot = out_csv.find_last_of('.');
    return dot == std::string::npos ? out_csv : out_csv.substr(0, dot);
  }();
  std::ofstream rout(out_stem + "_residuals.csv");
  rout << "t,sensor,channel,residual,agent_id\n";
  std::ofstream eout(out_stem + "_edges.csv");
  eout << "t,sensor,n_edges,agent_id\n";

  // Divergence-monitoring covariance: planar (x, y, yaw) sub-block of the
  // full 6x6 Pose3 marginal, with cross terms (not just diagonal) so a
  // Mahalanobis divergence test between two agents' estimates is possible
  // downstream. Tangent order is (rx,ry,rz,tx,ty,tz) — see the planar_pin
  // comment above — so yaw is index 2, x is index 3, y is index 4.
  auto planarCov = [&](gtsam::Key key) -> std::array<double, 6> {
    Eigen::Matrix<double, 6, 6> C = smoother.marginalCovariance(key);
    return {C(3, 3), C(4, 4), C(2, 2), C(3, 4), C(3, 2), C(4, 2)};
  };

  auto write_row = [&](double t, const gtsam::Pose3& P, const gtsam::Vector3& v,
                       int gt_t, const std::array<double, 6>& cov) {
    auto q = P.rotation().toQuaternion();
    auto pt = P.translation();
    auto gp = d.gtPose(ship, gt_t);
    double gt_yaw = 2.0 * std::atan2(gp[6], gp[3]);
    out << t << ',' << pt.x() << ',' << pt.y() << ',' << pt.z() << ','
        << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z() << ','
        << v.x() << ',' << v.y() << ',' << v.z() << ','
        << gp[0] << ',' << gp[1] << ',' << gt_yaw << ','
        << cov[0] << ',' << cov[1] << ',' << cov[2] << ','
        << cov[3] << ',' << cov[4] << ',' << cov[5] << ','
        << agent_col << '\n';
  };
  write_row(d.time[0], pose0, v0, 0, planarCov(X(0)));

  // Per-factor record for the residual/edge-share diagnostics: one entry per
  // graph.add(...) call this step (odometry excluded — diagnostics cover
  // measurement factors only). Populated regardless of trust_cfg.enable,
  // since these outputs are meant to stand alone from the trust EWMA.
  struct FactorRecord {
    std::string sensor_key;
    std::string channel;  // "pos" | "yaw" | "range" | "az"
    gtsam::NoiseModelFactor::shared_ptr factor;
  };

  // Nominal measurement rate per trust key for continuous-time forgetting:
  // min(sensor frequency, 1/dt_nominal) -- the smoother cannot see a sensor
  // faster than once per step. Unknown frequency -> default (1/dt_nominal).
  if (trust_continuous) {
    const double max_rate = 1.0 / meta.dt_nominal;
    auto rate = [&](double f) { return f > 0.0 ? std::min(f, max_rate) : 0.0; };
    for (const auto& gk : gnss_keys) {
      const double r = rate(meta.gnss[gk.sensor_idx].frequency);
      trust.setNominalRate(gk.pos_key, r);
      trust.setNominalRate(gk.hdg_key, r);
    }
    for (const auto& ps : polar_sensors)
      trust.setNominalRate(ps.key, rate(meta.polar[ps.sensor_idx].frequency));
    for (const auto& cs : camera_sensors)
      trust.setNominalRate(cs.key, rate(meta.camera[cs.sensor_idx].frequency));
    for (const auto& os : odom_sensors)
      trust.setNominalRate(os.key, rate(meta.odom[os.sensor_idx].frequency));
    for (const auto& rs : range_sensors)
      trust.setNominalRate(rs.key, rate(meta.range[rs.sensor_idx].frequency));
  }

  // ---- Main loop ----
  for (int t = 1; t < T; ++t) {
    trust.setTime(d.time[t]);
    // Preintegrate the high-rate IMU sub-samples covering (t-1, t].
    for (int k = 0; k < d.M; ++k) {
      gtsam::Vector3 omega, accel;
      double dt;
      if (d.imuValid(t, k)) {
        auto s = d.imuSub(ship, t, k);
        omega = gtsam::Vector3(s[0], s[1], s[2]);
        accel = gtsam::Vector3(s[3], s[4], s[5]);
        dt = d.imuSubDt(t, k);
        last_valid_omega = omega;
        last_valid_accel = accel;
      } else {
        // Coast: hold last-valid measurements over the (zero-padded) dropout
        // substep at its nominal duration.
        omega = last_valid_omega;
        accel = last_valid_accel;
        dt = dt_substep_nominal;
      }
      pim->integrateMeasurement(accel, omega, dt);
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

    std::vector<FactorRecord> factor_records;
    std::map<std::string, int> n_edges;
    auto recordFactor = [&](const std::string& key, const std::string& channel) {
      factor_records.push_back(
          {key, channel,
           std::dynamic_pointer_cast<gtsam::NoiseModelFactor>(graph.back())});
      ++n_edges[key];
    };

    // ALARM1 feedback: hard-reject a sensor's measurements entirely (no
    // factor, no robust-kernel bookkeeping) while `t` falls inside one of
    // its reject windows. Explicitly votes the sensor "bad" into the trust
    // model (below) rather than leaving it unseen -- ALARM1 has positively
    // identified misbehaviour, which should read as rising disbelief, not
    // the rising *uncertainty* that the ordinary decay-on-silence path
    // (further below) would otherwise produce.
    auto isRejected = [&](const std::string& name) {
      auto it = args.reject_windows.find(name);
      if (it == args.reject_windows.end()) return false;
      for (const auto& w : it->second)
        if (d.time[t] >= w.first && d.time[t] < w.second) return true;
      return false;
    };
    auto rejectVote = [&](const std::string& key) {
      if (!trust_cfg.enable) return;
      trust.update(key, false);
      sensors_seen_this_step.insert(key);
    };

    for (const auto& gk : gnss_keys) {
      if (isRejected(meta.gnss[gk.sensor_idx].name)) {
        rejectVote(gk.pos_key);
        rejectVote(gk.hdg_key);
        continue;
      }
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

        const bool robust_pos_on = (robust_pos_kind != "none" &&
                                    !robust_pos_kind.empty());
        const bool robust_yaw_on = (robust_yaw_kind != "none" &&
                                    !robust_yaw_kind.empty());

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
          gating.push(gk.pos_key, d.time[t], 1, pos_bad ? 0 : 1);
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
          gating.push(gk.hdg_key, d.time[t], 1, yaw_bad ? 0 : 1);
          sensors_seen_this_step.insert(gk.hdg_key);
        }
      }

      if (has_pos && !veto_pos) {
        auto r = d.gnssPos(s, t);
        graph.add(gtsam::GPSFactor(X(t),
                                   gtsam::Point3(r[0], r[1], r[2]),
                                   build_gnss_pos_noise(s, t, trust.get(gk.pos_key))));
        recordFactor(gk.pos_key, "pos");
      }
      if (has_yaw) {
        graph.add(parnav::CompassFactor<gtsam::Pose3>(
            X(t), d.gnssYaw(s, t),
            build_gnss_yaw_noise(trust.get(gk.hdg_key))));
        recordFactor(gk.hdg_key, "yaw");
      }
    }

    // ---- Odometry (relative pose) ----
    // A single BetweenFactor<Pose3> per valid step, from X(ref) to X(t),
    // where `ref` is this sensor's own reference grid index (usually t-1,
    // but can be further back if the odometry samples slower than the
    // graph step grid -- see odomRefStep()). No association step (unlike
    // polar/camera): identity of "which pose pair this measures" is given
    // directly by the sensor, so the only gate is whether X(ref) is still
    // live in the fixed-lag window.
    if (args.use_odom) {
      for (const auto& os : odom_sensors) {
        if (isRejected(os.key)) { rejectVote(os.key); continue; }
        if (!d.odomDeltaValid(os.sensor_idx, t)) continue;

        const long ref = d.odomRefStep(os.sensor_idx, t);
        if (ref < 0 || ref >= t || !smoother.timestamps().count(X(ref))) {
          // Malformed ref, or X(ref) already marginalized out of the lag
          // window (odometry period exceeded --lag) -- skip, don't crash.
          continue;
        }

        auto delta = d.odomDelta(os.sensor_idx, t);  // sensor-frame [dx,dy,dyaw]
        const double co = std::cos(os.yaw_offset_rad);
        const double so = std::sin(os.yaw_offset_rad);
        const double dx = co * delta[0] - so * delta[1];
        const double dy = so * delta[0] + co * delta[1];
        const gtsam::Pose3 delta_pose(gtsam::Rot3::Yaw(delta[2]),
                                      gtsam::Point3(dx, dy, 0.0));

        double sp = os.sigma_pos, syaw = os.sigma_yaw;
        if (d.hasOdomDeltaStd()) {
          auto sd = d.odomDeltaStd(os.sensor_idx, t);
          if (sd[0] > 0.0) sp = sd[0];
          if (sd[2] > 0.0) syaw = sd[2];
        }
        const double odom_k = trust_cfg.enable
                             ? parnav::trustScale(trust.get(os.key), scale_cfg)
                             : 1.0;
        // Tangent order (rx,ry,rz,tx,ty,tz): roll/pitch/z pinned tight (the
        // delta is planar by construction), yaw/x/y carry the measurement.
        auto base = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << sig_planar, sig_planar, odom_k * syaw,
             odom_k * sp, odom_k * sp, sig_planar)
                .finished());
        auto noise = wrapRobust(robust_odom_kind, args.robust_odom_k, base);

        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(X(ref), X(t), delta_pose,
                                                      noise));
        recordFactor(os.key, "odom");

        if (trust_cfg.enable) {
          trust.update(os.key, true);
          gating.push(os.key, d.time[t], 1, 1);
          sensors_seen_this_step.insert(os.key);
        }
      }
    }

    // ---- Range-only ----
    // Mirrors the Polar landmarks block below, minus azimuth. Known-ID ship
    // targets (RangeSensorMeta::detect_ship_ids set) skip association
    // entirely -- the measurement already carries the target's identity, so
    // this is a single-hypothesis consistency check, not a search. Static
    // markers still need range-only association (associateMarkerRangeOnly /
    // associateMarkerOrShipRangeOnly), which is more ambiguous than the
    // range+azimuth case without a bearing to help discriminate.
    if (args.use_range) {
      for (const auto& rs : range_sensors) {
        if (isRejected(rs.key)) { rejectVote(rs.key); continue; }
        int n_total = 0;
        int n_rejected = 0;
        const double rs_scale = trust_cfg.enable
                                     ? parnav::trustScale(trust.get(rs.key), scale_cfg)
                                     : 1.0;

        if (rs.is_ship_tracker) {
          const gtsam::Pose3 observer_pose(gtsam::Rot3::Identity(), rs.world_pos);
          const gtsam::Point3 ship_xy(pred.pose().x(), pred.pose().y(), 0.0);
          for (int k = 0; k < d.kmax_range_marker; ++k) {
            if (!d.rangeMarkerValid(rs.sensor_idx, t, k)) continue;
            const double range_m = d.rangeMarkerReading(rs.sensor_idx, t, k);
            ++n_total;

            parnav::LandAssocResult ar = parnav::associateMarkerOrShipRangeOnly(
                observer_pose, range_m, rs.sigma_range, {}, 0, ship_xy,
                assoc_gate_chi2);
            if (!ar.matched) {
              ++n_rejected;
              continue;
            }

            double sr = rs.sigma_range;
            if (d.hasRangeMarkerStd()) {
              double v = d.rangeMarkerStd(rs.sensor_idx, t, k);
              if (v > 0.0) sr = v;
            }
            auto range_base = gtsam::noiseModel::Isotropic::Sigma(1, rs_scale * sr);
            auto range_noise = wrapRobust(robust_range_kind, args.robust_range_k, range_base);
            graph.add(parnav::RangeFactor<gtsam::Pose3>(
                X(t), range_noise, range_m, rs.world_pos));
            recordFactor(rs.key, "range");
          }
          if (trust_cfg.enable && n_total > 0) {
            const int n_accepted = n_total - n_rejected;
            trust.update(rs.key, n_accepted > 0);
            gating.push(rs.key, d.time[t], n_total, n_accepted);
            sensors_seen_this_step.insert(rs.key);
          }
          continue;
        }

        const gtsam::Pose3 observer_pose =
            rs.is_land ? gtsam::Pose3(gtsam::Rot3::Identity(), rs.world_pos)
                      : pred.pose();
        bool ship_matched = false;
        if (d.n_markers > 0 || rs.is_land) {
          for (int k = 0; k < d.kmax_range_marker; ++k) {
            if (!d.rangeMarkerValid(rs.sensor_idx, t, k)) continue;
            const double range_m = d.rangeMarkerReading(rs.sensor_idx, t, k);
            ++n_total;

            double sr = rs.sigma_range;
            if (d.hasRangeMarkerStd()) {
              double v = d.rangeMarkerStd(rs.sensor_idx, t, k);
              if (v > 0.0) sr = v;
            }
            auto range_base = gtsam::noiseModel::Isotropic::Sigma(1, rs_scale * sr);
            auto range_noise = wrapRobust(robust_range_kind, args.robust_range_k, range_base);

            if (rs.is_land) {
              const gtsam::Point3 ship_xy(pred.pose().x(), pred.pose().y(), 0.0);
              parnav::LandAssocResult ar = parnav::associateMarkerOrShipRangeOnly(
                  observer_pose, range_m, rs.sigma_range, d.markers,
                  d.n_markers, ship_xy, assoc_gate_chi2);
              if (!ar.matched) {
                ++n_rejected;
                continue;
              }
              if (ar.is_ship) {
                ship_matched = true;
                graph.add(parnav::RangeFactor<gtsam::Pose3>(
                    X(t), range_noise, range_m, rs.world_pos));
                recordFactor(rs.key, "range");
              }
              // else: land sensor matched a known marker, not the ship --
              // counted as valid for gating, no factor, no trust credit
              // (land sensors only earn trust by matching the ship).
              continue;
            }

            parnav::AssocResult ar = parnav::associateMarkerRangeOnly(
                observer_pose, range_m, rs.sigma_range, d.markers,
                d.n_markers, assoc_gate_chi2);
            if (ar.marker_idx < 0) {
              ++n_rejected;
              continue;
            }
            graph.add(parnav::RangeFactor<gtsam::Pose3>(
                X(t), range_noise, range_m, ar.marker_world));
            recordFactor(rs.key, "range");
            if (trust_cfg.enable) {
              trust.update(rs.key, true);
              sensors_seen_this_step.insert(rs.key);
            }
          }
        }
        if (trust_cfg.enable && n_total > 0) {
          gating.push(rs.key, d.time[t], n_total, n_total - n_rejected);
          if (rs.is_land && ship_matched) {
            trust.update(rs.key, true);
            sensors_seen_this_step.insert(rs.key);
          }
        }
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
    // Land sensors (hybrid marker-or-ship) only get trust credit for
    // matching the ship itself -- we don't care whether they also see
    // static markers. Ship-mounted sensors (plain marker association, since
    // a ship can't detect itself) get one good vote PER matched marker this
    // step, not one aggregate vote for the whole step. Trust scale is
    // computed once per sensor per step (not per detection) so a sensor's
    // own inline votes this step can't feed back into its own noise before
    // the step finishes.
    if (args.use_landmarks) {
      // sigma_range/sigma_az_rad are resolved by the caller per detection
      // (static per-sensor value, or the npz's per-detection _std when
      // present and positive) before this is invoked.
      auto build_polar_noises = [&](double sigma_range, double sigma_az_rad,
                                    double scale) {
        auto range_base = gtsam::noiseModel::Isotropic::Sigma(
            1, scale * sigma_range);
        auto az_base = gtsam::noiseModel::Isotropic::Sigma(
            1, scale * sigma_az_rad);
        return std::pair<gtsam::SharedNoiseModel, gtsam::SharedNoiseModel>{
            wrapRobust(robust_polar_kind, args.robust_polar_k, range_base),
            wrapRobust(robust_polar_kind, args.robust_polar_k, az_base)};
      };
      // Per-detection sigma with fallback to the sensor's static value,
      // shared by all marker/shoreline call sites below.
      auto polar_marker_sigmas = [&](const PolarSensor& ps, int t, int k) {
        double sr = ps.sigma_range, sa = ps.sigma_az_rad;
        if (d.hasPolarMarkerStd()) {
          auto sd = d.polarMarkerStd(ps.sensor_idx, t, k);
          if (sd[0] > 0.0) sr = sd[0];
          if (sd[1] > 0.0) sa = sd[1];
        }
        return std::pair<double, double>{sr, sa};
      };
      auto polar_shoreline_sigmas = [&](const PolarSensor& ps, int t, int k) {
        double sr = ps.sigma_range, sa = ps.sigma_az_rad;
        if (d.hasPolarShorelineStd()) {
          auto sd = d.polarShorelineStd(ps.sensor_idx, t, k);
          if (sd[0] > 0.0) sr = sd[0];
          if (sd[1] > 0.0) sa = sd[1];
        }
        return std::pair<double, double>{sr, sa};
      };

      for (const auto& ps : polar_sensors) {
        if (isRejected(ps.key)) { rejectVote(ps.key); continue; }
        int n_total = 0;
        int n_rejected = 0;
        const double ps_scale = trust_cfg.enable
                                     ? parnav::trustScale(trust.get(ps.key), scale_cfg)
                                     : 1.0;

        if (ps.is_ship_tracker) {
          // Ship-tracking sensor (e.g. NTNU_bluetooth): the simulator's
          // detect_ship_ids already fully replaced this sensor's target list
          // with the live ship position, so every valid polar_marker reading
          // IS this ship directly. Skip associateMarker (nearest-neighbor
          // against static ENC points is the wrong pool here) but still gate
          // each detection against the predicted ship pose via
          // associateMarkerOrShip with an empty marker pool (n_markers=0),
          // which degenerates to a single-hypothesis chi^2 test against the
          // ship estimate — the same consistency check every other polar
          // sensor gets, just without the marker-vs-ship ambiguity.
          const gtsam::Pose3 observer_pose(gtsam::Rot3::Identity(), ps.world_pos);
          const gtsam::Matrix3 R_rn = gtsam::Rot3::Yaw(ps.yaw_offset_rad).matrix();
          const gtsam::Point3 ship_xy(pred.pose().x(), pred.pose().y(), 0.0);
          for (int k = 0; k < d.kmax_polar_marker; ++k) {
            if (!d.polarMarkerValid(ps.sensor_idx, t, k)) continue;
            auto rd = d.polarMarkerReading(ps.sensor_idx, t, k);
            const double range_m = rd[0];
            const double az_rad = rd[1];
            ++n_total;

            parnav::LandAssocResult ar = parnav::associateMarkerOrShip(
                observer_pose, ps.yaw_offset_rad, range_m, az_rad,
                ps.sigma_range, ps.sigma_az_rad, {}, 0, ship_xy,
                assoc_gate_chi2);
            if (!ar.matched) {
              ++n_rejected;
              continue;
            }

            auto [sr, sa] = polar_marker_sigmas(ps, t, k);
            auto [range_noise, az_noise] = build_polar_noises(sr, sa, ps_scale);
            graph.add(parnav::RangeFactor<gtsam::Pose3>(
                X(t), range_noise, range_m, ps.world_pos));
            recordFactor(ps.key, "range");
            graph.add(parnav::AzimuthFactor<gtsam::Pose3>(
                X(t), az_noise, az_rad, ps.world_pos, R_rn));
            recordFactor(ps.key, "az");
          }
          if (trust_cfg.enable && n_total > 0) {
            const int n_accepted = n_total - n_rejected;
            trust.update(ps.key, n_accepted > 0);
            gating.push(ps.key, d.time[t], n_total, n_accepted);
            sensors_seen_this_step.insert(ps.key);
          }
          continue;
        }

        // Land sensors observe from their own fixed world position/heading,
        // not the (moving) ship's predicted pose. Ship-mounted sensors keep
        // using the predicted ship pose as before.
        const gtsam::Pose3 observer_pose =
            ps.is_land ? gtsam::Pose3(gtsam::Rot3::Identity(), ps.world_pos)
                      : pred.pose();

        // Marker pathway — hybrid marker-or-ship for land sensors (their
        // raw stream doesn't tag which is which), plain marker association
        // for ship-mounted sensors. `|| ps.is_land` so a land sensor still
        // gets a chance at a ship match even in a marker-less scenario.
        // Trust credit differs by mounting: land sensors only care about
        // matching the ship itself (set ship_matched below, voted once at
        // the end of this block); ship-mounted sensors can't see the ship
        // at all, so instead get one good vote per matched marker, fired
        // inline right here.
        bool ship_matched = false;
        if (d.n_markers > 0 || ps.is_land) {
          for (int k = 0; k < d.kmax_polar_marker; ++k) {
            if (!d.polarMarkerValid(ps.sensor_idx, t, k)) continue;
            auto rd = d.polarMarkerReading(ps.sensor_idx, t, k);
            const double range_m = rd[0];
            const double az_rad = rd[1];
            ++n_total;

            if (ps.is_land) {
              const gtsam::Point3 ship_xy(pred.pose().x(), pred.pose().y(), 0.0);
              parnav::LandAssocResult ar = parnav::associateMarkerOrShip(
                  observer_pose, ps.yaw_offset_rad, range_m, az_rad,
                  ps.sigma_range, ps.sigma_az_rad, d.markers, d.n_markers,
                  ship_xy, assoc_gate_chi2);
              if (!ar.matched) {
                ++n_rejected;
                continue;
              }
              auto [sr, sa] = polar_marker_sigmas(ps, t, k);
              auto [range_noise, az_noise] = build_polar_noises(sr, sa, ps_scale);
              if (ar.is_ship) {
                ship_matched = true;
                const gtsam::Matrix3 R_rn =
                    gtsam::Rot3::Yaw(ps.yaw_offset_rad).matrix();
                graph.add(parnav::RangeFactor<gtsam::Pose3>(
                    X(t), range_noise, range_m, ps.world_pos));
                recordFactor(ps.key, "range");
                graph.add(parnav::AzimuthFactor<gtsam::Pose3>(
                    X(t), az_noise, az_rad, ps.world_pos, R_rn));
                recordFactor(ps.key, "az");
              }
              // else: land sensor matched a known marker, not the ship.
              // Markers are fixed constants here (no landmark variable to
              // estimate — that would make this SLAM), so this detection
              // carries no ship-pose information: counted as a valid
              // (non-rejected) detection for gating, but no trust credit —
              // land sensors only earn trust by matching the ship.
              continue;
            }

            parnav::AssocResult ar = parnav::associateMarker(
                observer_pose, ps.yaw_offset_rad, range_m, az_rad,
                ps.sigma_range, ps.sigma_az_rad, d.markers, d.n_markers,
                assoc_gate_chi2);
            if (ar.marker_idx < 0) {
              ++n_rejected;
              continue;
            }
            auto [sr, sa] = polar_marker_sigmas(ps, t, k);
            auto [range_noise, az_noise] = build_polar_noises(sr, sa, ps_scale);
            graph.add(parnav::RangeFactor<gtsam::Pose3>(
                X(t), range_noise, range_m, ar.marker_world));
            recordFactor(ps.key, "range");
            graph.add(parnav::MarkerAzimuthFactor<gtsam::Pose3>(
                X(t), az_noise, az_rad, ar.marker_world, ps.yaw_offset_rad));
            recordFactor(ps.key, "az");
            // Ship's own sensor: one good vote per matched marker, not one
            // aggregate vote for the whole step.
            if (trust_cfg.enable) {
              trust.update(ps.key, true);
              sensors_seen_this_step.insert(ps.key);
            }
          }
        }

        // Shoreline pathway (off by default; see Args::use_shoreline). Not
        // hypothesized to have the marker-vs-ship ambiguity, but needs
        // observer_pose (not always pred.pose()) to be correct now that
        // land sensors reach this code — it was never exercised for them
        // before, since they used to be excluded entirely. A shoreline
        // segment is a static feature like a marker, never the ship, so
        // it follows the same trust-credit split as the marker pathway.
        if (args.use_shoreline && d.n_shoreline > 0) {
          for (int k = 0; k < d.kmax_polar_shoreline; ++k) {
            if (!d.polarShorelineValid(ps.sensor_idx, t, k)) continue;
            auto rd = d.polarShorelineReading(ps.sensor_idx, t, k);
            const double range_m = rd[0];
            const double az_rad = rd[1];
            ++n_total;

            parnav::AssocResult ar = parnav::associateShoreline(
                observer_pose, ps.yaw_offset_rad, range_m, az_rad,
                ps.sigma_range, ps.sigma_az_rad, d.shoreline, d.n_shoreline,
                assoc_gate_chi2);
            if (ar.marker_idx < 0) {
              ++n_rejected;
              continue;
            }
            auto [sr, sa] = polar_shoreline_sigmas(ps, t, k);
            auto [range_noise, az_noise] = build_polar_noises(sr, sa, ps_scale);
            graph.add(parnav::RangeFactor<gtsam::Pose3>(
                X(t), range_noise, range_m, ar.marker_world));
            recordFactor(ps.key, "range");
            graph.add(parnav::MarkerAzimuthFactor<gtsam::Pose3>(
                X(t), az_noise, az_rad, ar.marker_world, ps.yaw_offset_rad));
            recordFactor(ps.key, "az");
            if (trust_cfg.enable && !ps.is_land) {
              trust.update(ps.key, true);
              sensors_seen_this_step.insert(ps.key);
            }
          }
        }

        if (trust_cfg.enable && n_total > 0) {
          gating.push(ps.key, d.time[t], n_total, n_total - n_rejected);
          // Land sensors: no reliable "bad" signal without FOV/LOS-
          // awareness (clutter/markers can produce an all-reject-of-ship
          // step even when the ship was genuinely out of range or
          // obstructed), and marker-only matches don't count here either
          // (land sensors only earn trust by matching the ship). Only vote
          // good when the ship itself was matched; otherwise left unvoted
          // and decays toward neutral below, same as a silent sensor.
          // Revisit once FOV-aware disbelief lands. Ship-mounted sensors
          // already voted inline above, per matched marker.
          if (ps.is_land && ship_matched) {
            trust.update(ps.key, true);
            sensors_seen_this_step.insert(ps.key);
          }
        }
      }
    }

    // ---- Camera bearing-only factors (off by default; --camera) ----
    // Cameras measure azimuth to a landmark with no range, so each detection
    // is bearing-only associated to a known marker and adds a single
    // MarkerAzimuthFactor. Trust credit follows the same split as polar
    // sensors: land cameras only earn credit for matching the ship (one
    // vote at the end of the step); ship-mounted cameras get one good vote
    // per matched marker, fired inline. No `d.n_markers > 0` clause on the
    // outer guard: a land camera must still get a chance at a ship-
    // hypothesis match in a marker-less scenario, and
    // associateMarkerBearing/-OrShip with n_markers=0 already degrade
    // gracefully (always reject) for ship-mounted cameras.
    if (args.use_camera) {
      for (const auto& cs : camera_sensors) {
        if (isRejected(cs.key)) { rejectVote(cs.key); continue; }
        int n_total = 0, n_rejected = 0;
        const double cs_scale = trust_cfg.enable
                                     ? parnav::trustScale(trust.get(cs.key), scale_cfg)
                                     : 1.0;
        const gtsam::Pose3 observer_pose =
            cs.is_land ? gtsam::Pose3(gtsam::Rot3::Identity(), cs.world_pos)
                      : pred.pose();
        bool ship_matched = false;
        for (int k = 0; k < d.kmax_camera; ++k) {
          if (!d.cameraValid(cs.sensor_idx, t, k)) continue;
          const double az_rad = d.cameraReading(cs.sensor_idx, t, k)[0];
          ++n_total;

          double sa = cs.sigma_az_rad;
          if (d.hasCameraStd()) {
            double v = d.cameraStd(cs.sensor_idx, t, k);
            if (v > 0.0) sa = v;
          }
          auto az_base = gtsam::noiseModel::Isotropic::Sigma(1, cs_scale * sa);
          auto az_noise =
              wrapRobust(robust_polar_kind, args.robust_polar_k, az_base);

          if (cs.is_land) {
            const gtsam::Point3 ship_xy(pred.pose().x(), pred.pose().y(), 0.0);
            parnav::LandAssocResult ar = parnav::associateMarkerBearingOrShip(
                observer_pose, cs.yaw_offset_rad, az_rad, cs.sigma_az_rad,
                d.markers, d.n_markers, ship_xy, assoc_gate_chi2);
            if (!ar.matched) { ++n_rejected; continue; }
            if (ar.is_ship) {
              ship_matched = true;
              const gtsam::Matrix3 R_rn =
                  gtsam::Rot3::Yaw(cs.yaw_offset_rad).matrix();
              graph.add(parnav::AzimuthFactor<gtsam::Pose3>(
                  X(t), az_noise, az_rad, cs.world_pos, R_rn));
              recordFactor(cs.key, "az");
            }
            // else: land camera matched a known marker, not the ship —
            // counted as a valid (non-rejected) detection for gating, no
            // factor added, no trust credit (land sensors only earn trust
            // by matching the ship).
            continue;
          }

          parnav::AssocResult ar = parnav::associateMarkerBearing(
              observer_pose, cs.yaw_offset_rad, az_rad, cs.sigma_az_rad,
              d.markers, d.n_markers, assoc_gate_chi2);
          if (ar.marker_idx < 0) { ++n_rejected; continue; }
          graph.add(parnav::MarkerAzimuthFactor<gtsam::Pose3>(
              X(t), az_noise, az_rad, ar.marker_world, cs.yaw_offset_rad));
          recordFactor(cs.key, "az");
          // Ship's own sensor: one good vote per matched marker.
          if (trust_cfg.enable) {
            trust.update(cs.key, true);
            sensors_seen_this_step.insert(cs.key);
          }
        }
        if (trust_cfg.enable && n_total > 0) {
          gating.push(cs.key, d.time[t], n_total, n_total - n_rejected);
          if (cs.is_land && ship_matched) {
            trust.update(cs.key, true);
            sensors_seen_this_step.insert(cs.key);
          }
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
      for (const auto& os : odom_sensors) {
        if (!sensors_seen_this_step.count(os.key)) trust.decay(os.key);
      }
      for (const auto& rs : range_sensors) {
        if (!sensors_seen_this_step.count(rs.key)) trust.decay(rs.key);
      }
      trust.record();
      gating.record();
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

    write_row(d.time[t], est_pose, est_vel, t, planarCov(X(t)));

    // ---- Residual / edge-share diagnostics ----
    // unweightedWhiten() gives the whitened residual with the robust
    // M-estimator's reweight bypassed (see Robust::unweightedWhiten() in
    // NoiseModel.h) — an outlier-detection score, not the value actually
    // used to solve the graph.
    {
      gtsam::Values current_values;
      current_values.insert(X(t), est_pose);
      // Odometry's BetweenFactor<Pose3> is the one binary factor recorded
      // here (every other channel is unary on X(t)) -- it also needs its
      // X(ref) key's current estimate before unwhitenedError() can evaluate.
      for (const auto& rec : factor_records) {
        if (rec.channel != "odom") continue;
        for (const auto& key : rec.factor->keys()) {
          if (!current_values.exists(key)) {
            current_values.insert(key, smoother.calculateEstimate<gtsam::Pose3>(key));
          }
        }
      }
      for (const auto& rec : factor_records) {
        gtsam::Vector unwhitened = rec.factor->unwhitenedError(current_values);
        gtsam::Vector residual = rec.factor->noiseModel()->unweightedWhiten(unwhitened);
        // GNSS "pos" is 3-dim (x, y, z) but z is pinned/uninformative —
        // only report x, y.
        int n = static_cast<int>(residual.size());
        if (rec.channel == "pos" && n == 3) n = 2;
        for (int i = 0; i < n; ++i) {
          rout << d.time[t] << ',' << rec.sensor_key << ',' << rec.channel
               << ',' << residual[i] << ',' << agent_col << '\n';
        }
      }
      for (const auto& kv : n_edges) {
        eout << d.time[t] << ',' << kv.first << ',' << kv.second << ','
             << agent_col << '\n';
      }
    }

    prev_state = gtsam::NavState(est_pose, est_vel);
    prev_bias = est_bias;
    pim->resetIntegrationAndSetBias(prev_bias);

    graph.resize(0);
    values.clear();
    timestamps.clear();
  }

  out.close();
  std::printf("Wrote %s (T=%d, ship=%d, lag=%.2fs)\n",
              out_csv.c_str(), T, ship, args.lag);

  rout.close();
  std::printf("Wrote %s (residuals)\n", (out_stem + "_residuals.csv").c_str());
  eout.close();
  std::printf("Wrote %s (edges)\n", (out_stem + "_edges.csv").c_str());

  if (trust_cfg.enable) {
    std::string trust_path = out_stem + "_trust.csv";
    std::ofstream tout(trust_path);
    tout << "t,sensor,belief,disbelief,uncertainty,agent_id\n";
    const auto& hist = trust.history();
    for (std::size_t k = 0; k < hist.size(); ++k) {
      const double tk = d.time[k + 1];  // history starts at first stepped t
      for (const auto& kv : hist[k]) {
        tout << tk << ',' << kv.first << ',' << kv.second.belief << ','
             << kv.second.disbelief << ',' << kv.second.uncertainty << ','
             << agent_col << '\n';
      }
    }
    tout.close();
    std::printf("Wrote %s (trust history, %zu steps)\n", trust_path.c_str(),
                hist.size());

    std::string gating_path = out_stem + "_gating.csv";
    std::ofstream gout(gating_path);
    gout << "t,sensor,pass_ratio,n_window,agent_id\n";
    const auto& ghist = gating.history();
    for (std::size_t k = 0; k < ghist.size(); ++k) {
      const double tk = d.time[k + 1];
      for (const auto& kv : ghist[k]) {
        gout << tk << ',' << kv.first << ',' << kv.second.pass_ratio << ','
             << kv.second.n_window << ',' << agent_col << '\n';
      }
    }
    gout.close();
    std::printf("Wrote %s (gating history, %zu steps)\n", gating_path.c_str(),
                ghist.size());
  }
}

int main(int argc, char** argv) {
  Args args = parseArgs(argc, argv);
  if (!args.reject_episodes_csv.empty())
    args.reject_windows = loadRejectEpisodes(args.reject_episodes_csv);

  parnav::SimMeta meta = parnav::loadSimMeta(args.meta);
  parnav::SimData3D d = parnav::loadSimData(args.sim_data, meta);

  if (args.ship_index < 0 || args.ship_index >= d.n_ships)
    throw std::runtime_error("ship-index out of range");

  // Which agents to run: --agent-id => just that one (back-compat); otherwise
  // every agent in the sim meta; if the meta declares none, a single
  // all-sensors "default" agent (reproduces the pre-agents single-graph run).
  std::vector<parnav::AgentMeta> selected;
  if (!args.agent_id.empty()) {
    const parnav::AgentMeta* found = nullptr;
    for (const auto& am : meta.agents)
      if (am.id == args.agent_id) { found = &am; break; }
    if (found == nullptr)
      throw std::runtime_error("unknown --agent-id: " + args.agent_id);
    selected.push_back(*found);
  } else if (!meta.agents.empty()) {
    selected = meta.agents;
  } else {
    parnav::AgentMeta def;
    def.id = "default";  // empty sensors => all sensors
    selected.push_back(def);
  }

  // Each agent writes its own CSVs under <out-parent>/<agent-id>/, keeping the
  // --out basename. Agents are independent, so this is a plain serial loop.
  std::filesystem::path outp(args.out_csv);
  std::filesystem::path root =
      outp.has_parent_path() ? outp.parent_path() : std::filesystem::path(".");
  const std::string fname = outp.filename().string();
  for (const auto& agent : selected) {
    std::filesystem::path od = root / agent.id;
    std::filesystem::create_directories(od);
    const std::string out_csv = (od / fname).string();
    std::printf("== agent %s -> %s ==\n", agent.id.c_str(), out_csv.c_str());
    runAgent(d, meta, args, agent, out_csv);
  }
  return 0;
}
