#include "MultiModalSimLoader.hpp"

#include <cnpy.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace parnav {

namespace {

std::vector<double> takeDouble(cnpy::npz_t& z, const std::string& key) {
  auto it = z.find(key);
  if (it == z.end()) {
    throw std::runtime_error("MultiModalSimLoader: missing key '" + key + "'");
  }
  const auto& arr = it->second;
  if (arr.word_size != sizeof(double)) {
    throw std::runtime_error("MultiModalSimLoader: '" + key +
                             "' is not float64");
  }
  const double* data = arr.data<double>();
  return std::vector<double>(data, data + arr.num_vals);
}

std::vector<std::uint8_t> takeU8(cnpy::npz_t& z, const std::string& key) {
  auto it = z.find(key);
  if (it == z.end()) {
    throw std::runtime_error("MultiModalSimLoader: missing key '" + key + "'");
  }
  const auto& arr = it->second;
  if (arr.word_size != sizeof(std::uint8_t)) {
    throw std::runtime_error("MultiModalSimLoader: '" + key +
                             "' is not uint8");
  }
  const std::uint8_t* data = arr.data<std::uint8_t>();
  return std::vector<std::uint8_t>(data, data + arr.num_vals);
}

// Like takeU8, but returns an empty vector instead of throwing when the key
// is absent (older exports don't carry it).
std::vector<std::uint8_t> takeU8Optional(cnpy::npz_t& z, const std::string& key) {
  auto it = z.find(key);
  if (it == z.end()) return {};
  return takeU8(z, key);
}

// Like takeDouble, but returns an empty vector instead of throwing when the
// key is absent (older/synthetic exports don't carry per-detection _std).
std::vector<double> takeDoubleOptional(cnpy::npz_t& z, const std::string& key) {
  auto it = z.find(key);
  if (it == z.end()) return {};
  return takeDouble(z, key);
}

std::vector<std::int64_t> takeInt64(cnpy::npz_t& z, const std::string& key) {
  auto it = z.find(key);
  if (it == z.end()) {
    throw std::runtime_error("MultiModalSimLoader: missing key '" + key + "'");
  }
  const auto& arr = it->second;
  if (arr.word_size != sizeof(std::int64_t)) {
    throw std::runtime_error("MultiModalSimLoader: '" + key +
                             "' is not int64");
  }
  const std::int64_t* data = arr.data<std::int64_t>();
  return std::vector<std::int64_t>(data, data + arr.num_vals);
}

// Like takeInt64, but returns an empty vector instead of throwing when the
// key is absent (odom_ref_step is optional -- absence means "default t-1").
std::vector<std::int64_t> takeInt64Optional(cnpy::npz_t& z, const std::string& key) {
  auto it = z.find(key);
  if (it == z.end()) return {};
  return takeInt64(z, key);
}

template <std::size_t N>
std::array<double, N> jsonToArr(const nlohmann::json& j,
                                std::array<double, N> fallback) {
  if (!j.is_array() || j.size() != N) return fallback;
  std::array<double, N> out;
  for (std::size_t i = 0; i < N; ++i) out[i] = j.at(i).get<double>();
  return out;
}

}  // namespace

SimMeta loadSimMeta(const std::string& json_path) {
  std::ifstream f(json_path);
  if (!f) throw std::runtime_error("cannot open " + json_path);
  nlohmann::json j;
  f >> j;

  SimMeta m;
  m.scenario = j.value("scenario", std::string{});
  m.n_ships = j.value("n_ships", 0);
  m.T = j.value("T", 0);
  // Top-level key (synthetic f_export_3d.py exports) takes priority; some
  // captured real-data sidecars only carry it nested under "imu" instead.
  m.imu_substeps = j.value(
      "imu_substeps",
      j.contains("imu") ? j.at("imu").value("imu_substeps", 1) : 1);
  m.dt_nominal = j.value("dt_nominal", 1.0);
  m.gravity = j.value("gravity", 9.81);

  const auto& sensors = j.at("sensors");
  for (const auto& g : sensors.value("gnss", nlohmann::json::array())) {
    GnssSensorMeta s;
    s.name = g.value("name", std::string{});
    s.frequency = g.value("frequency", 0.0);
    s.ship = g.value("ship", -1);
    s.relative_pose =
        jsonToArr<3>(g.value("relative_pose", nlohmann::json::array()),
                     std::array<double, 3>{{0, 0, 0}});
    s.noise_xy = g.value("noise_xy", 0.0);
    s.noise_heading = g.value("noise_heading", 0.0);
    m.gnss.push_back(std::move(s));
  }
  for (const auto& p : sensors.value("polar", nlohmann::json::array())) {
    PolarSensorMeta s;
    s.name = p.value("name", std::string{});
    s.frequency = p.value("frequency", 0.0);
    s.ship = p.value("ship", -1);
    s.relative_pose =
        jsonToArr<3>(p.value("relative_pose", nlohmann::json::array()),
                     std::array<double, 3>{{0, 0, 0}});
    s.range_noise = p.value("range_noise", 0.0);
    s.angle_noise_deg = p.value("angle_noise_deg", 0.0);
    // `detect_ship_ids: null` (or the key omitted) -> empty = normal sensor.
    if (p.contains("detect_ship_ids") && !p.at("detect_ship_ids").is_null()) {
      s.detect_ship_ids = p.at("detect_ship_ids").get<std::vector<int>>();
    }
    m.polar.push_back(std::move(s));
  }
  for (const auto& c : sensors.value("camera", nlohmann::json::array())) {
    CameraSensorMeta s;
    s.name = c.value("name", std::string{});
    s.frequency = c.value("frequency", 0.0);
    s.ship = c.value("ship", -1);
    s.relative_pose =
        jsonToArr<3>(c.value("relative_pose", nlohmann::json::array()),
                     std::array<double, 3>{{0, 0, 0}});
    s.angle_noise_deg = c.value("angle_noise_deg", 0.0);
    m.camera.push_back(std::move(s));
  }
  for (const auto& o : sensors.value("odom", nlohmann::json::array())) {
    OdomSensorMeta s;
    s.name = o.value("name", std::string{});
    s.frequency = o.value("frequency", 0.0);
    s.ship = o.value("ship", -1);
    s.relative_pose =
        jsonToArr<3>(o.value("relative_pose", nlohmann::json::array()),
                     std::array<double, 3>{{0, 0, 0}});
    s.noise_pos = o.value("noise_pos", 0.0);
    s.noise_yaw = o.value("noise_yaw", 0.0);
    m.odom.push_back(std::move(s));
  }
  for (const auto& r : sensors.value("range", nlohmann::json::array())) {
    RangeSensorMeta s;
    s.name = r.value("name", std::string{});
    s.frequency = r.value("frequency", 0.0);
    s.ship = r.value("ship", -1);
    s.relative_pose =
        jsonToArr<3>(r.value("relative_pose", nlohmann::json::array()),
                     std::array<double, 3>{{0, 0, 0}});
    s.range_noise = r.value("range_noise", 0.0);
    if (r.contains("detect_ship_ids") && !r.at("detect_ship_ids").is_null()) {
      s.detect_ship_ids = r.at("detect_ship_ids").get<std::vector<int>>();
    }
    m.range.push_back(std::move(s));
  }

  if (j.contains("trust")) {
    const auto& t = j.at("trust");
    m.trust.activate_gating = t.value("activate_gating", false);
    m.trust.subjective_opinion = t.value("subjective_opinion", false);
    m.trust.alpha1 = t.value("alpha1", 0.9);
    m.trust.alpha2 = t.value("alpha2", 0.99);
    m.trust.forget_good = t.value("forget_good", -1.0);
    m.trust.forget_bad = t.value("forget_bad", -1.0);
    m.trust.gnss_pos_thresh = t.value("gnss_pos_thresh", 5.99);
    m.trust.gnss_hdg_thresh = t.value("gnss_hdg_thresh", 3.84);
    m.trust.odom_thresh = t.value("odom_thresh", 7.81);
    m.trust.range_thresh = t.value("range_thresh", 3.84);
    m.trust.robust_k_mult = t.value("robust_k_mult", 3.0);
  }

  if (j.contains("gating")) {
    m.gating.assoc_gate_chi2 = j.at("gating").value("assoc_gate_chi2", 5.99);
  }

  if (j.contains("imu")) {
    const auto& i = j.at("imu");
    m.imu.sigma_acc = i.value("sigma_acc", m.imu.sigma_acc);
    m.imu.sigma_gyro = i.value("sigma_gyro", m.imu.sigma_gyro);
    m.imu.sigma_b_acc = i.value("sigma_b_acc", m.imu.sigma_b_acc);
    m.imu.sigma_b_gyro = i.value("sigma_b_gyro", m.imu.sigma_b_gyro);
  }

  if (j.contains("agents")) {
    for (const auto& a : j.at("agents")) {
      AgentMeta am;
      am.id = a.value("id", std::string{});
      // `sensors: null` (or the key omitted) -> empty vector = all sensors.
      if (a.contains("sensors") && !a.at("sensors").is_null()) {
        am.sensors = a.at("sensors").get<std::vector<std::string>>();
      }
      if (a.contains("activate_gating") && !a.at("activate_gating").is_null()) {
        am.activate_gating = a.at("activate_gating").get<bool>();
      }
      if (a.contains("subjective_opinion") &&
          !a.at("subjective_opinion").is_null()) {
        am.subjective_opinion = a.at("subjective_opinion").get<bool>();
      }
      m.agents.push_back(std::move(am));
    }
  }
  return m;
}

SimData3D loadSimData(const std::string& npz_path, const SimMeta& meta) {
  cnpy::npz_t z = cnpy::npz_load(npz_path);

  SimData3D d;
  d.n_ships = meta.n_ships;
  d.T = meta.T;
  d.M = meta.imu_substeps > 0 ? meta.imu_substeps : 1;
  d.n_gnss = static_cast<int>(meta.gnss.size());
  d.n_polar = static_cast<int>(meta.polar.size());
  d.n_camera = static_cast<int>(meta.camera.size());
  d.n_odom = static_cast<int>(meta.odom.size());
  d.n_range = static_cast<int>(meta.range.size());

  d.time = takeDouble(z, "time");
  d.imu_dt = takeDouble(z, "imu_dt");
  d.imu_valid = takeU8Optional(z, "imu_valid");
  d.gt_pose = takeDouble(z, "gt_pose");
  d.gt_vel = takeDouble(z, "gt_vel");
  d.imu = takeDouble(z, "imu");

  d.gnss_pos = takeDouble(z, "gnss_pos");
  d.gnss_pos_valid = takeU8(z, "gnss_pos_valid");
  d.gnss_pos_std = takeDoubleOptional(z, "gnss_pos_std");
  d.gnss_yaw = takeDouble(z, "gnss_yaw");
  d.gnss_yaw_valid = takeU8(z, "gnss_yaw_valid");
  d.polar_marker = takeDouble(z, "polar_marker");
  d.polar_marker_valid = takeU8(z, "polar_marker_valid");
  d.polar_marker_std = takeDoubleOptional(z, "polar_marker_std");
  d.polar_shoreline = takeDouble(z, "polar_shoreline");
  d.polar_shoreline_valid = takeU8(z, "polar_shoreline_valid");
  d.polar_shoreline_std = takeDoubleOptional(z, "polar_shoreline_std");
  d.camera = takeDouble(z, "camera");
  d.camera_valid = takeU8(z, "camera_valid");
  d.camera_std = takeDoubleOptional(z, "camera_std");

  // Odometry/range arrays are entirely absent from the npz when the config
  // has no sensors of that type (older exports, or synthetic scenarios that
  // predate these sensor types) -- guard on n_odom/n_range rather than using
  // the unconditional takeDouble every other required array uses.
  if (d.n_odom > 0) {
    d.odom_delta = takeDouble(z, "odom_delta");
    d.odom_delta_valid = takeU8(z, "odom_delta_valid");
    d.odom_delta_std = takeDoubleOptional(z, "odom_delta_std");
    d.odom_ref_step = takeInt64Optional(z, "odom_ref_step");
  }
  if (d.n_range > 0) {
    d.range_marker = takeDouble(z, "range_marker");
    d.range_marker_valid = takeU8(z, "range_marker_valid");
    d.range_marker_std = takeDoubleOptional(z, "range_marker_std");
  }

  d.markers = takeDouble(z, "markers");
  d.n_markers = static_cast<int>(d.markers.size() / 3);
  d.shoreline = takeDouble(z, "shoreline");
  d.n_shoreline = static_cast<int>(d.shoreline.size() / 6);

  // Recover Kpm / Kps / Kc from total sizes.
  if (d.n_polar > 0 && d.T > 0) {
    const std::size_t per_marker = d.polar_marker.size() / d.n_polar;
    d.kmax_polar_marker = static_cast<int>(per_marker / (d.T * 3));
    const std::size_t per_shore = d.polar_shoreline.size() / d.n_polar;
    d.kmax_polar_shoreline = static_cast<int>(per_shore / (d.T * 3));
  }
  if (d.n_camera > 0 && d.T > 0) {
    const std::size_t per_sensor = d.camera.size() / d.n_camera;
    d.kmax_camera = static_cast<int>(per_sensor / (d.T * 2));
  }
  if (d.n_range > 0 && d.T > 0) {
    const std::size_t per_range = d.range_marker.size() / d.n_range;
    d.kmax_range_marker = static_cast<int>(per_range / (d.T * 1));
  }

  // Sanity checks.
  auto must = [](bool ok, const char* msg) {
    if (!ok) throw std::runtime_error(std::string("sim data: ") + msg);
  };
  must(static_cast<int>(d.time.size()) == d.T, "time length mismatch");
  must(static_cast<int>(d.imu_dt.size()) == d.T * d.M,
       "imu_dt shape mismatch (expected T*M)");
  must(d.imu_valid.empty() || static_cast<int>(d.imu_valid.size()) == d.T * d.M,
       "imu_valid shape mismatch (expected T*M)");
  must(static_cast<int>(d.gt_pose.size()) == d.n_ships * d.T * 7,
       "gt_pose shape mismatch");
  must(static_cast<int>(d.imu.size()) == d.n_ships * d.T * d.M * 6,
       "imu shape mismatch (expected n_ships*T*M*6)");
  must(static_cast<int>(d.gnss_pos.size()) == d.n_gnss * d.T * 3,
       "gnss_pos shape mismatch");
  must(static_cast<int>(d.gnss_pos_valid.size()) == d.n_gnss * d.T,
       "gnss_pos_valid shape mismatch");
  must(static_cast<int>(d.gnss_yaw.size()) == d.n_gnss * d.T,
       "gnss_yaw shape mismatch");
  must(static_cast<int>(d.gnss_yaw_valid.size()) == d.n_gnss * d.T,
       "gnss_yaw_valid shape mismatch");
  must(d.gnss_pos_std.empty() ||
           static_cast<int>(d.gnss_pos_std.size()) == d.n_gnss * d.T * 2,
       "gnss_pos_std shape mismatch (expected Ng*T*2)");
  must(d.polar_marker_std.empty() ||
           static_cast<int>(d.polar_marker_std.size()) ==
               d.n_polar * d.T * d.kmax_polar_marker * 2,
       "polar_marker_std shape mismatch (expected Np*T*Kpm*2)");
  must(d.polar_shoreline_std.empty() ||
           static_cast<int>(d.polar_shoreline_std.size()) ==
               d.n_polar * d.T * d.kmax_polar_shoreline * 2,
       "polar_shoreline_std shape mismatch (expected Np*T*Kps*2)");
  must(d.camera_std.empty() ||
           static_cast<int>(d.camera_std.size()) ==
               d.n_camera * d.T * d.kmax_camera,
       "camera_std shape mismatch (expected Nc*T*Kc)");
  must(d.n_odom == 0 ||
           static_cast<int>(d.odom_delta.size()) == d.n_odom * d.T * 3,
       "odom_delta shape mismatch (expected No*T*3)");
  must(d.n_odom == 0 ||
           static_cast<int>(d.odom_delta_valid.size()) == d.n_odom * d.T,
       "odom_delta_valid shape mismatch (expected No*T)");
  must(d.odom_delta_std.empty() ||
           static_cast<int>(d.odom_delta_std.size()) == d.n_odom * d.T * 3,
       "odom_delta_std shape mismatch (expected No*T*3)");
  must(d.odom_ref_step.empty() ||
           static_cast<int>(d.odom_ref_step.size()) == d.n_odom * d.T,
       "odom_ref_step shape mismatch (expected No*T)");
  must(d.n_range == 0 ||
           static_cast<int>(d.range_marker_valid.size()) ==
               d.n_range * d.T * d.kmax_range_marker,
       "range_marker_valid shape mismatch (expected Nr*T*Krm)");
  must(d.range_marker_std.empty() ||
           static_cast<int>(d.range_marker_std.size()) ==
               d.n_range * d.T * d.kmax_range_marker,
       "range_marker_std shape mismatch (expected Nr*T*Krm)");
  return d;
}

}  // namespace parnav
