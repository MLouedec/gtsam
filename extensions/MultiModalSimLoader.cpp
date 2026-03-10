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
  m.imu_substeps = j.value("imu_substeps", 1);
  m.dt_nominal = j.value("dt_nominal", 1.0);
  m.gravity = j.value("gravity", 9.81);

  const auto& sensors = j.at("sensors");
  for (const auto& g : sensors.value("gnss", nlohmann::json::array())) {
    GnssSensorMeta s;
    s.name = g.value("name", std::string{});
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
    s.ship = p.value("ship", -1);
    s.relative_pose =
        jsonToArr<3>(p.value("relative_pose", nlohmann::json::array()),
                     std::array<double, 3>{{0, 0, 0}});
    s.range_noise = p.value("range_noise", 0.0);
    s.angle_noise_deg = p.value("angle_noise_deg", 0.0);
    m.polar.push_back(std::move(s));
  }
  for (const auto& c : sensors.value("camera", nlohmann::json::array())) {
    CameraSensorMeta s;
    s.name = c.value("name", std::string{});
    s.ship = c.value("ship", -1);
    s.relative_pose =
        jsonToArr<3>(c.value("relative_pose", nlohmann::json::array()),
                     std::array<double, 3>{{0, 0, 0}});
    s.angle_noise_deg = c.value("angle_noise_deg", 0.0);
    m.camera.push_back(std::move(s));
  }

  if (j.contains("trust")) {
    const auto& t = j.at("trust");
    m.trust.enable = t.value("enable", false);
    m.trust.scaling = t.value("scaling", std::string{"inverse"});
    m.trust.floor = t.value("floor", 0.001);
    m.trust.linear_k = t.value("linear_k", 5.0);
    m.trust.alpha1 = t.value("alpha1", 0.9);
    m.trust.alpha2 = t.value("alpha2", 0.99);
    m.trust.gate_bad_ratio = t.value("gate_bad_ratio", 0.5);
    m.trust.gnss_pos_thresh = t.value("gnss_pos_thresh", 5.99);
    m.trust.gnss_hdg_thresh = t.value("gnss_hdg_thresh", 3.84);
    m.trust.robust_k_mult = t.value("robust_k_mult", 3.0);
    m.trust.gnss_veto = t.value("gnss_veto", false);
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

  d.time = takeDouble(z, "time");
  d.imu_dt = takeDouble(z, "imu_dt");
  d.gt_pose = takeDouble(z, "gt_pose");
  d.gt_vel = takeDouble(z, "gt_vel");
  d.imu = takeDouble(z, "imu");

  d.gnss_pos = takeDouble(z, "gnss_pos");
  d.gnss_pos_valid = takeU8(z, "gnss_pos_valid");
  d.gnss_yaw = takeDouble(z, "gnss_yaw");
  d.gnss_yaw_valid = takeU8(z, "gnss_yaw_valid");
  d.polar_marker = takeDouble(z, "polar_marker");
  d.polar_marker_valid = takeU8(z, "polar_marker_valid");
  d.polar_shoreline = takeDouble(z, "polar_shoreline");
  d.polar_shoreline_valid = takeU8(z, "polar_shoreline_valid");
  d.camera = takeDouble(z, "camera");
  d.camera_valid = takeU8(z, "camera_valid");

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

  // Sanity checks.
  auto must = [](bool ok, const char* msg) {
    if (!ok) throw std::runtime_error(std::string("sim data: ") + msg);
  };
  must(static_cast<int>(d.time.size()) == d.T, "time length mismatch");
  must(static_cast<int>(d.imu_dt.size()) == d.T * d.M,
       "imu_dt shape mismatch (expected T*M)");
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
  return d;
}

}  // namespace parnav
