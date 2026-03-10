#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Unit3.h>
#include <gtsam/navigation/AttitudeFactor.h>

#include <Eigen/Core>
#include <cmath>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// CSV parsing (copied from ifac_wc_smoother.cpp, minimal changes)
// -----------------------------------------------------------------------------

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
      d.gnss_mb_meas_idx.push_back(d.gnss_meas_idx.back());
      d.gnss_rover_meas_idx.push_back(d.gnss_meas_idx.back());
    }
  }
  return d;
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d S;
  S << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return S;
}

static bool find_first_rover_tick(const MultirotorData& d, size_t& idx) {
  for (size_t i = 0; i < d.t.size(); ++i) {
    if (d.gnss_rover_meas_idx[i] != 0) {
      idx = i;
      return true;
    }
  }
  return false;
}

// -----------------------------------------------------------------------------
// Test
// -----------------------------------------------------------------------------

TEST(Pose3AttitudeFactor, CsvBaselineDebug) {
  const std::string kDefaultCsv =
      "/Users/ghms/ws/ntnu/parnav/parnav-scripts/post_processing/"
      "df_flat_data.csv";

  auto data = parse_multirotor_csv(kDefaultCsv);
  if (!data) {
    std::cerr << "[SKIP] CSV not found at default path: " << kDefaultCsv
              << "\n";
    return;
  }

  size_t idx = 0;
  if (!find_first_rover_tick(*data, idx)) {
    std::cerr << "[SKIP] No rover measurement tick found in CSV\n";
    return;
  }

  // Baseline geometry (same as ifac_wc_smoother.cpp)
  const gtsam::Point3 p_imu_rover_b(0.153, -0.019, -0.302);
  const gtsam::Point3 p_imu_bm_b(-0.310, 0.156, -0.300);
  const gtsam::Point3 baseline_body = p_imu_rover_b - p_imu_bm_b;

  const Eigen::Vector3d z_nav = data->z_gnss_comp[idx];
  const gtsam::Unit3 nZ_meas(z_nav);

  const double yaw = data->yaw[idx];
  const gtsam::Rot3 Rz = gtsam::Rot3::Yaw(yaw);
  const gtsam::Pose3 pose(Rz, gtsam::Point3::Zero());

  auto noise = gtsam::noiseModel::Isotropic::Sigma(2, 1.0);
  gtsam::Pose3AttitudeFactor factor(1, nZ_meas, noise,
                                    gtsam::Unit3(baseline_body));

  gtsam::Matrix H;
  gtsam::Vector r = factor.evaluateError(pose, H);

  std::cout << "\n[Debug] idx=" << idx << " t=" << data->t[idx] << "\n";
  std::cout << "z_gnss_comp (nav): " << z_nav.transpose() << "\n";
  std::cout << "baseline_body:    " << baseline_body.transpose() << "\n";
  std::cout << "yaw (rad): " << yaw << "\n";
  std::cout << "Residual (2x1 Unit3 tangent):\n" << r << "\n";
  std::cout << "Jacobian (2x6):\n" << H << "\n";

  // Expected Jacobian from error-state model (3D residual): H_att = -R * [b]_x
  Eigen::Vector3d b_hat = baseline_body.normalized();
  Eigen::Matrix3d Rmat = Rz.matrix();
  Eigen::Matrix3d H_att = -Rmat * skew(b_hat);

  // Project 3D Jacobian into Unit3 tangent (2D)
  Eigen::Vector3d z_hat = z_nav.normalized();
  Eigen::Vector3d u = z_hat.unitOrthogonal();
  Eigen::Vector3d v = z_hat.cross(u);
  Eigen::Matrix<double, 2, 3> P;
  P.row(0) = u.transpose();
  P.row(1) = v.transpose();

  Eigen::Matrix<double, 2, 3> H_proj_eig = P * H_att;
  Eigen::Matrix<double, 2, 3> H_rot_eig = H.block<2, 3>(0, 0);

  const gtsam::Matrix H_proj = H_proj_eig;
  const gtsam::Matrix H_rot = H_rot_eig;

  const double tol = 1e-5;
  EXPECT(gtsam::assert_equal(-H_proj, H_rot, tol));

  // -------------------------------------------------------------------------
  // Second check: compare measured vs predicted baseline direction
  // -------------------------------------------------------------------------
  Eigen::Vector3d pred_nav = (Rz.matrix() * b_hat).normalized();
  Eigen::Vector3d meas_nav = z_nav.normalized();

  double cosang = pred_nav.dot(meas_nav);
  cosang = std::max(-1.0, std::min(1.0, cosang));
  const double angle = std::acos(cosang);

  double cosang_flip = pred_nav.dot(-meas_nav);
  cosang_flip = std::max(-1.0, std::min(1.0, cosang_flip));
  const double angle_flip = std::acos(cosang_flip);

  const double dot = pred_nav.dot(meas_nav);
  const char* tag = (dot >= 0.0) ? "DIRECT" : "FLIPPED";
  std::cout << "Baseline direction match: " << tag << " (dot=" << dot << ")\n";

  std::cout << "Angle (meas): " << angle << " rad\n";
  std::cout << "Angle (meas flipped): " << angle_flip << " rad\n";

  const double angle_tol = 0.2;  // ~11.5 deg
  EXPECT(angle < angle_tol || angle_flip < angle_tol);
}

// -----------------------------------------------------------------------------

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
