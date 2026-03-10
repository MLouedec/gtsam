#pragma once

#include "JacobianTraits.hpp"

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace parnav {

/// Barometric height factor with an additive bias state.
///
/// Measurement model converts a pressure sample (kPa) to height via the ICAO
/// standard atmosphere, then subtracts a fixed base height to obtain a
/// relative-height observation `z_rel`. The predicted height is
/// `-p_D + baro_bias` (NED convention), so the residual is
/// `(-p_D + baro_bias) - z_rel`.
///
/// Jacobian layout:
///   Pose3          (tangent order [theta, rho]):     H1 = [ 0_{1x3} | zvec' * R ]
///   Se23  (tangent order [theta, nu, rho]): H1 = [ 0_{1x3} | 0_{1x3} | zvec' * R ]
/// where `zvec = (0, 0, -1)` and `R = pose.rotation().matrix()`.
/// H2 = 1 (scalar bias).
template <typename Pose>
class BaroFactor
    : public gtsam::NoiseModelFactorN<Pose, double> {
 private:
  using Base = gtsam::NoiseModelFactorN<Pose, double>;
  using Jacobian = Eigen::Matrix<double, 1, JacobianTraits<Pose>::Cols>;

  double base_height_;
  double height_rel_;  // z_rel = heightOut(pressure) - base_height_

 public:
  using Base::evaluateError;

  BaroFactor() = default;
  ~BaroFactor() override = default;

  /// @param pressure_kPa raw pressure measurement (kilopascal).
  /// @param p0_kPa local sea-level reference pressure (kPa); the standard
  ///        101.29 is only correct on a standard day -- on any other day this
  ///        is the local QFF/QNH and must be calibrated.
  /// @param T0_K sea-level reference temperature [K] (default 288.08 = std
  ///        day, 15 C). Cold-day flights shrink `dh/dp` linearly with T0;
  ///        passing the actual ground temperature corrects the altitude
  ///        slope.
  BaroFactor(gtsam::Key pose_key, gtsam::Key baro_bias_key, double pressure_kPa,
             const gtsam::SharedNoiseModel& model, double base_height = 0.0,
             double p0_kPa = 101.29, double T0_K = 288.08)
      : Base(model, pose_key, baro_bias_key),
        base_height_(base_height),
        height_rel_(height_from_pressure(pressure_kPa, p0_kPa, T0_K) -
                    base_height) {}

  auto evaluateError(const Pose& pose, const double& bias,
                     gtsam::OptionalMatrixType H1 = OptionalNone,
                     gtsam::OptionalMatrixType H2 = OptionalNone) const
      -> gtsam::Vector override;

  /// NASA-GRC simple troposphere model: pressure (kPa) -> height (m).
  /// `p0_kPa` is the local sea-level reference; `T0_K` is the sea-level
  /// reference temperature in kelvin. Both default to standard-day.
  static inline double height_from_pressure(double n_kPa,
                                            double p0_kPa = 101.29,
                                            double T0_K = 288.08) {
    return (std::pow(n_kPa / p0_kPa, 1.0 / 5.256) * T0_K - T0_K) / -0.00649;
  }

  /// Solve for p0 such that height_from_pressure(p_meas, p0, T0_K) ==
  /// h_target. Pass the local T0 for cold-day corrections.
  static inline double p0_from_known_altitude(double p_meas_kPa,
                                              double h_target_m,
                                              double T0_K = 288.08) {
    const double T_K = T0_K - 0.00649 * h_target_m;
    return p_meas_kPa / std::pow(T_K / T0_K, 5.256);
  }

  /// Convert a ground-temperature reading (°C, measured at altitude H) to
  /// the sea-level reference T0 [K] consistent with the lapse-rate model.
  static inline double T0_from_ground_temp_c(double T_ground_c,
                                             double H_known_m) {
    return (T_ground_c + 273.15) + 0.00649 * H_known_m;
  }
};

}  // namespace parnav
