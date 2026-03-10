#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include "JacobianTraits.hpp"

namespace parnav {

/// Barometric height factor that uses a pre-calibrated *pressure bias*
/// instead of a live additive bias state. Strategy B ("Known Point") from
/// the field-setup notes: during a stationary calibration at known altitude
/// `H_known` we compute
///
///   p_bias = p_measured_static - p0_std * (1 - L * H_known / T0)^(g*M/(R*L))
///
/// and subtract that from every subsequent measurement. The remaining
/// "corrected" pressure is then converted to altitude via the same
/// standard-day formula used by `heightOut.m`, and the factor constrains
///
///   -p_D == height_from_pressure(p_meas - p_bias) - base_height
///
/// (NED Down is positive-down, so the predicted "-p_D" equals altitude.)
///
/// Compared to `BaroFactor`, this is a single-key factor on `Pose` only --
/// no `D(0)` baro-bias state, no random-walk chain.
template <typename Pose>
class BaroFactorKnownPoint : public gtsam::NoiseModelFactorN<Pose> {
 private:
  using Base = gtsam::NoiseModelFactorN<Pose>;
  using Jacobian = Eigen::Matrix<double, 1, JacobianTraits<Pose>::Cols>;

  double base_height_;
  double height_rel_;  // z_rel = heightOut(pressure - p_bias) - base_height

 public:
  using Base::evaluateError;

  BaroFactorKnownPoint() = default;
  ~BaroFactorKnownPoint() override = default;

  /// @param pressure_kPa  raw pressure measurement [kPa]
  /// @param p_bias_kPa    sensor bias [kPa], pre-computed from the known-
  ///                      point strategy (positive => sensor reads high)
  /// @param base_height   nav-frame origin altitude [m]
  /// @param p0_kPa        local sea-level reference pressure [kPa]
  ///                      (default 101.29 = standard-day)
  /// @param T0_K          sea-level reference temperature [K]
  ///                      (default 288.08 = standard-day; lower on cold
  ///                      days, which shrinks dh/dp)
  BaroFactorKnownPoint(gtsam::Key pose_key, double pressure_kPa,
                       double p_bias_kPa, const gtsam::SharedNoiseModel& model,
                       double base_height = 0.0, double p0_kPa = 101.29,
                       double T0_K = 288.08)
      : Base(model, pose_key),
        base_height_(base_height),
        height_rel_(height_from_pressure(pressure_kPa - p_bias_kPa, p0_kPa,
                                         T0_K) -
                    base_height) {}

  auto evaluateError(const Pose& pose,
                     gtsam::OptionalMatrixType H1 = OptionalNone) const
      -> gtsam::Vector override;

  /// NASA-GRC simple troposphere model: pressure (kPa) -> height (m).
  /// Parameterised over `T0_K` so cold-day flights can correct the
  /// pressure-altitude slope (`dh/dp` scales linearly with T0).
  static inline double height_from_pressure(double n_kPa,
                                            double p0_kPa = 101.29,
                                            double T0_K = 288.08) {
    return (std::pow(n_kPa / p0_kPa, 1.0 / 5.256) * T0_K - T0_K) / -0.00649;
  }

  /// Forward model: altitude (m) -> pressure (kPa). Inverse of
  /// `height_from_pressure`.
  static inline double pressure_from_height(double h_m,
                                            double p0_kPa = 101.29,
                                            double T0_K = 288.08) {
    const double T_K = T0_K - 0.00649 * h_m;
    return p0_kPa * std::pow(T_K / T0_K, 5.256);
  }

  /// Strategy B "Known Point" bias.
  /// Positive return means the sensor reads higher than the model predicts
  /// for the known altitude.
  static inline double known_point_bias(double p_measured_kPa, double H_known_m,
                                        double p0_kPa = 101.29,
                                        double T0_K = 288.08) {
    return p_measured_kPa - pressure_from_height(H_known_m, p0_kPa, T0_K);
  }

  /// Solve for the local sea-level reference pressure that makes the
  /// hypsometric formula evaluate to `h_target` at `p_meas`. Inverse of
  /// `pressure_from_height`. Combine with `p_bias = 0` to reproduce the
  /// legacy BaroFactor's auto-calibration semantics. Pass the local `T0_K`
  /// for cold-day corrections.
  static inline double p0_from_known_altitude(double p_meas_kPa,
                                              double H_known_m,
                                              double T0_K = 288.08) {
    const double T_K = T0_K - 0.00649 * H_known_m;
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
