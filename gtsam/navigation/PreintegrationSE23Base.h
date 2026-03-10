/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   PreintegrationSE23Base.h
 *  @brief  SE_2(3) sibling of PreintegrationBase (Barrau ordering).
 *
 *  State type is gtsam::Se23 with (R, v, p) storage and
 *  tangent ordering [theta(3), nu(3), rho(3)].  Kept separate from
 *  PreintegrationBase<Bias> so that the legacy NavState-based factors
 *  (ImuFactor, ImuFactor2, CombinedImuFactor) remain untouched.
 */

#pragma once

#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/PreintegrationParams.h>

#include <iosfwd>
#include <memory>
#include <ostream>
#include <string>
#include <utility>

namespace gtsam {

/**
 * Base class for SE_2(3) IMU preintegration. Mirrors PreintegrationBase
 * but operates on Se23 (Barrau ordering).
 */
template <typename Bias = imuBias::ConstantBias>
class GTSAM_EXPORT PreintegrationSE23Base {
 public:
  using Params = PreintegrationParams;

  PreintegrationSE23Base(const std::shared_ptr<Params>& p,
                         const Bias& biasHat = Bias())
      : p_(p), biasHat_(biasHat), deltaTij_(0.0) {}

 protected:
  std::shared_ptr<Params> p_;
  Bias biasHat_;       ///< Bias used during preintegration
  double deltaTij_;    ///< Total integration time from i to j

  PreintegrationSE23Base() : deltaTij_(0.0) {}
  virtual ~PreintegrationSE23Base() = default;

 public:
  /// @name Basic utilities
  /// @{

  virtual void resetIntegration() = 0;

  void resetIntegrationAndSetBias(const Bias& biasHat) {
    biasHat_ = biasHat;
    resetIntegration();
  }

  bool matchesParamsWith(const PreintegrationSE23Base& other) const {
    return p_.get() == other.p_.get();
  }

  const std::shared_ptr<Params>& params() const { return p_; }
  Params& p() const { return *p_; }

  /// @}
  /// @name Instance variables access
  /// @{

  const Bias& biasHat() const { return biasHat_; }
  double deltaTij() const { return deltaTij_; }

  virtual Se23 deltaXij() const = 0;
  virtual Rot3 deltaRij() const = 0;
  virtual Vector3 deltaVij() const = 0;
  virtual Vector3 deltaPij() const = 0;

  Vector6 biasHatVector() const { return biasHat_.vector(); }

  /// @}
  /// @name Testable
  /// @{

  virtual void print(const std::string& s = "") const;

  /// @}
  /// @name Main functionality
  /// @{

  /**
   * Compensate for non-identity body_P_sensor (rotation + centrifugal acc).
   * Shared with the NavState preintegrator — same formulae, SE_2(3) is
   * irrelevant to this correction.
   */
  std::pair<Vector3, Vector3> correctMeasurementsBySensorPose(
      const Vector3& unbiasedAcc, const Vector3& unbiasedOmega,
      OptionalJacobian<3, 3> correctedAcc_H_unbiasedAcc = {},
      OptionalJacobian<3, 3> correctedAcc_H_unbiasedOmega = {},
      OptionalJacobian<3, 3> correctedOmega_H_unbiasedOmega = {}) const;

  /// Advance the integration with a new IMU sample. Optionally outputs the
  /// bias/sensor-corrected specific force and angular rate (f_hat, w_hat) used
  /// this step, which the covariance path needs to build the continuous F_c.
  virtual void update(const Vector3& measuredAcc, const Vector3& measuredOmega,
                      const double dt, Matrix9* A, Matrix93* B, Matrix93* C,
                      Vector3* correctedAcc = nullptr,
                      Vector3* correctedOmega = nullptr) = 0;

  /// Convenience overload without Jacobian outputs.
  virtual void integrateMeasurement(const Vector3& measuredAcc,
                                    const Vector3& measuredOmega,
                                    const double dt);

  /// Given the estimate of the bias, return a 9-vector tangent [theta, nu, rho]
  /// summarising the preintegrated IMU measurements so far.
  virtual Vector9 biasCorrectedDelta(const Bias& bias_i,
                                     OptionalJacobian<9, 6> H = {}) const = 0;

  /// Predict state at time j on SE_2(3).
  Se23 predict(const Se23& state_i, const Bias& bias_i,
                        OptionalJacobian<9, 9> H1 = {},
                        OptionalJacobian<9, 6> H2 = {}) const;

  /// Compute the 9-vector residual in SE_2(3) tangent (Barrau ordering).
  Vector9 computeError(const Se23& state_i,
                       const Se23& state_j, const Bias& bias_i,
                       OptionalJacobian<9, 9> H1, OptionalJacobian<9, 9> H2,
                       OptionalJacobian<9, 6> H3) const;

  /// @}

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& BOOST_SERIALIZATION_NVP(p_);
    ar& BOOST_SERIALIZATION_NVP(biasHat_);
    ar& BOOST_SERIALIZATION_NVP(deltaTij_);
  }
#endif

};

}  // namespace gtsam
