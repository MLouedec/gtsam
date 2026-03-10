/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file  ManifoldPreintegrationSE23.h
 *  @brief Manifold IMU preintegration on SE_2(3) (Barrau / Brossard ordering).
 *
 *  Sibling of ManifoldPreintegration<Bias>, but:
 *    - state lives in gtsam::Se23 (R, v, p) with tangent [theta, nu, rho]
 *    - per-step propagation uses the left-trivialized transition matrix F_dt and
 *      measurement Jacobian G_j from parnav::helpers::SE23.
 *
 *  Templated on Bias so that both imuBias::ConstantBias and
 *  imuBias::GaussMarkovBias instantiate cleanly.
 */

#pragma once

#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/navigation/PreintegrationSE23Base.h>

namespace gtsam {

/// Single-step increment model for the SE_2(3) preintegration (chooses both the
/// mean increment Ypsilon_hat and the matched measurement Jacobian G_j):
///   - SimpleGlobalAcc : constant GLOBAL acceleration over the step
///                       (dv = f_hat*dt, dp = f_hat*dt^2/2). Cheapest.
///   - ConstantBodyImu : constant BODY specific force over the step
///                       (dv = J_l(w*dt) f_hat dt, dp = C_hat(w,dt) f_hat).
/// The two must be paired with their matching G_j (Brossard Remark 5).
enum class SE23IncrementModel { SimpleGlobalAcc, ConstantBodyImu };

template <typename Bias = imuBias::ConstantBias>
class GTSAM_EXPORT ManifoldPreintegrationSE23
    : public PreintegrationSE23Base<Bias> {
 protected:
  using PreintegrationSE23Base<Bias>::biasHat_;
  using PreintegrationSE23Base<Bias>::deltaTij_;
  using PreintegrationSE23Base<Bias>::p_;
  using PreintegrationSE23Base<Bias>::p;

 public:
  using Params = typename PreintegrationSE23Base<Bias>::Params;
  using PreintegrationSE23Base<Bias>::deltaTij;
  using PreintegrationSE23Base<Bias>::params;
  using PreintegrationSE23Base<Bias>::biasHat;
  using PreintegrationSE23Base<Bias>::correctMeasurementsBySensorPose;

  /// Single-step increment model (mean + matched G_j). Default: cheapest.
  SE23IncrementModel incrementModel_ = SE23IncrementModel::SimpleGlobalAcc;

  /// Preintegrated navigation state on SE_2(3), from i to j.
  Se23 deltaXij_;

  /// Bias Jacobian blocks. Output rows follow the SE_2(3) tangent
  /// [theta, nu, rho]; input columns follow ImuBias ordering [acc, gyro].
  Matrix3 delRdelBiasOmega_;    ///< d(theta) / d(bias_gyro)
  Matrix3 delNUdelBiasAcc_;     ///< d(nu)    / d(bias_acc)
  Matrix3 delNUdelBiasOmega_;   ///< d(nu)    / d(bias_gyro)  (stays 0 in base Brossard)
  Matrix3 delRHOdelBiasAcc_;    ///< d(rho)   / d(bias_acc)
  Matrix3 delRHOdelBiasOmega_;  ///< d(rho)   / d(bias_gyro)

  /// Default ctor for serialization.
  ManifoldPreintegrationSE23() { resetIntegration(); }

  /// Constructor from params + bias (+ optional increment model).
  ManifoldPreintegrationSE23(
      const std::shared_ptr<Params>& p, const Bias& biasHat = Bias(),
      SE23IncrementModel incrementModel = SE23IncrementModel::SimpleGlobalAcc)
      : PreintegrationSE23Base<Bias>(p, biasHat),
        incrementModel_(incrementModel) {
    resetIntegration();
  }

  /// Which single-step increment model this preintegrator uses.
  SE23IncrementModel incrementModel() const { return incrementModel_; }

  /// @name Basic utilities
  /// @{
  void resetIntegration() override;
  /// @}

  /// @name Instance variables access
  /// @{
  Se23 deltaXij() const override { return deltaXij_; }
  Rot3 deltaRij() const override { return deltaXij_.rotation(); }
  Vector3 deltaVij() const override { return deltaXij_.x(0); }
  Vector3 deltaPij() const override { return deltaXij_.x(1); }

  Matrix3 delRdelBiasOmega() const { return delRdelBiasOmega_; }
  Matrix3 delNUdelBiasAcc() const { return delNUdelBiasAcc_; }
  Matrix3 delNUdelBiasOmega() const { return delNUdelBiasOmega_; }
  Matrix3 delRHOdelBiasAcc() const { return delRHOdelBiasAcc_; }
  Matrix3 delRHOdelBiasOmega() const { return delRHOdelBiasOmega_; }
  /// @}

  /// @name Testable
  /// @{
  bool equals(const ManifoldPreintegrationSE23& other, double tol) const;
  /// @}

  /// @name Main functionality
  /// @{

  /// Advance preintegration by one IMU sample.
  ///   A = d(xi_new)/d(xi_old)  (9x9, from F_dt)
  ///   B = d(xi_new)/d(acc)     (9x3, from G_j cols 0-2)
  ///   C = d(xi_new)/d(omega)   (9x3, from G_j cols 3-5)
  void update(const Vector3& measuredAcc, const Vector3& measuredOmega,
              const double dt, Matrix9* A, Matrix93* B, Matrix93* C,
              Vector3* correctedAcc = nullptr,
              Vector3* correctedOmega = nullptr) override;

  /// 9-vector tangent (Barrau ordering) of the bias-corrected preintegrated delta.
  Vector9 biasCorrectedDelta(const Bias& bias_i,
                             OptionalJacobian<9, 6> H = {}) const override;

  /// Clone for MATLAB.
  virtual std::shared_ptr<ManifoldPreintegrationSE23> clone() const {
    return std::shared_ptr<ManifoldPreintegrationSE23>();
  }
  /// @}

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    namespace bs = ::boost::serialization;
    ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(PreintegrationSE23Base<Bias>);
    ar& BOOST_SERIALIZATION_NVP(deltaXij_);
    ar& BOOST_SERIALIZATION_NVP(delRdelBiasOmega_);
    ar& BOOST_SERIALIZATION_NVP(delNUdelBiasAcc_);
    ar& BOOST_SERIALIZATION_NVP(delNUdelBiasOmega_);
    ar& BOOST_SERIALIZATION_NVP(delRHOdelBiasAcc_);
    ar& BOOST_SERIALIZATION_NVP(delRHOdelBiasOmega_);
  }
#endif
};

}  // namespace gtsam
