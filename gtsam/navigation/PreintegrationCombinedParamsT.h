/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 *  @file  PreintegrationCombinedParams.h
 *  @author Luca Carlone
 *  @author Stephen Williams
 *  @author Richard Roberts
 *  @author Vadim Indelman
 *  @author David Jensen
 *  @author Frank Dellaert
 *  @author Varun Agrawal
 **/

#pragma once

/* GTSAM includes */
#include <gtsam/base/Matrix.h>
#include <gtsam/base/MatrixConstants.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ManifoldPreintegration.h>
#include <gtsam/navigation/TangentPreintegration.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <cmath>
#include <type_traits>

namespace gtsam {

/// Parameters for pre-integration using PreintegratedCombinedMeasurements,
/// templated on the bias type so that Gauss-Markov dynamics can be
/// handled correctly during covariance propagation.
///
/// For ConstantBias (random walk):
///   F_bias = I_6x6,  Q_d = Q_c * dt
///
/// For GaussMarkovBias (1st-order Gauss-Markov):
///   F_bias = diag(exp(-dt/tauAcc)*I3, exp(-dt/tauGyro)*I3)
///   Q_d    = Q_c * (tau/2) * (1 - exp(-2*dt/tau))
///
template <class BIAS = imuBias::ConstantBias>
struct GTSAM_EXPORT PreintegrationCombinedParamsT : PreintegrationParams {
  Matrix3 biasAccCovariance;    ///< continuous-time "Covariance" describing
                                ///< accelerometer bias evolution
  Matrix3 biasOmegaCovariance;  ///< continuous-time "Covariance" describing
                                ///< gyroscope bias evolution

  /// Default constructor makes uninitialized params struct.
  /// Used for serialization.
  PreintegrationCombinedParamsT()
      : biasAccCovariance(I_3x3), biasOmegaCovariance(I_3x3) {
#ifdef GTSAM_ALLOW_DEPRECATED_SINCE_V43
    biasAccOmegaInt.setZero();
#endif
  }

  /// See two named constructors below for good values of n_gravity in body
  /// frame
  PreintegrationCombinedParamsT(const Vector3& n_gravity)
      : PreintegrationParams(n_gravity),
        biasAccCovariance(I_3x3),
        biasOmegaCovariance(I_3x3) {
#ifdef GTSAM_ALLOW_DEPRECATED_SINCE_V43
    biasAccOmegaInt.setZero();
#endif
  }

  // Default Params for a Z-down navigation frame, such as NED: gravity points
  // along positive Z-axis
  static std::shared_ptr<PreintegrationCombinedParamsT> MakeSharedD(
      double g = 9.81) {
    return std::shared_ptr<PreintegrationCombinedParamsT>(
        new PreintegrationCombinedParamsT(Vector3(0, 0, g)));
  }

  // Default Params for a Z-up navigation frame, such as ENU: gravity points
  // along negative Z-axis
  static std::shared_ptr<PreintegrationCombinedParamsT> MakeSharedU(
      double g = 9.81) {
    return std::shared_ptr<PreintegrationCombinedParamsT>(
        new PreintegrationCombinedParamsT(Vector3(0, 0, -g)));
  }

  //------------------------------------------------------------------------------
  // Inner class PreintegrationCombinedParams
  //------------------------------------------------------------------------------
  void print(const std::string& s = "") const override {
    PreintegrationParams::print(s);
    std::cout << "biasAccCovariance:\n[\n"
              << biasAccCovariance << "\n]" << std::endl;
    std::cout << "biasOmegaCovariance:\n[\n"
              << biasOmegaCovariance << "\n]" << std::endl;
  }

  bool equals(const PreintegratedRotationParams& other,
              double tol) const override {
    auto e = dynamic_cast<const PreintegrationCombinedParamsT*>(&other);
    return e != nullptr && PreintegrationParams::equals(other, tol) &&
           equal_with_abs_tol(biasAccCovariance, e->biasAccCovariance, tol) &&
           equal_with_abs_tol(biasOmegaCovariance, e->biasOmegaCovariance, tol);
  }

  void setBiasAccCovariance(const Matrix3& cov) { biasAccCovariance = cov; }
  void setBiasOmegaCovariance(const Matrix3& cov) { biasOmegaCovariance = cov; }

  const Matrix3& getBiasAccCovariance() const { return biasAccCovariance; }
  const Matrix3& getBiasOmegaCovariance() const { return biasOmegaCovariance; }

  // ---- Bias propagation helpers -----------------------------------------

  /**
   * Compute the discrete state-transition matrix for the bias block (6x6)
   * over an integration step of size dt.
   *
   * Needs access to the biasHat to read tau values for GaussMarkovBias.
   *
   * ConstantBias:     F_bias = I_6x6
   * GaussMarkovBias:  F_bias = diag(exp(-dt/tauAcc)*I3, exp(-dt/tauGyro)*I3)
   */
  Matrix6 biasFTransition(double dt, const BIAS& biasHat) const {
    if constexpr (std::is_same_v<BIAS, imuBias::GaussMarkovBias>) {
      Matrix6 F_bias;
      F_bias.setZero();
      F_bias.template topLeftCorner<3, 3>() =
          std::exp(-dt / biasHat.tauAcc()) * I_3x3;
      F_bias.template bottomRightCorner<3, 3>() =
          std::exp(-dt / biasHat.tauGyro()) * I_3x3;
      return F_bias;
    } else {
      (void)biasHat;  // unused
      return I_6x6;
    }
  }

  /**
   * Compute the discrete process-noise covariance for the accelerometer bias
   * block (3x3) over an integration step of size dt.
   *
   * For a 1st-order Gauss-Markov process  dx = -x/tau dt + sigma_c dW,
   * the exact discrete noise covariance is:
   *   Q_d = sigma_c^2 * (tau/2) * (1 - exp(-2*dt/tau))
   *
   * biasAccCovariance is sigma_c^2 (continuous PSD, same as ConstantBias).
   *
   * ConstantBias:     Q_d = biasAccCovariance * dt
   * GaussMarkovBias:  Q_d = biasAccCovariance * (tau/2) * (1 - exp(-2*dt/tau))
   */
  Matrix3 discreteBiasAccCovariance(double dt, const BIAS& biasHat) const {
    if constexpr (std::is_same_v<BIAS, imuBias::GaussMarkovBias>) {
      const double tau = biasHat.tauAcc();
      const double scale = (tau / 2.0) * (1.0 - std::exp(-2.0 * dt / tau));
      return biasAccCovariance * scale;
    } else {
      (void)biasHat;
      return biasAccCovariance * dt;
    }
  }

  /**
   * Compute the discrete process-noise covariance for the gyroscope bias
   * block (3x3) over an integration step of size dt.
   *
   * ConstantBias:     Q_d = biasOmegaCovariance * dt
   * GaussMarkovBias:  Q_d = biasOmegaCovariance * (tau/2) * (1 - exp(-2*dt/tau))
   */
  Matrix3 discreteBiasOmegaCovariance(double dt, const BIAS& biasHat) const {
    if constexpr (std::is_same_v<BIAS, imuBias::GaussMarkovBias>) {
      const double tau = biasHat.tauGyro();
      const double scale = (tau / 2.0) * (1.0 - std::exp(-2.0 * dt / tau));
      return biasOmegaCovariance * scale;
    } else {
      (void)biasHat;
      return biasOmegaCovariance * dt;
    }
  }

#ifdef GTSAM_ALLOW_DEPRECATED_SINCE_V43
  Matrix6 biasAccOmegaInt;
  /// @deprecated: biasAccOmegaInt is no longer used.
  void setBiasAccOmegaInit(const Matrix6& cov) {
    std::cerr << "Warning: setBiasAccOmegaInit() is deprecated and no longer "
                 "used."
              << std::endl;
    biasAccOmegaInt = cov;
  }
  /// @deprecated: biasAccOmegaInt is no longer used.
  const Matrix6& getBiasAccOmegaInit() const {
    std::cerr << "Warning: getBiasAccOmegaInit() is deprecated and no longer "
                 "used."
              << std::endl;
    return biasAccOmegaInt;
  }
#endif

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  /** Serialization function */
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    namespace bs = ::boost::serialization;
    ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(PreintegrationParams);
    ar& BOOST_SERIALIZATION_NVP(biasAccCovariance);
    ar& BOOST_SERIALIZATION_NVP(biasOmegaCovariance);
#ifdef GTSAM_ALLOW_DEPRECATED_SINCE_V43
    ar& BOOST_SERIALIZATION_NVP(biasAccOmegaInt);
#endif
  }
#endif
};

}  // namespace gtsam
