/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   CombinedImuFactor2.h
 * @brief  SE_2(3) CombinedImuFactor2: 4-way factor on (state_i, state_j,
 *         bias_i, bias_j) using Se23 (Barrau ordering) and a
 *         bias-templated SE_2(3) preintegrator.
 *
 * Covariance block layout in preintMeasCov_ (15 x 15):
 *   [theta(3), nu(3), rho(3), bias_acc(3), bias_gyro(3)]
 * note: (nu, rho) order, NOT the legacy (rho, nu).
 */

#pragma once

#include <gtsam/navigation/ManifoldPreintegrationSE23.h>
#include <gtsam/navigation/PreintegrationCombinedParamsT.h>
#include <gtsam/navigation/SE23CovariancePropagation.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <ostream>

namespace gtsam {

/**
 * SE_2(3) preintegration + 15x15 covariance tracking bias correlations.
 *
 * Inherits from `PIM`, which must be a subclass of
 * `PreintegrationSE23Base<BiasType>` (default: `ManifoldPreintegrationSE23`).
 */
template <class PreintegrationType, class BiasType>
class GTSAM_EXPORT PreintegratedCombinedMeasurements2T
    : public PreintegrationType {
 public:
  typedef PreintegrationCombinedParamsT<BiasType> Params;

 protected:
  /// 15x15 covariance in [theta, nu, rho, bias_acc, bias_gyro] order.
  Eigen::Matrix<double, 15, 15> preintMeasCov_;

  /// Discrete process-noise method (Brossard / Ours / VanLoan).
  SE23CovarianceMethod covMethod_ = SE23CovarianceMethod::Brossard;

  template <class PIM, class BIAS>
  friend class CombinedImuFactor2T;

 public:
  /// @name Constructors
  /// @{

  /// Default ctor (serialization / wrappers only).
  PreintegratedCombinedMeasurements2T() { this->resetIntegration(); }

  PreintegratedCombinedMeasurements2T(
      const std::shared_ptr<Params>& p, const BiasType& biasHat = BiasType(),
      const Eigen::Matrix<double, 15, 15>& preintMeasCov =
          Eigen::Matrix<double, 15, 15>::Zero(),
      SE23IncrementModel incrementModel = SE23IncrementModel::SimpleGlobalAcc,
      SE23CovarianceMethod covMethod = SE23CovarianceMethod::Brossard)
      : PreintegrationType(p, biasHat, incrementModel),
        preintMeasCov_(preintMeasCov),
        covMethod_(covMethod) {
    this->PreintegrationType::resetIntegration();
  }

  /// Which discrete process-noise method this preintegrator uses.
  SE23CovarianceMethod covMethod() const { return covMethod_; }

  PreintegratedCombinedMeasurements2T(
      const PreintegrationType& base,
      const Eigen::Matrix<double, 15, 15>& preintMeasCov)
      : PreintegrationType(base), preintMeasCov_(preintMeasCov) {
    this->PreintegrationType::resetIntegration();
  }

  ~PreintegratedCombinedMeasurements2T() override {}

  /// @}
  /// @name Basic utilities
  /// @{

  void resetIntegration() override {
    PreintegrationType::resetIntegration();
    preintMeasCov_.setZero();
  }

  /// const-reference to typed params; shadows the base class
  Params& p() const { return *std::static_pointer_cast<Params>(this->p_); }

  /// @}
  /// @name Accessors
  /// @{

  /// Return preintegrated measurement covariance.
  Matrix preintMeasCov() const { return preintMeasCov_; }

  /// @}
  /// @name Testable
  /// @{

  void print(const std::string& s =
                 "Preintegrated SE_2(3) Measurements:") const override {
    PreintegrationType::print(s);
    std::cout << "  preintMeasCov [ " << preintMeasCov_ << " ]" << std::endl;
  }

  bool equals(
      const PreintegratedCombinedMeasurements2T<PreintegrationType, BiasType>&
          expected,
      double tol = 1e-9) const {
    return PreintegrationType::equals(expected, tol) &&
           equal_with_abs_tol(preintMeasCov_, expected.preintMeasCov_, tol);
  }

  /// @}
  /// @name Main functionality
  /// @{

  void integrateMeasurement(const Vector3& measuredAcc,
                            const Vector3& measuredOmega,
                            const double dt) override;

  /// @}

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    namespace bs = ::boost::serialization;
    ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(PreintegrationType);
    ar& BOOST_SERIALIZATION_NVP(preintMeasCov_);
  }
#endif

};

//==============================================================================
// SE_2(3) covariance propagation
//==============================================================================

// sugar for 15x15 derivative blocks in the new (theta, nu, rho, ba, bg) layout.
#define SE23_CIF_D_theta_theta(H) (H)->block<3, 3>(0, 0)
#define SE23_CIF_D_nu_nu(H) (H)->block<3, 3>(3, 3)
#define SE23_CIF_D_rho_rho(H) (H)->block<3, 3>(6, 6)
#define SE23_CIF_D_nu_rho(H) (H)->block<3, 3>(3, 6)
#define SE23_CIF_D_rho_nu(H) (H)->block<3, 3>(6, 3)
#define SE23_CIF_D_ba_ba(H) (H)->block<3, 3>(9, 9)
#define SE23_CIF_D_bg_bg(H) (H)->block<3, 3>(12, 12)

template <class PreintegrationType, class BiasType>
void PreintegratedCombinedMeasurements2T<PreintegrationType, BiasType>::
    integrateMeasurement(const Vector3& measuredAcc,
                         const Vector3& measuredOmega, double dt) {
  if (dt <= 0) {
    throw std::runtime_error(
        "PreintegratedCombinedMeasurements2::integrateMeasurement: dt <= 0");
  }

  // 1. Underlying SE_2(3) state update + per-step Jacobians.
  //    A : 9x9 tangent transition; B : 9x3 d(xi_inc)/d(acc); C : 9x3 d(xi_inc)/d(omega).
  //    f_hat/w_hat are the corrected specific force / rate (for the continuous
  //    covariance methods).
  Matrix9 A;
  Matrix93 B, C;
  Vector3 f_hat, w_hat;
  PreintegrationType::update(measuredAcc, measuredOmega, dt, &A, &B, &C, &f_hat,
                             &w_hat);

  // Row blocks for the SE_2(3) tangent: [0..2] theta, [3..5] nu, [6..8] rho.
  const Matrix3 theta_H_omega = C.topRows<3>();
  const Matrix3 nu_H_acc = B.middleRows<3>(3);
  const Matrix3 rho_H_acc = B.bottomRows<3>();

  // 2. Build the full 15x15 transition A_i in [theta, nu, rho, ba, bg] order.
  //    Its top-left 9x9 block is the SE_2(3) transition A returned by update().
  Eigen::Matrix<double, 15, 15> A_i;
  A_i.setZero();
  A_i.block<9, 9>(0, 0) = A;

  // Off-diagonal pose<->bias coupling. d(corrected)/d(bias) = -I (beta = 1),
  // because the IMU is debiased with the FROZEN window-start bias b_i for BOTH
  // bias types (see ManifoldPreintegrationSE23::update). The off-diag gets the
  // *positive* measurement Jacobian (sign cancels via A_i P A_i^T). The
  // bias-block self-decay (GM mean reversion) lives in A_i.block<6,6>(9,9)
  // below, NOT here.
  // ALT (disabled): mid-window GM mean reversion scaled the coupling by
  //   beta_* = exp(-(deltaTij_-dt)/tau_*):
  //     A_i.block<3,3>(0,12) = beta_omega * theta_H_omega; etc.
  A_i.block<3, 3>(0, 12) = theta_H_omega;  // theta <- b_gyro
  A_i.block<3, 3>(3, 9) = nu_H_acc;        // nu    <- b_acc
  A_i.block<3, 3>(6, 9) = rho_H_acc;       // rho   <- b_acc

  // Bias block transition (identity for ConstantBias, exp(-dt/tau)*I for GM).
  A_i.block<6, 6>(9, 9) = this->p().biasFTransition(dt, this->biasHat_);

  preintMeasCov_ = A_i * preintMeasCov_ * A_i.transpose();

  // 3. Process-noise injection Q_d, chosen by covMethod_.
  const Matrix3& aCov = this->p().accelerometerCovariance;
  const Matrix3& wCov = this->p().gyroscopeCovariance;
  const Matrix3& iCov = this->p().integrationCovariance;

  Eigen::Matrix<double, 15, 15> G_measCov_Gt;
  G_measCov_Gt.setZero();

  if (covMethod_ == SE23CovarianceMethod::Brossard) {
    // Discrete gain reconstruction G (Qc/dt) G^T (ref §6c), using the FULL 9x6
    // G = [B | C]. Exact for SimpleGlobalAcc (the classic block form) and also
    // captures the ConstantBodyImu gyro->nu, gyro->rho cross terms.
    G_measCov_Gt.block<9, 9>(0, 0).noalias() =
        B * (aCov / dt) * B.transpose() + C * (wCov / dt) * C.transpose();
    G_measCov_Gt.block<3, 3>(6, 6).noalias() += dt * iCov;  // integration r.w.
    // Bias process noise (GM: discrete OU covariance; CB: Qc*dt).
    SE23_CIF_D_ba_ba(&G_measCov_Gt) =
        this->p().discreteBiasAccCovariance(dt, this->biasHat_);
    SE23_CIF_D_bg_bg(&G_measCov_Gt) =
        this->p().discreteBiasOmegaCovariance(dt, this->biasHat_);
  } else {
    // Continuous-time methods: build F_c, Q~ and integrate over dt. The full
    // 15x15 Q_d already carries the bias blocks AND the bias<->pose coupling,
    // so it replaces the block form above wholesale.
    double biasFcAcc = 0.0, biasFcGyro = 0.0;  // Wiener default
    if constexpr (std::is_same_v<BiasType, imuBias::GaussMarkovBias>) {
      biasFcAcc = -1.0 / this->biasHat_.tauAcc();
      biasFcGyro = -1.0 / this->biasHat_.tauGyro();
    }
    const SE23Covariance Fc = se23ContinuousFc(w_hat, f_hat, biasFcAcc, biasFcGyro);
    const SE23Covariance Qtil = se23ContinuousQtilde(
        aCov, wCov, iCov, this->p().biasAccCovariance,
        this->p().biasOmegaCovariance);
    G_measCov_Gt = (covMethod_ == SE23CovarianceMethod::Ours)
                       ? se23DiscreteQd_Ours(Fc, Qtil, dt)
                       : se23DiscreteQd_VanLoan(Fc, Qtil, dt);
  }

  preintMeasCov_.noalias() += G_measCov_Gt;
}

#undef SE23_CIF_D_theta_theta
#undef SE23_CIF_D_nu_nu
#undef SE23_CIF_D_rho_rho
#undef SE23_CIF_D_nu_rho
#undef SE23_CIF_D_rho_nu
#undef SE23_CIF_D_ba_ba
#undef SE23_CIF_D_bg_bg

//==============================================================================
// CombinedImuFactor2T: 4-way factor on (state_i, state_j, bias_i, bias_j).
//==============================================================================

/**
 * 4-way combined IMU factor on Se23 (SE_2(3)) states plus biases.
 * Uses the bias-templated SE_2(3) preintegrator, so it supports both
 * `ConstantBias` (Wiener) and `GaussMarkovBias` models.
 *
 * Residual layout (15 x 1): [r_SE23 (9), r_bias (6)].
 * Bias residual: r_bias = bias_j - F_total * bias_i, where
 *   F_total = I_6x6                       (ConstantBias)
 *   F_total = diag(exp(-T/tauAcc)*I3, exp(-T/tauGyro)*I3)  (GaussMarkovBias)
 */
template <class PIM = PreintegratedCombinedMeasurements2T<
                         ManifoldPreintegrationSE23<imuBias::ConstantBias>,
                         imuBias::ConstantBias>,
          class BIAS = imuBias::ConstantBias>
class GTSAM_EXPORT CombinedImuFactor2T
    : public NoiseModelFactorN<Se23, Se23, BIAS, BIAS> {
 private:
  typedef CombinedImuFactor2T<PIM, BIAS> This;
  typedef NoiseModelFactorN<Se23, Se23, BIAS, BIAS> Base;

  PIM pim_;

 public:
  using Base::evaluateError;

  typedef std::shared_ptr<This> shared_ptr;

  CombinedImuFactor2T() {}

  CombinedImuFactor2T(Key state_i, Key state_j, Key bias_i, Key bias_j,
                      const PIM& preintegratedMeasurements)
      : Base(noiseModel::Gaussian::Covariance(
                 preintegratedMeasurements.preintMeasCov()),
             state_i, state_j, bias_i, bias_j),
        pim_(preintegratedMeasurements) {}

  ~CombinedImuFactor2T() override {}

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<This>(*this);
  }

  /// @name Testable
  /// @{
  void print(const std::string& s = "", const KeyFormatter& keyFormatter =
                                            DefaultKeyFormatter) const override {
    std::cout << (s.empty() ? s : s + "\n") << "CombinedImuFactor2("
              << keyFormatter(this->template key<1>()) << ","
              << keyFormatter(this->template key<2>()) << ","
              << keyFormatter(this->template key<3>()) << ","
              << keyFormatter(this->template key<4>()) << ")\n";
    pim_.print("  preintegrated measurements:");
    this->noiseModel_->print("  noise model: ");
  }

  bool equals(const NonlinearFactor& other, double tol = 1e-9) const override {
    const This* e = dynamic_cast<const This*>(&other);
    return e != nullptr && Base::equals(*e, tol) && pim_.equals(e->pim_, tol);
  }
  /// @}

  const PIM& preintegratedMeasurements() const { return pim_; }

  Vector evaluateError(const Se23& state_i,
                       const Se23& state_j, const BIAS& bias_i,
                       const BIAS& bias_j, OptionalMatrixType H1,
                       OptionalMatrixType H2, OptionalMatrixType H3,
                       OptionalMatrixType H4) const override {
    // Bias transition over the full preintegration interval.
    //   ConstantBias:    F_total = I_6x6
    //   GaussMarkovBias: F_total = diag(exp(-T/tauAcc)*I3, exp(-T/tauGyro)*I3)
    const Matrix6 F_total =
        pim_.p().biasFTransition(pim_.deltaTij(), pim_.biasHat());

    // Bias residual r_bias = b_j - F_total * b_i  (ref §7b):
    //   d r_bias / d b_j = I ,  d r_bias / d b_i = -F_total.
    const Vector6 fbias = bias_j.vector() - F_total * bias_i.vector();

    Matrix9 D_r_state_i, D_r_state_j;
    Matrix96 D_r_bias_i;
    const Vector9 r_SE23 = pim_.computeError(
        state_i, state_j, bias_i, H1 ? &D_r_state_i : nullptr,
        H2 ? &D_r_state_j : nullptr, H3 ? &D_r_bias_i : nullptr);

    if (H1) {
      H1->resize(15, 9);
      H1->block<9, 9>(0, 0) = D_r_state_i;
      H1->block<6, 9>(9, 0).setZero();
    }
    if (H2) {
      H2->resize(15, 9);
      H2->block<9, 9>(0, 0) = D_r_state_j;
      H2->block<6, 9>(9, 0).setZero();
    }
    if (H3) {
      H3->resize(15, 6);
      H3->block<9, 6>(0, 0) = D_r_bias_i;
      H3->block<6, 6>(9, 0) = -F_total;  // d r_bias / d b_i
    }
    if (H4) {
      H4->resize(15, 6);
      H4->block<9, 6>(0, 0).setZero();
      H4->block<6, 6>(9, 0) = I_6x6;  // d r_bias / d b_j
    }

    Vector r(15);
    r << r_SE23, fbias;
    return r;
  }

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& boost::serialization::make_nvp(
        "NoiseModelFactor4", boost::serialization::base_object<Base>(*this));
    ar& BOOST_SERIALIZATION_NVP(pim_);
  }
#endif

};

/// Default alias — SE_2(3) + ConstantBias, replacing the legacy
/// NavState-based CombinedImuFactor2.
using CombinedImuFactor2 = CombinedImuFactor2T<>;

template <class PIM, class BIAS>
std::ostream& operator<<(std::ostream& os,
                         const CombinedImuFactor2T<PIM, BIAS>& f) {
  f.preintegratedMeasurements().print("combined preintegrated measurements:\n");
  os << "  noise model sigmas: " << f.noiseModel()->sigmas().transpose();
  return os;
}

// Traits for downstream Testable interfaces.
template <class PreintegrationType, class BiasType>
struct traits<PreintegratedCombinedMeasurements2T<PreintegrationType, BiasType>>
    : public Testable<PreintegratedCombinedMeasurements2T<PreintegrationType,
                                                          BiasType>> {};

template <class PIM, class BIAS>
struct traits<CombinedImuFactor2T<PIM, BIAS>>
    : public Testable<CombinedImuFactor2T<PIM, BIAS>> {};

}  // namespace gtsam
