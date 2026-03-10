/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 *  @file  CombinedImuFactor.h
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
#include <gtsam/navigation/ManifoldPreintegrationT.h>
#include <gtsam/navigation/PreintegrationCombinedParamsT.h>
#include <gtsam/navigation/TangentPreintegration.h>

#include <ostream>

#include "gtsam/navigation/ImuBias.h"

namespace gtsam {

/*
 * If you are using the factor, please cite:
 * L. Carlone, Z. Kira, C. Beall, V. Indelman, F. Dellaert, Eliminating
 * conditionally independent sets in factor graphs: a unifying perspective based
 * on smart factors, Int. Conf. on Robotics and Automation (ICRA), 2014.
 *
 * REFERENCES:
 * [1] G.S. Chirikjian, "Stochastic Models, Information Theory, and Lie Groups",
 *     Volume 2, 2008.
 * [2] T. Lupton and S.Sukkarieh, "Visual-Inertial-Aided Navigation for
 *     High-Dynamic Motion in Built Environments Without Initial Conditions",
 *     TRO, 28(1):61-76, 2012.
 * [3] L. Carlone, S. Williams, R. Roberts, "Preintegrated IMU factor:
 *     Computation of the Jacobian Matrices", Tech. Report, 2013.
 *     Available in this repo as "PreintegratedIMUJacobians.pdf".
 * [4] C. Forster, L. Carlone, F. Dellaert, D. Scaramuzza, IMU Preintegration on
 *     Manifold for Efficient Visual-Inertial Maximum-a-Posteriori Estimation,
 *     Robotics: Science and Systems (RSS), 2015.
 */

/**
 * PreintegratedCombinedMeasurements integrates the IMU measurements
 * (rotation rates and accelerations) and the corresponding covariance matrix.
 * The measurements are then used to build the CombinedImuFactor. Integration
 * is done incrementally (ideally, one integrates the measurement as soon as
 * it is received from the IMU) so as to avoid costly integration at time of
 * factor construction.
 *
 * @ingroup navigation
 */
template <class PreintegrationType, class BiasType>
class GTSAM_EXPORT PreintegratedCombinedMeasurementsGMT
    : public PreintegrationType {
 public:
  typedef PreintegrationCombinedParamsT<BiasType> Params;

 protected:
  /* Covariance matrix of the preintegrated measurements
   * COVARIANCE OF: [PreintROTATION PreintPOSITION PreintVELOCITY BiasAcc
   * BiasOmega] (first-order propagation from *measurementCovariance*).
   * PreintegratedCombinedMeasurements also include the biases and keep the
   * correlation between the preintegrated measurements and the biases
   */
  Eigen::Matrix<double, 15, 15> preintMeasCov_;

  template <class PIM, class BIAS>
  friend class CombinedImuFactorGMT;

 public:
  /// @name Constructors
  /// @{

  /// Default constructor only for serialization and wrappers
  PreintegratedCombinedMeasurementsGMT() { this->resetIntegration(); }

  /**
   *  Default constructor, initializes the class with no measurements
   *  @param p Parameters, typically fixed in a single application
   *  @param biasHat Current estimate of acceleration and rotation rate biases
   *  @param preintMeasCov Covariance matrix used in noise model.
   */
  PreintegratedCombinedMeasurementsGMT(
      const std::shared_ptr<Params>& p, const BiasType& biasHat = BiasType(),
      const Eigen::Matrix<double, 15, 15>& preintMeasCov =
          Eigen::Matrix<double, 15, 15>::Zero())
      : PreintegrationType(p, biasHat), preintMeasCov_(preintMeasCov) {
    this->PreintegrationType::resetIntegration();
  }

  /**
   *  Construct preintegrated directly from members: base class and
   * preintMeasCov
   *  @param base               PreintegrationType instance
   *  @param preintMeasCov      Covariance matrix used in noise model.
   */
  PreintegratedCombinedMeasurementsGMT(
      const PreintegrationType& base,
      const Eigen::Matrix<double, 15, 15>& preintMeasCov)
      : PreintegrationType(base), preintMeasCov_(preintMeasCov) {
    this->PreintegrationType::resetIntegration();
  }

  /// Virtual destructor
  ~PreintegratedCombinedMeasurementsGMT() override {}

  /// @}

  /// @name Basic utilities
  /// @{

  /// Re-initialize PreintegratedCombinedMeasurements
  void resetIntegration() override;

  /// const reference to params, shadows definition in base class
  Params& p() const { return *std::static_pointer_cast<Params>(this->p_); }
  /// @}

  /// @name Access instance variables
  /// @{
  /// Return pre-integrated measurement covariance
  Matrix preintMeasCov() const { return preintMeasCov_; }
  /// @}

  /// @name Testable
  /// @{
  /// print
  void print(
      const std::string& s = "Preintegrated Measurements:") const override;
  /// equals
  bool equals(const PreintegratedCombinedMeasurementsGMT<PreintegrationType,
                                                       BiasType>& expected,
              double tol = 1e-9) const;
  /// @}

  /// @name Main functionality
  /// @{

  /**
   * Add a single IMU measurement to the preintegration.
   * Both accelerometer and gyroscope measurements are taken to be in the sensor
   * frame and conversion to the body frame is handled by `body_P_sensor` in
   * `PreintegrationParams`.
   *
   * @param measuredAcc Measured acceleration (as given by the sensor)
   * @param measuredOmega Measured angular velocity (as given by the sensor)
   * @param dt Time interval between two consecutive IMU measurements
   */
  void integrateMeasurement(const Vector3& measuredAcc,
                            const Vector3& measuredOmega,
                            const double dt) override;

  /// @}

#ifdef GTSAM_ALLOW_DEPRECATED_SINCE_V43
  /// @deprecated: biasAccOmegaInt is no longer used. Use a prior on first bias
  /// instead.
  void resetIntegration(const gtsam::Matrix6& Q_init) {
    std::cerr
        << "Warning: setBiasAccOmegaInit() is deprecated and no longer used."
        << std::endl;
    PreintegrationType::resetIntegration();
    preintMeasCov_.setZero();
  }
#endif

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION  ///
  /// Serialization function
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    namespace bs = ::boost::serialization;
    ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(PreintegrationType);
    ar& BOOST_SERIALIZATION_NVP(preintMeasCov_);
  }
#endif

};

/**
 * CombinedImuFactor is a 6-ways factor involving previous state (pose and
 * velocity of the vehicle, as well as bias at previous time step), and current
 * state (pose, velocity, bias at current time step). Following the pre-
 * integration scheme proposed in [2], the CombinedImuFactor includes many IMU
 * measurements, which are "summarized" using the
 * PreintegratedCombinedMeasurements class. There are 3 main differences wrpt
 * the ImuFactor class: 1) The factor is 6-ways, meaning that it also involves
 * both biases (previous and current time step).Therefore, the factor internally
 * imposes the biases to be slowly varying; in particular, the matrices
 * "biasAccCovariance" and "biasOmegaCovariance" described the random walk that
 * models bias evolution. 2) The preintegration covariance takes into account
 * the noise in the bias estimate used for integration. 3) The covariance matrix
 * of the PreintegratedCombinedMeasurements preserves the correlation between
 * the bias uncertainty and the preintegrated measurements uncertainty.
 *
 * @ingroup navigation
 */
template <class PIM, class BIAS>
class GTSAM_EXPORT CombinedImuFactorGMT
    : public NoiseModelFactorN<Pose3, Vector3, Pose3, Vector3, BIAS, BIAS> {
 public:
 private:
  typedef CombinedImuFactorGMT<PIM, BIAS> This;
  typedef NoiseModelFactorN<Pose3, Vector3, Pose3, Vector3, BIAS, BIAS> Base;

  PIM pim_;

 public:
  // Provide access to Matrix& version of evaluateError:
  using Base::evaluateError;

  /** Shorthand for a smart pointer to a factor */
  typedef std::shared_ptr<This> shared_ptr;

  /** Default constructor - only use for serialization */
  CombinedImuFactorGMT() {}

  /**
   * Constructor
   * @param pose_i Previous pose key
   * @param vel_i  Previous velocity key
   * @param pose_j Current pose key
   * @param vel_j  Current velocity key
   * @param bias_i Previous bias key
   * @param bias_j Current bias key
   * @param PreintegratedCombinedMeasurements Combined IMU measurements
   */
  CombinedImuFactorGMT(Key pose_i, Key vel_i, Key pose_j, Key vel_j, Key bias_i,
                     Key bias_j, const PIM& preintegratedMeasurements)
      : Base(noiseModel::Gaussian::Covariance(
                 preintegratedMeasurements.preintMeasCov()),
             pose_i, vel_i, pose_j, vel_j, bias_i, bias_j),
        pim_(preintegratedMeasurements) {}

  ~CombinedImuFactorGMT() override {}

  /// @return a deep copy of this factor
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<This>(*this);
  }

  /** implement functions needed for Testable */

  /// @name Testable
  /// @{
  /// print
  void print(const std::string& s = "", const KeyFormatter& keyFormatter =
                                            DefaultKeyFormatter) const override;

  /// equals
  bool equals(const NonlinearFactor& expected,
              double tol = 1e-9) const override;
  /// @}

  /** Access the preintegrated measurements. */

  const PIM& preintegratedMeasurements() const { return pim_; }

  /** implement functions needed to derive from Factor */

  /// vector of errors
  Vector evaluateError(const Pose3& pose_i, const Vector3& vel_i,
                       const Pose3& pose_j, const Vector3& vel_j,
                       const BIAS& bias_i, const BIAS& bias_j,
                       OptionalMatrixType H1, OptionalMatrixType H2,
                       OptionalMatrixType H3, OptionalMatrixType H4,
                       OptionalMatrixType H5,
                       OptionalMatrixType H6) const override;

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  /** Serialization function */
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    // NoiseModelFactor6 instead of NoiseModelFactorN for backward compatibility
    ar& boost::serialization::make_nvp(
        "NoiseModelFactor6", boost::serialization::base_object<Base>(*this));
    ar& BOOST_SERIALIZATION_NVP(pim_);
  }
#endif

};
// class CombinedImuFactorGMT

// operator<< for CombinedImuFactorGMT
template <class PIM, class BIAS>
GTSAM_EXPORT std::ostream& operator<<(std::ostream& os,
                                      const CombinedImuFactorGMT<PIM, BIAS>& f);


template <class PreintegrationType, class BiasType>
struct traits<PreintegratedCombinedMeasurementsGMT<PreintegrationType, BiasType>>
    : public Testable<
          PreintegratedCombinedMeasurementsGMT<PreintegrationType, BiasType>> {};

template <class PIM, class BIAS>
struct traits<CombinedImuFactorGMT<PIM, BIAS>>
    : public Testable<CombinedImuFactorGMT<PIM, BIAS>> {};

// NOTE: The legacy NavState-based CombinedImuFactor2T has been removed.
// The replacement — an SE_2(3) / ExtendedPose3 / bias-templated factor —
// lives in <gtsam/navigation/CombinedImuFactor2.h>.

//==============================================================================
// Template method implementations
//==============================================================================

// sugar for derivative blocks
#define D_R_R(H) (H)->block<3, 3>(0, 0)
#define D_R_t(H) (H)->block<3, 3>(0, 3)
#define D_R_v(H) (H)->block<3, 3>(0, 6)
#define D_t_R(H) (H)->block<3, 3>(3, 0)
#define D_t_t(H) (H)->block<3, 3>(3, 3)
#define D_t_v(H) (H)->block<3, 3>(3, 6)
#define D_v_R(H) (H)->block<3, 3>(6, 0)
#define D_v_t(H) (H)->block<3, 3>(6, 3)
#define D_v_v(H) (H)->block<3, 3>(6, 6)
#define D_a_a(H) (H)->block<3, 3>(9, 9)
#define D_g_g(H) (H)->block<3, 3>(12, 12)

//------------------------------------------------------------------------------
// PreintegratedCombinedMeasurementsGMT implementations
//------------------------------------------------------------------------------

template <class PreintegrationType, class BiasType>
void PreintegratedCombinedMeasurementsGMT<PreintegrationType, BiasType>::print(
    const std::string& s) const {
  PreintegrationType::print(s);
  std::cout << "  preintMeasCov [ " << preintMeasCov_ << " ]" << std::endl;
}

template <class PreintegrationType, class BiasType>
bool PreintegratedCombinedMeasurementsGMT<PreintegrationType, BiasType>::equals(
    const PreintegratedCombinedMeasurementsGMT<PreintegrationType, BiasType>&
        other,
    double tol) const {
  return PreintegrationType::equals(other, tol) &&
         equal_with_abs_tol(preintMeasCov_, other.preintMeasCov_, tol);
}

template <class PreintegrationType, class BiasType>
void PreintegratedCombinedMeasurementsGMT<PreintegrationType,
                                        BiasType>::resetIntegration() {
  PreintegrationType::resetIntegration();
  preintMeasCov_.setZero();
}

template <class PreintegrationType, class BiasType>
void PreintegratedCombinedMeasurementsGMT<PreintegrationType, BiasType>::
    integrateMeasurement(const Vector3& measuredAcc,
                         const Vector3& measuredOmega, double dt) {
  if (dt <= 0) {
    throw std::runtime_error(
        "PreintegratedCombinedMeasurements::integrateMeasurement: dt <=0");
  }

  Matrix9 A;
  Matrix93 B, C;
  PreintegrationType::update(measuredAcc, measuredOmega, dt, &A, &B, &C);

  Matrix3 theta_H_omega = C.topRows<3>();
  Matrix3 pos_H_acc = B.middleRows<3>(3);
  Matrix3 vel_H_acc = B.bottomRows<3>();

  Eigen::Matrix<double, 15, 15> F;
  F.setZero();
  F.block<9, 9>(0, 0) = A;

  // Off-diagonal blocks: NavState dependence on bias.
  // B and C are Jacobians w.r.t. corrected measurements; to get Jacobians
  // w.r.t. bias we chain through the bias correction:
  //   ConstantBias:    d(corrected)/d(bias) = -I  (sign cancels in F*P*F^T)
  //   GaussMarkovBias: d(corrected)/d(bias) = -beta*I  (beta matters!)
  if constexpr (std::is_same_v<BiasType, imuBias::GaussMarkovBias>) {
    // deltaTij_ was already incremented by dt in update(), so subtract dt
    // to get the elapsed time at the start of this measurement step.
    const double t_k = this->deltaTij_ - dt;
    const double beta_acc = std::exp(-t_k / this->biasHat_.tauAcc());
    const double beta_omega = std::exp(-t_k / this->biasHat_.tauGyro());
    F.block<3, 3>(0, 12) = beta_omega * theta_H_omega;
    F.block<3, 3>(3, 9) = beta_acc * pos_H_acc;
    F.block<3, 3>(6, 9) = beta_acc * vel_H_acc;
  } else {
    F.block<3, 3>(0, 12) = theta_H_omega;
    F.block<3, 3>(3, 9) = pos_H_acc;
    F.block<3, 3>(6, 9) = vel_H_acc;
  }

  // Bias state transition:
  //   ConstantBias  (random walk):   I_6x6
  //   GaussMarkovBias (1st-order GM): diag(exp(-dt/tauAcc)*I3,
  //                                        exp(-dt/tauGyro)*I3)
  F.block<6, 6>(9, 9) = this->p().biasFTransition(dt, this->biasHat_);

  preintMeasCov_ = F * preintMeasCov_ * F.transpose();

  const Matrix3& aCov = this->p().accelerometerCovariance;
  const Matrix3& wCov = this->p().gyroscopeCovariance;
  const Matrix3& iCov = this->p().integrationCovariance;

  Eigen::Matrix<double, 15, 15> G_measCov_Gt;
  G_measCov_Gt.setZero(15, 15);

  D_R_R(&G_measCov_Gt) =
      (theta_H_omega * (wCov / dt) * theta_H_omega.transpose());
  D_t_t(&G_measCov_Gt) =
      (pos_H_acc * (aCov / dt) * pos_H_acc.transpose()) + (dt * iCov);
  D_v_v(&G_measCov_Gt) = (vel_H_acc * (aCov / dt) * vel_H_acc.transpose());

  // Bias process noise:
  //   ConstantBias:     Q_c * dt
  //   GaussMarkovBias:  Q_c * (1 - exp(-2*dt/tau)) / 2
  D_a_a(&G_measCov_Gt) =
      this->p().discreteBiasAccCovariance(dt, this->biasHat_);
  D_g_g(&G_measCov_Gt) =
      this->p().discreteBiasOmegaCovariance(dt, this->biasHat_);

  D_t_v(&G_measCov_Gt) = (pos_H_acc * (aCov / dt) * vel_H_acc.transpose());
  D_v_t(&G_measCov_Gt) = (vel_H_acc * (aCov / dt) * pos_H_acc.transpose());

  preintMeasCov_.noalias() += G_measCov_Gt;
}
#undef D_R_R
#undef D_R_t
#undef D_R_v
#undef D_t_R
#undef D_t_t
#undef D_t_v
#undef D_v_R
#undef D_v_t
#undef D_v_v
#undef D_a_a
#undef D_g_g

//------------------------------------------------------------------------------
// CombinedImuFactorGMT implementations
//------------------------------------------------------------------------------

template <class PIM, class BIAS>
void CombinedImuFactorGMT<PIM, BIAS>::print(
    const std::string& s, const KeyFormatter& keyFormatter) const {
  std::cout << (s.empty() ? s : s + "\n") << "CombinedImuFactor("
            << keyFormatter(this->template key<1>()) << ","
            << keyFormatter(this->template key<2>()) << ","
            << keyFormatter(this->template key<3>()) << ","
            << keyFormatter(this->template key<4>()) << ","
            << keyFormatter(this->template key<5>()) << ","
            << keyFormatter(this->template key<6>()) << ")\n";
  pim_.print("  preintegrated measurements:");
  this->noiseModel_->print("  noise model: ");
}

template <class PIM, class BIAS>
bool CombinedImuFactorGMT<PIM, BIAS>::equals(const NonlinearFactor& other,
                                           double tol) const {
  const This* e = dynamic_cast<const This*>(&other);
  return e != nullptr && Base::equals(*e, tol) && pim_.equals(e->pim_, tol);
}

template <class PIM, class BIAS>
Vector CombinedImuFactorGMT<PIM, BIAS>::evaluateError(
    const Pose3& pose_i, const Vector3& vel_i, const Pose3& pose_j,
    const Vector3& vel_j, const BIAS& bias_i, const BIAS& bias_j,
    OptionalMatrixType H1, OptionalMatrixType H2, OptionalMatrixType H3,
    OptionalMatrixType H4, OptionalMatrixType H5, OptionalMatrixType H6) const {
  // Compute the bias state transition over the full preintegration interval.
  // For ConstantBias: F_total = I_6x6
  // For GaussMarkovBias: F_total = diag(exp(-T/tauAcc)*I3, exp(-T/tauGyro)*I3)
  const Matrix6 F_total =
      pim_.p().biasFTransition(pim_.deltaTij(), pim_.biasHat());

  // Bias error: predicted bias_j minus actual bias_j
  //   predicted bias_j = F_total * bias_i (+ noise)
  //   error = F_total * bias_i - bias_j
  // For ConstantBias (F_total=I) this reduces to bias_i - bias_j.
  const Vector6 fbias = F_total * bias_i.vector() - bias_j.vector();

  Matrix96 D_r_pose_i, D_r_pose_j, D_r_bias_i;
  Matrix93 D_r_vel_i, D_r_vel_j;

  Vector9 r_Rpv = pim_.computeErrorAndJacobians(
      pose_i, vel_i, pose_j, vel_j, bias_i, H1 ? &D_r_pose_i : 0,
      H2 ? &D_r_vel_i : 0, H3 ? &D_r_pose_j : 0, H4 ? &D_r_vel_j : 0,
      H5 ? &D_r_bias_i : 0);

  if (H1) {
    H1->resize(15, 6);
    H1->block<9, 6>(0, 0) = D_r_pose_i;
    H1->block<6, 6>(9, 0).setZero();
  }
  if (H2) {
    H2->resize(15, 3);
    H2->block<9, 3>(0, 0) = D_r_vel_i;
    H2->block<6, 3>(9, 0).setZero();
  }
  if (H3) {
    H3->resize(15, 6);
    H3->block<9, 6>(0, 0) = D_r_pose_j;
    H3->block<6, 6>(9, 0).setZero();
  }
  if (H4) {
    H4->resize(15, 3);
    H4->block<9, 3>(0, 0) = D_r_vel_j;
    H4->block<6, 3>(9, 0).setZero();
  }
  if (H5) {
    H5->resize(15, 6);
    H5->block<9, 6>(0, 0) = D_r_bias_i;
    // d(fbias)/d(bias_i) = F_total
    H5->block<6, 6>(9, 0) = F_total;
  }
  if (H6) {
    H6->resize(15, 6);
    H6->block<9, 6>(0, 0).setZero();
    // d(fbias)/d(bias_j) = -I
    H6->block<6, 6>(9, 0) = -I_6x6;
  }

  Vector r(15);
  r << r_Rpv, fbias;
  return r;
}

template <class PIM, class BIAS>
std::ostream& operator<<(std::ostream& os,
                         const CombinedImuFactorGMT<PIM, BIAS>& f) {
  f.preintegratedMeasurements().print("combined preintegrated measurements:\n");
  os << "  noise model sigmas: " << f.noiseModel()->sigmas().transpose();
  return os;
}

}  // namespace gtsam
//
