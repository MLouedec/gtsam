/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 *  @file  ManifoldPreintegrationT.cpp
 *  @author Luca Carlone
 *  @author Stephen Williams
 *  @author Richard Roberts
 *  @author Vadim Indelman
 *  @author David Jensen
 *  @author Frank Dellaert
 **/

#include "ManifoldPreintegrationT.h"

#include "gtsam/navigation/ImuBias.h"

using namespace std;

namespace gtsam {

namespace {
// Kernels replicating gtsam/navigation/ManifoldPreintegrationSE23.cpp so the
// NavState (SE3) legacy path uses the SAME constant-body-IMU mean increment as
// the SE_2(3) path (piecewise-constant body specific force / angular rate).
inline Matrix3 so3_J_l_(const Vector3& theta) {
  const double phi = theta.norm();
  const Matrix3 S = skewSymmetric(theta);
  if (phi < 1e-8) return I_3x3 + 0.5 * S + (1.0 / 6.0) * S * S;
  const double phi2 = phi * phi;
  return I_3x3 + ((1.0 - std::cos(phi)) / phi2) * S +
         ((phi - std::sin(phi)) / (phi2 * phi)) * S * S;
}
inline Matrix3 se23_C_hat_(const Vector3& w_hat, double dt) {
  const double w = w_hat.norm();
  const Matrix3 Sw = skewSymmetric(w_hat);
  const double dt2 = dt * dt;
  Matrix3 C = 0.5 * dt2 * I_3x3;
  if (w < 1e-8)
    return C + (-dt2 * dt / 3.0) * Sw + (dt2 * dt2 / 8.0) * Sw * Sw;
  const double th = w * dt, w2 = w * w, w3 = w2 * w, w4 = w2 * w2;
  C += ((w * dt * std::cos(th) - std::sin(th)) / w3) * Sw +
       ((0.5 * w2 * dt2 - std::cos(th) - w * dt * std::sin(th) + 1.0) / w4) * Sw *
           Sw;
  return C;
}
inline Matrix3 se23_pos_kernel_(const Vector3& w, double dt) {
  return dt * dt * so3_J_l_(w * dt) - se23_C_hat_(-w, dt);
}
// One gravity-free constant-body-IMU delta step on a NavState (R, t, v).
inline NavState cbiMeanStep(const NavState& X, const Vector3& acc,
                            const Vector3& omega, double dt) {
  const Rot3 dR = Rot3::Expmap(omega * dt);
  const Vector3 dv_b = so3_J_l_(omega * dt) * acc * dt;
  const Vector3 dp_b = se23_pos_kernel_(omega, dt) * acc;
  const Rot3 R = X.attitude();
  return NavState(R * dR, X.position() + X.velocity() * dt + R.matrix() * dp_b,
                  X.velocity() + R.matrix() * dv_b);
}
}  // namespace

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename Bias>
void ManifoldPreintegrationT<Bias>::resetIntegration() {
  deltaTij_ = 0.0;
  deltaXij_ = NavState();
  delRdelBiasOmega_.setZero();
  delPdelBiasAcc_.setZero();
  delPdelBiasOmega_.setZero();
  delVdelBiasAcc_.setZero();
  delVdelBiasOmega_.setZero();
}

//------------------------------------------------------------------------------
template <typename Bias>
bool ManifoldPreintegrationT<Bias>::equals(const ManifoldPreintegrationT& other,
                                          double tol) const {
  return p_->equals(*other.p_, tol) &&
         std::abs(deltaTij_ - other.deltaTij_) < tol &&
         biasHat_.equals(other.biasHat_, tol) &&
         deltaXij_.equals(other.deltaXij_, tol) &&
         equal_with_abs_tol(delRdelBiasOmega_, other.delRdelBiasOmega_, tol) &&
         equal_with_abs_tol(delPdelBiasAcc_, other.delPdelBiasAcc_, tol) &&
         equal_with_abs_tol(delPdelBiasOmega_, other.delPdelBiasOmega_, tol) &&
         equal_with_abs_tol(delVdelBiasAcc_, other.delVdelBiasAcc_, tol) &&
         equal_with_abs_tol(delVdelBiasOmega_, other.delVdelBiasOmega_, tol);
}

//------------------------------------------------------------------------------
template <typename Bias>
void ManifoldPreintegrationT<Bias>::update(const Vector3& measuredAcc,
                                          const Vector3& measuredOmega,
                                          const double dt, Matrix9* A,
                                          Matrix93* B, Matrix93* C) {
  // Correct for bias in the sensor frame. For GaussMarkovBias with in-window
  // decay (gmDecay_, default) step k is debiased with beta_k * b_i,
  // beta_k = exp(-t_k/tau), t_k = deltaTij_ (time *before* this step). beta = 1
  // for ConstantBias or frozen GM. The same beta scales the bias Jacobian.
  double beta_acc = 1.0, beta_omega = 1.0;
  if constexpr (std::is_same_v<Bias, imuBias::GaussMarkovBias>) {
    if (gmDecay_) {
      beta_acc = std::exp(-deltaTij_ / biasHat_.tauAcc());
      beta_omega = std::exp(-deltaTij_ / biasHat_.tauGyro());
    }
  }
  Vector3 acc = measuredAcc - beta_acc * biasHat_.accelerometer();
  Vector3 omega = measuredOmega - beta_omega * biasHat_.gyroscope();

  // Possibly correct for sensor pose
  Matrix3 D_correctedAcc_acc, D_correctedAcc_omega, D_correctedOmega_omega;
  if (p().body_P_sensor) {
    std::tie(acc, omega) = this->correctMeasurementsBySensorPose(
        acc, omega, D_correctedAcc_acc, D_correctedAcc_omega,
        D_correctedOmega_omega);
  }

  // ================= GtsamStandard: upstream NavState::update =================
  if (increment_ == LegacyIncrement::GtsamStandard) {
    const Rot3 oldRij = deltaXij_.attitude();
    deltaTij_ += dt;
    deltaXij_ = deltaXij_.update(acc, omega, dt, A, B, C);  // functional
    if (p().body_P_sensor) {
      *C *= D_correctedOmega_omega;
      if (!p().body_P_sensor->translation().isZero())
        *C += *B * D_correctedAcc_omega;
      *B *= D_correctedAcc_acc;  // must be last
    }
    Matrix3 D_acc_R;
    oldRij.rotate(acc, D_acc_R);
    const Matrix3 D_acc_biasOmega = D_acc_R * delRdelBiasOmega_;
    const Vector3 integratedOmega = omega * dt;
    Matrix3 D_incrR_integratedOmega;
    const Rot3 incrR = Rot3::Expmap(integratedOmega, D_incrR_integratedOmega);
    const Matrix3 incrRt = incrR.transpose();
    const double dt22 = 0.5 * dt * dt;
    const Matrix3 dRij = oldRij.matrix();
    delRdelBiasOmega_ =
        incrRt * delRdelBiasOmega_ - beta_omega * D_incrR_integratedOmega * dt;
    delPdelBiasAcc_ += delVdelBiasAcc_ * dt - beta_acc * dt22 * dRij;
    delPdelBiasOmega_ += dt * delVdelBiasOmega_ + dt22 * D_acc_biasOmega;
    delVdelBiasAcc_ += -beta_acc * dRij * dt;
    delVdelBiasOmega_ += D_acc_biasOmega * dt;
    return;
  }

  // ================= ConstantBodyImu: piecewise (matches SE_2(3)) =============
  const NavState oldX = deltaXij_;
  const NavState newX = cbiMeanStep(oldX, acc, omega, dt);

  // --- Jacobians by central finite differences, in NavState [R, t, v] tangent.
  //     A = d(newX)/d(oldX), B = d(newX)/d(acc), C = d(newX)/d(omega). Using FD
  //     keeps them consistent with the constant-body-IMU mean (the SE_2(3) path
  //     likewise finite-differences its input Jacobian).
  const double eps = 1e-6, inv2e = 1.0 / (2.0 * eps);
  for (int i = 0; i < 9; ++i) {
    Vector9 d = Vector9::Zero();
    d(i) = eps;
    A->col(i) = newX.localCoordinates(cbiMeanStep(oldX.retract(d), acc, omega,
                                                  dt)) *
                    inv2e -
                newX.localCoordinates(cbiMeanStep(oldX.retract(-d), acc, omega,
                                                  dt)) *
                    inv2e;
  }
  for (int j = 0; j < 3; ++j) {
    Vector3 e = Vector3::Zero();
    e(j) = eps;
    B->col(j) = (newX.localCoordinates(cbiMeanStep(oldX, acc + e, omega, dt)) -
                 newX.localCoordinates(cbiMeanStep(oldX, acc - e, omega, dt))) *
                inv2e;
    C->col(j) = (newX.localCoordinates(cbiMeanStep(oldX, acc, omega + e, dt)) -
                 newX.localCoordinates(cbiMeanStep(oldX, acc, omega - e, dt))) *
                inv2e;
  }

  deltaTij_ += dt;
  deltaXij_ = newX;

  if (p().body_P_sensor) {
    *C *= D_correctedOmega_omega;
    if (!p().body_P_sensor->translation().isZero()) *C += *B * D_correctedAcc_omega;
    *B *= D_correctedAcc_acc;  // must be last
  }

  // --- Bias Jacobian J = d(preint delta)/d(bias_i) via J <- A*J + G_bias.
  //     acc = meas - beta_acc*bias_acc, omega = meas - beta_omega*bias_omega, so
  //     d(delta)/d(bias) = [-beta_acc*B | -beta_omega*C] (beta from the bias
  //     correction above). Blocks are in NavState [R, t(P), v(V)] x [acc, gyro].
  //     A, B, C live in the NavState tangent chart (component-wise: dp_abs =
  //     R * dp_tan, same for v), but the stored p/v blocks are absolute, as
  //     biasCorrectedDelta adds them to deltaPij()/deltaVij() directly. So
  //     rotate p/v rows into the tangent at oldX, recurse, rotate back at newX.
  const Matrix3 R_old = oldX.attitude().matrix();
  const Matrix3 R_new = newX.attitude().matrix();
  Matrix96 J = Matrix96::Zero();
  J.block<3, 3>(0, 3) = delRdelBiasOmega_;
  J.block<3, 3>(3, 0) = R_old.transpose() * delPdelBiasAcc_;
  J.block<3, 3>(3, 3) = R_old.transpose() * delPdelBiasOmega_;
  J.block<3, 3>(6, 0) = R_old.transpose() * delVdelBiasAcc_;
  J.block<3, 3>(6, 3) = R_old.transpose() * delVdelBiasOmega_;
  Matrix96 Gb = Matrix96::Zero();
  Gb.block<9, 3>(0, 0) = -beta_acc * (*B);
  Gb.block<9, 3>(0, 3) = -beta_omega * (*C);
  J = (*A) * J + Gb;
  delRdelBiasOmega_ = J.block<3, 3>(0, 3);
  delPdelBiasAcc_ = R_new * J.block<3, 3>(3, 0);
  delPdelBiasOmega_ = R_new * J.block<3, 3>(3, 3);
  delVdelBiasAcc_ = R_new * J.block<3, 3>(6, 0);
  delVdelBiasOmega_ = R_new * J.block<3, 3>(6, 3);
}

//------------------------------------------------------------------------------
template <typename Bias>
Vector9 ManifoldPreintegrationT<Bias>::biasCorrectedDelta(
    const Bias& bias_i, OptionalJacobian<9, 6> H) const {
  // Correct deltaRij, derivative is delRdelBiasOmega_
  const Bias biasIncr = bias_i - biasHat_;
  Matrix3 D_correctedRij_bias;
  const Vector3 biasInducedOmega = delRdelBiasOmega_ * biasIncr.gyroscope();
  const Rot3 correctedRij =
      deltaRij().expmap(biasInducedOmega, {}, H ? &D_correctedRij_bias : 0);
  if (H) D_correctedRij_bias *= delRdelBiasOmega_;

  Vector9 xi;
  Matrix3 D_dR_correctedRij;
  // TODO(frank): could line below be simplified? It is equivalent to
  //   LogMap(deltaRij_.compose(Expmap(biasInducedOmega)))
  NavState::dR(xi) = Rot3::Logmap(correctedRij, H ? &D_dR_correctedRij : 0);
  NavState::dP(xi) = deltaPij() + delPdelBiasAcc_ * biasIncr.accelerometer() +
                     delPdelBiasOmega_ * biasIncr.gyroscope();
  NavState::dV(xi) = deltaVij() + delVdelBiasAcc_ * biasIncr.accelerometer() +
                     delVdelBiasOmega_ * biasIncr.gyroscope();

  if (H) {
    Matrix36 D_dR_bias, D_dP_bias, D_dV_bias;
    D_dR_bias << Z_3x3, D_dR_correctedRij * D_correctedRij_bias;
    D_dP_bias << delPdelBiasAcc_, delPdelBiasOmega_;
    D_dV_bias << delVdelBiasAcc_, delVdelBiasOmega_;
    (*H) << D_dR_bias, D_dP_bias, D_dV_bias;
  }
  return xi;
}

//------------------------------------------------------------------------------

}  // namespace gtsam

// Explicit instantiation
template class gtsam::ManifoldPreintegrationT<gtsam::imuBias::ConstantBias>;
template class gtsam::ManifoldPreintegrationT<gtsam::imuBias::GaussMarkovBias>;
