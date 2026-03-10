/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    PreintegrationSE23Base.cpp
 * @brief   Predict + computeError for SE_2(3) IMU preintegration.
 */

#include <gtsam/navigation/PreintegrationSE23Base.h>

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/navigation/ImuBias.h>

#include <cassert>
#include <iostream>

namespace gtsam {

//------------------------------------------------------------------------------
template <typename Bias>
void PreintegrationSE23Base<Bias>::print(const std::string& s) const {
  std::cout << (s.empty() ? s : s + "\n")
            << "    deltaTij = " << deltaTij_ << "\n"
            << "    deltaRij.ypr = (" << deltaRij().ypr().transpose() << ")\n"
            << "    deltaVij = " << deltaVij().transpose() << "\n"
            << "    deltaPij = " << deltaPij().transpose() << "\n"
            << "    acc_bias = " << biasHat_.accelerometer().transpose() << "\n"
            << "    gyrobias = " << biasHat_.gyroscope().transpose()
            << std::endl;
}

//------------------------------------------------------------------------------
template <typename Bias>
std::pair<Vector3, Vector3>
PreintegrationSE23Base<Bias>::correctMeasurementsBySensorPose(
    const Vector3& unbiasedAcc, const Vector3& unbiasedOmega,
    OptionalJacobian<3, 3> correctedAcc_H_unbiasedAcc,
    OptionalJacobian<3, 3> correctedAcc_H_unbiasedOmega,
    OptionalJacobian<3, 3> correctedOmega_H_unbiasedOmega) const {
  assert(p().body_P_sensor);

  const Matrix3 bRs = p().body_P_sensor->rotation().matrix();

  Vector3 correctedAcc = bRs * unbiasedAcc;
  const Vector3 correctedOmega = bRs * unbiasedOmega;

  if (correctedAcc_H_unbiasedAcc) *correctedAcc_H_unbiasedAcc = bRs;
  if (correctedAcc_H_unbiasedOmega) *correctedAcc_H_unbiasedOmega = Z_3x3;
  if (correctedOmega_H_unbiasedOmega) *correctedOmega_H_unbiasedOmega = bRs;

  // Centrifugal acceleration for offset sensors.
  const Vector3 b_arm = p().body_P_sensor->translation();
  if (!b_arm.isZero()) {
    const Matrix3 body_Omega_body = skewSymmetric(correctedOmega);
    const Vector3 b_velocity_bs = body_Omega_body * b_arm;
    correctedAcc -= body_Omega_body * b_velocity_bs;

    if (correctedAcc_H_unbiasedOmega) {
      const double wdp = correctedOmega.dot(b_arm);
      const Matrix3 diag_wdp = Vector3::Constant(wdp).asDiagonal();
      *correctedAcc_H_unbiasedOmega =
          -(diag_wdp + correctedOmega * b_arm.transpose()) * bRs +
          2 * b_arm * unbiasedOmega.transpose();
    }
  }

  return std::make_pair(correctedAcc, correctedOmega);
}

//------------------------------------------------------------------------------
template <typename Bias>
void PreintegrationSE23Base<Bias>::integrateMeasurement(
    const Vector3& measuredAcc, const Vector3& measuredOmega, double dt) {
  Matrix9 A;
  Matrix93 B, C;
  update(measuredAcc, measuredOmega, dt, &A, &B, &C);
}

//------------------------------------------------------------------------------
// SE_2(3) prediction using the Brossard form:
//
//   X_pred = Gamma_t(g, dt) * Phi_t(X_i, dt) * Expmap(biasCorrectedDelta)
//
// where:
//   Phi_t(X_i, dt) = (R_i, v_i, p_i + dt * v_i)         — free-flight position step
//   Gamma_t(g, dt) = (I, dt * g, 0.5 * dt^2 * g)        — gravity contribution
//
// Left-multiplication order matters: Gamma is on the left so that the
// gravity vector is applied in the WORLD frame (not rotated by R_i).
// Direct expansion gives the expected kinematics
//   R_j = R_i * Exp(theta);  v_j = v_i + g*dt + R_i J_l nu;
//   p_j = p_i + v_i*dt + 0.5*g*dt^2 + R_i J_l rho.
//
// Jacobians are computed via chain rule across the compose operations.
//------------------------------------------------------------------------------
template <typename Bias>
Se23 PreintegrationSE23Base<Bias>::predict(
    const Se23& state_i, const Bias& bias_i,
    OptionalJacobian<9, 9> H1, OptionalJacobian<9, 6> H2) const {
  // 1. Bias-corrected preintegrated tangent.
  Matrix96 D_bc_bias;
  const Vector9 bc = biasCorrectedDelta(bias_i, H2 ? &D_bc_bias : nullptr);

  // 2. Lift to group: Delta = Expmap(bc).
  Matrix9 D_Delta_bc;
  const Se23 Delta =
      Se23::Expmap(bc, (H1 || H2) ? &D_Delta_bc : nullptr);

  // 3. Phi_t(X_i, dt) = (R_i, v_i, p_i + dt * v_i).
  //    Jacobian of Phi wrt X_i computed below in tangent coords.
  const Se23 Phi(state_i.rotation(), state_i.x(0),
                          state_i.x(1) + deltaTij_ * state_i.x(0));

  // Derivative of Phi (9-dim tangent at Phi) w.r.t. X_i (9-dim tangent at X_i).
  // Parameterise perturbations: X_i' = X_i * Exp(xi_body),
  // Phi' has
  //   R'   = R_i Exp(theta)
  //   v'   = v_i + R_i J_l(theta) nu
  //   p'+ dt v' = p_i + R_i J_l(theta) rho + dt (v_i + R_i J_l(theta) nu)
  //             = (p_i + dt v_i) + R_i J_l(theta) (rho + dt nu).
  // To first order: d(theta_Phi)/d(xi_body) = I on the theta slot;
  //                 d(nu_Phi) / d(xi_body) = I on the nu slot;
  //                 d(rho_Phi)/d(xi_body) = I on rho, plus dt * I from nu (cross coupling).
  // Then translate into Phi's own body tangent using Phi's rotation R_Phi = R_i
  // (unchanged), i.e., the Jacobian above is directly in Phi body tangent.
  Matrix9 D_Phi_Xi = Matrix9::Zero();
  D_Phi_Xi.block<3, 3>(0, 0) = I_3x3;
  D_Phi_Xi.block<3, 3>(3, 3) = I_3x3;
  D_Phi_Xi.block<3, 3>(6, 6) = I_3x3;
  D_Phi_Xi.block<3, 3>(6, 3) = deltaTij_ * I_3x3;

  // 4. Gamma_t(g, dt): constant element (depends only on params).
  const Vector3& g = p().n_gravity;
  const double dt22 = 0.5 * deltaTij_ * deltaTij_;
  const Se23 Gamma(Rot3::Identity(), deltaTij_ * g, dt22 * g);

  // 5. Compose X_pred = Gamma * Phi * Delta.
  //    Accumulate Jacobians w.r.t. the two variable inputs: Phi (depends on X_i) and Delta (depends on bias_i).
  Matrix9 D_GP_Phi;           // d(Gamma*Phi) / d(Phi)
  Matrix9 D_XPred_GP;         // d((Gamma*Phi)*Delta) / d(Gamma*Phi)
  Matrix9 D_XPred_Delta;      // d((Gamma*Phi)*Delta) / d(Delta)
  Matrix9 D_GP_Gamma_unused;  // placeholder (Gamma constant, not needed)

  const Se23 GP = Gamma.compose(
      Phi, (H1 || H2) ? &D_GP_Gamma_unused : nullptr,
      (H1 || H2) ? &D_GP_Phi : nullptr);
  (void)D_GP_Gamma_unused;

  const Se23 X_pred = GP.compose(
      Delta, (H1 || H2) ? &D_XPred_GP : nullptr,
      (H1 || H2) ? &D_XPred_Delta : nullptr);

  if (H1) {
    *H1 = D_XPred_GP * D_GP_Phi * D_Phi_Xi;
  }
  if (H2) {
    *H2 = D_XPred_Delta * D_Delta_bc * D_bc_bias;
  }
  return X_pred;
}

//------------------------------------------------------------------------------
template <typename Bias>
Vector9 PreintegrationSE23Base<Bias>::computeError(
    const Se23& state_i, const Se23& state_j,
    const Bias& bias_i, OptionalJacobian<9, 9> H1, OptionalJacobian<9, 9> H2,
    OptionalJacobian<9, 6> H3) const {
  Matrix9 D_predict_state_i;
  Matrix96 D_predict_bias_i;
  const Se23 predictedState_j = predict(
      state_i, bias_i, H1 ? &D_predict_state_i : nullptr,
      H3 ? &D_predict_bias_i : nullptr);

  Matrix9 D_error_state_j, D_error_predict;
  const Vector9 error = state_j.localCoordinates(
      predictedState_j, H2 ? &D_error_state_j : nullptr,
      (H1 || H3) ? &D_error_predict : nullptr);

  if (H1) *H1 = D_error_predict * D_predict_state_i;
  if (H2) *H2 = D_error_state_j;
  if (H3) *H3 = D_error_predict * D_predict_bias_i;

  return error;
}

}  // namespace gtsam

// Explicit instantiations.
template class gtsam::PreintegrationSE23Base<gtsam::imuBias::ConstantBias>;
template class gtsam::PreintegrationSE23Base<gtsam::imuBias::GaussMarkovBias>;
