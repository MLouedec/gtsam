/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 *  @file  PreintegrationBaseT.h
 *  @author Luca Carlone
 *  @author Stephen Williams
 *  @author Richard Roberts
 *  @author Vadim Indelman
 *  @author David Jensen
 *  @author Frank Dellaert
 *  @author Varun Agrawal
 **/

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/navigation/PreintegrationBaseT.h>

#include <cassert>
#include <stdexcept>

#include "gtsam/navigation/ImuBias.h"

using namespace std;

namespace gtsam {

//------------------------------------------------------------------------------
// template <typename Bias>
// PreintegrationBaseT::PreintegrationBaseT(const std::shared_ptr<Params>& p,
//                                             const Bias& biasHat)
//    : p_(p), biasHat_(biasHat), deltaTij_(0.0) {}

//------------------------------------------------------------------------------
template <typename Bias>
ostream& operator<<(ostream& os, const PreintegrationBaseT<Bias>& pim) {
  os << "    deltaTij = " << pim.deltaTij_ << endl;
  os << "    deltaRij.ypr = (" << pim.deltaRij().ypr().transpose() << ")"
     << endl;
  os << "    deltaPij = " << pim.deltaPij().transpose() << endl;
  os << "    deltaVij = " << pim.deltaVij().transpose() << endl;
  os << "    gyrobias = " << pim.biasHat_.gyroscope().transpose() << endl;
  os << "    acc_bias = " << pim.biasHat_.accelerometer().transpose() << endl;
  return os;
}

//------------------------------------------------------------------------------
template <typename Bias>
void PreintegrationBaseT<Bias>::print(const string& s) const {
  cout << (s.empty() ? s : s + "\n") << *this << endl;
}

//------------------------------------------------------------------------------
template <typename Bias>
void PreintegrationBaseT<Bias>::resetIntegrationAndSetBias(const Bias& biasHat) {
  biasHat_ = biasHat;
  resetIntegration();
}

//------------------------------------------------------------------------------
template <typename Bias>
pair<Vector3, Vector3>
PreintegrationBaseT<Bias>::correctMeasurementsBySensorPose(
    const Vector3& unbiasedAcc, const Vector3& unbiasedOmega,
    OptionalJacobian<3, 3> correctedAcc_H_unbiasedAcc,
    OptionalJacobian<3, 3> correctedAcc_H_unbiasedOmega,
    OptionalJacobian<3, 3> correctedOmega_H_unbiasedOmega) const {
  assert(p().body_P_sensor);

  // Compensate for sensor-body displacement if needed: we express the
  // quantities (originally in the IMU frame) into the body frame Equations
  // below assume the "body" frame is the CG

  // Get sensor to body rotation matrix
  const Matrix3 bRs = p().body_P_sensor->rotation().matrix();

  // Convert angular velocity and acceleration from sensor to body frame
  Vector3 correctedAcc = bRs * unbiasedAcc;
  const Vector3 correctedOmega = bRs * unbiasedOmega;

  // Jacobians
  if (correctedAcc_H_unbiasedAcc) *correctedAcc_H_unbiasedAcc = bRs;
  if (correctedAcc_H_unbiasedOmega) *correctedAcc_H_unbiasedOmega = Z_3x3;
  if (correctedOmega_H_unbiasedOmega) *correctedOmega_H_unbiasedOmega = bRs;

  // Centrifugal acceleration
  const Vector3 b_arm = p().body_P_sensor->translation();
  if (!b_arm.isZero()) {
    // Subtract out the the centripetal acceleration from the unbiased one
    // to get linear acceleration vector in the body frame:
    const Matrix3 body_Omega_body = skewSymmetric(correctedOmega);
    const Vector3 b_velocity_bs =
        body_Omega_body * b_arm;  // magnitude: omega * arm
    correctedAcc -= body_Omega_body * b_velocity_bs;

    // Update derivative: centrifugal causes the correlation between acc and
    // omega!!!
    if (correctedAcc_H_unbiasedOmega) {
      double wdp = correctedOmega.dot(b_arm);
      const Matrix3 diag_wdp = Vector3::Constant(wdp).asDiagonal();
      *correctedAcc_H_unbiasedOmega =
          -(diag_wdp + correctedOmega * b_arm.transpose()) * bRs.matrix() +
          2 * b_arm * unbiasedOmega.transpose();
    }
  }

  return make_pair(correctedAcc, correctedOmega);
}

//------------------------------------------------------------------------------
template <typename Bias>
void PreintegrationBaseT<Bias>::integrateMeasurement(
    const Vector3& measuredAcc, const Vector3& measuredOmega, double dt) {
  // NOTE(frank): integrateMeasurement always needs to compute the derivatives,
  // even when not of interest to the caller. Provide scratch space here.
  Matrix9 A;
  Matrix93 B, C;
  update(measuredAcc, measuredOmega, dt, &A, &B, &C);
}

//------------------------------------------------------------------------------
template <typename Bias>
NavState PreintegrationBaseT<Bias>::predict(const NavState& state_i,
                                           const Bias& bias_i,
                                           OptionalJacobian<9, 9> H1,
                                           OptionalJacobian<9, 6> H2) const {
  Matrix96 D_biasCorrected_bias;
  Vector9 biasCorrected =
      biasCorrectedDelta(bias_i, H2 ? &D_biasCorrected_bias : nullptr);

  // Correct the preintegrated delta for the initial velocity and gravity.
  // This reproduces the (now deprecation-gated) NavState::correctPIM verbatim
  // using only public NavState API, so it is mathematically identical to the
  // upstream inertial prediction. D_delta_state = d(xi)/d(state_i); the
  // Jacobian d(xi)/d(biasCorrected) is the identity (correctPIM's H2).
  // Tangent order is NavState's [dR, dP, dV].
  const double dt = deltaTij_;
  const double dt22 = 0.5 * dt * dt;
  const Rot3 nRb = state_i.attitude();
  const Velocity3 n_v = state_i.velocity();  // derivative wrt state is Ri
  const Vector3& n_gravity = p().n_gravity;
  const auto& omegaCoriolis = p().omegaCoriolis;

  Vector9 xi;
  Matrix3 D_dP_Ri1, D_dP_Ri2, D_dP_nv, D_dV_Ri;
  NavState::dR(xi) = NavState::dR(biasCorrected);
  NavState::dP(xi) =
      NavState::dP(biasCorrected) +
      dt * nRb.unrotate(n_v, H1 ? &D_dP_Ri1 : nullptr, H1 ? &D_dP_nv : nullptr) +
      dt22 * nRb.unrotate(n_gravity, H1 ? &D_dP_Ri2 : nullptr);
  NavState::dV(xi) =
      NavState::dV(biasCorrected) +
      dt * nRb.unrotate(n_gravity, H1 ? &D_dV_Ri : nullptr);

  // The rotating-frame (Coriolis) term relied on NavState::coriolis(), which is
  // deprecation-gated in 4.3.0. The sim runs in an inertial frame
  // (omegaCoriolis unset); fail loudly rather than silently drop it.
  if (omegaCoriolis && !omegaCoriolis->isZero(0.0)) {
    throw std::runtime_error(
        "PreintegrationBaseT::predict: omegaCoriolis (rotating-frame dynamics) "
        "is not supported in the Gauss-Markov NavState path (inertial only).");
  }
  Matrix9 D_delta_state;
  if (H1) {
    D_delta_state.setZero();
    const Matrix3 Ri = nRb.matrix();
    D_delta_state.block<3, 3>(3, 0) += dt * D_dP_Ri1 + dt22 * D_dP_Ri2;  // dP/dR
    D_delta_state.block<3, 3>(3, 6) += dt * D_dP_nv * Ri;                // dP/dV
    D_delta_state.block<3, 3>(6, 0) += dt * D_dV_Ri;                     // dV/dR
  }

  // Retract back to the NavState manifold. d(xi)/d(biasCorrected) == I.
  Matrix9 D_predict_state, D_predict_delta;
  NavState state_j = state_i.retract(xi, H1 ? &D_predict_state : nullptr,
                                     H1 || H2 ? &D_predict_delta : nullptr);
  if (H1) *H1 = D_predict_state + D_predict_delta * D_delta_state;
  if (H2) *H2 = D_predict_delta * D_biasCorrected_bias;
  return state_j;
}

//------------------------------------------------------------------------------
template <typename Bias>
Vector9 PreintegrationBaseT<Bias>::computeError(
    const NavState& state_i, const NavState& state_j, const Bias& bias_i,
    OptionalJacobian<9, 9> H1, OptionalJacobian<9, 9> H2,
    OptionalJacobian<9, 6> H3) const {
  // Predict state at time j
  Matrix9 D_predict_state_i;
  Matrix96 D_predict_bias_i;
  NavState predictedState_j = predict(
      state_i, bias_i, H1 ? &D_predict_state_i : 0, H3 ? &D_predict_bias_i : 0);

  // Calculate error
  Matrix9 D_error_state_j, D_error_predict;
  Vector9 error =
      state_j.localCoordinates(predictedState_j, H2 ? &D_error_state_j : 0,
                               H1 || H3 ? &D_error_predict : 0);

  if (H1) *H1 << D_error_predict * D_predict_state_i;
  if (H2) *H2 << D_error_state_j;
  if (H3) *H3 << D_error_predict * D_predict_bias_i;

  return error;
}

//------------------------------------------------------------------------------
template <typename Bias>
Vector9 PreintegrationBaseT<Bias>::computeErrorAndJacobians(
    const Pose3& pose_i, const Vector3& vel_i, const Pose3& pose_j,
    const Vector3& vel_j, const Bias& bias_i, OptionalJacobian<9, 6> H1,
    OptionalJacobian<9, 3> H2, OptionalJacobian<9, 6> H3,
    OptionalJacobian<9, 3> H4, OptionalJacobian<9, 6> H5) const {
  // Note that derivative of constructors below is not identity for velocity,
  // but a 9*3 matrix == Z_3x3, Z_3x3, state.R().transpose()
  NavState state_i(pose_i, vel_i);
  NavState state_j(pose_j, vel_j);

  // Predict state at time j
  Matrix9 D_error_state_i, D_error_state_j;
  Vector9 error =
      computeError(state_i, state_j, bias_i, H1 || H2 ? &D_error_state_i : 0,
                   H3 || H4 ? &D_error_state_j : 0, H5);

  // Separate out derivatives in terms of 5 arguments
  // Note that doing so requires special treatment of velocities, as when
  // treated as separate variables the retract applied will not be the
  // semi-direct product in NavState Instead, the velocities in nav are updated
  // using a straight addition This is difference is accounted for by the
  // R().transpose calls below
  if (H1) *H1 << D_error_state_i.leftCols<6>();
  if (H2) *H2 << D_error_state_i.rightCols<3>() * state_i.R().transpose();
  if (H3) *H3 << D_error_state_j.leftCols<6>();
  if (H4) *H4 << D_error_state_j.rightCols<3>() * state_j.R().transpose();

  return error;
}

//------------------------------------------------------------------------------

}  // namespace gtsam

// Explicit instantiation
template class gtsam::PreintegrationBaseT<gtsam::imuBias::ConstantBias>;
template class gtsam::PreintegrationBaseT<gtsam::imuBias::GaussMarkovBias>;
template std::ostream& gtsam::operator<< <gtsam::imuBias::ConstantBias>(
    std::ostream& os,
    const gtsam::PreintegrationBaseT<gtsam::imuBias::ConstantBias>& pim);
template std::ostream& gtsam::operator<< <gtsam::imuBias::GaussMarkovBias>(
    std::ostream& os,
    const gtsam::PreintegrationBaseT<gtsam::imuBias::GaussMarkovBias>& pim);
