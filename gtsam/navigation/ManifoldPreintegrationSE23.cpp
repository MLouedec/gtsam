/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file  ManifoldPreintegrationSE23.cpp
 *  @brief Implementation of SE_2(3) manifold preintegration.
 */

#include "ManifoldPreintegrationSE23.h"

#include <gtsam/geometry/Rot3.h>

#include <cmath>
#include <limits>
#include <type_traits>

using namespace std;

namespace gtsam {

namespace {

// ---------------------------------------------------------------------------
// Small SE_2(3) helpers, inlined here to keep ManifoldPreintegrationSE23
// self-contained inside gtsam (the canonical copies live in
// extensions/PreintegrationHelpers.{hpp,cpp} where they are unit-tested).
// ---------------------------------------------------------------------------

/// Inverse of the left Jacobian of SO(3).
inline Matrix3 so3_J_l_inv(const Vector3& theta) {
  const double theta_ = theta.norm();
  const double theta_sq = theta_ * theta_;
  const double twothetasintheta = 2.0 * theta_ * std::sin(theta_);
  const Matrix3 theta_x = skewSymmetric(theta);
  Matrix3 J_inv = I_3x3 - 0.5 * theta_x;
  if (std::numeric_limits<double>::epsilon() <= theta_sq &&
      std::numeric_limits<double>::epsilon() <= twothetasintheta) {
    const double a =
        (1.0 / theta_sq) - ((1.0 + std::cos(theta_)) / twothetasintheta);
    J_inv += a * theta_x * theta_x;
  }
  return J_inv;
}

/// Left Jacobian of SO(3): J_l = I + ((1-cosφ)/φ²)S + ((φ-sinφ)/φ³)S².
inline Matrix3 so3_J_l(const Vector3& theta) {
  const double phi = theta.norm();
  const Matrix3 S = skewSymmetric(theta);
  if (phi < 1e-8) {
    // Series: I + S/2 + S²/6.
    return I_3x3 + 0.5 * S + (1.0 / 6.0) * S * S;
  }
  const double phi2 = phi * phi;
  const double c1 = (1.0 - std::cos(phi)) / phi2;
  const double c2 = (phi - std::sin(phi)) / (phi2 * phi);
  return I_3x3 + c1 * S + c2 * S * S;
}

/// C_hat(w_hat, dt) = ∫₀ᵈᵗ τ Exp(-w_hat τ) dτ — first-moment integral (ref §2),
/// the position-increment kernel of the constant-body-IMU model.
inline Matrix3 se23_C_hat(const Vector3& w_hat, double dt) {
  const double w = w_hat.norm();
  const Matrix3 Sw = skewSymmetric(w_hat);
  const double dt2 = dt * dt;
  Matrix3 C = 0.5 * dt2 * I_3x3;
  if (w < 1e-8) {
    // Leading small-angle terms: coeff1 → -dt³/3 (·Sw, itself O(w)),
    // coeff2 → dt⁴/8 (·Sw², O(w²)). Both vanish as w→0; series keeps them
    // well-conditioned near zero.
    C += (-dt2 * dt / 3.0) * Sw + (dt2 * dt2 / 8.0) * Sw * Sw;
    return C;
  }
  const double th = w * dt;
  const double w2 = w * w, w3 = w2 * w, w4 = w2 * w2;
  const double c1 = (w * dt * std::cos(th) - std::sin(th)) / w3;
  const double c2 =
      (0.5 * w2 * dt2 - std::cos(th) - w * dt * std::sin(th) + 1.0) / w4;
  C += c1 * Sw + c2 * Sw * Sw;
  return C;
}

/// Constant-body-force position-increment kernel (the MEAN position kernel):
///   M(w,dt) = ∫₀ᵈᵗ ∫₀ᵘ Exp(w τ) dτ du = dt²·J_l(w·dt) − ∫₀ᵈᵗ τ·Exp(w τ) dτ.
/// NOTE: this is NOT se23_C_hat = ∫₀ᵈᵗ τ·Exp(−w τ) dτ, which is the noise-gain
/// (G_j) kernel. Using C_hat as the mean kernel drifts position under rotation.
/// Since ∫τExp(+w τ) equals se23_C_hat evaluated at −w (only the odd Sω term
/// flips), M(w,dt) = dt²·J_l(w dt) − se23_C_hat(−w, dt).
inline Matrix3 se23_pos_kernel(const Vector3& w_hat, double dt) {
  return dt * dt * so3_J_l(w_hat * dt) - se23_C_hat(-w_hat, dt);
}

/// SE_2(3) Ypsilon_hat(w_hat, f_hat, dt) — Eq. 36 Brossard et al.
/// No longer used for state propagation (we compose intrinsically instead to
/// keep R on SO(3)); retained for reference / potential covariance checks.
[[maybe_unused]] inline Matrix5 se23_Ypsilon_hat(const Vector3& w_hat,
                                                 const Vector3& f_hat,
                                                 double dt) {
  const double dt22 = 0.5 * dt * dt;
  Matrix5 T = Matrix5::Identity();
  T.block<3, 3>(0, 0) = Rot3::Expmap(w_hat * dt).matrix();
  T.block<3, 1>(0, 3) = f_hat * dt;
  T.block<3, 1>(0, 4) = f_hat * dt22;
  return T;
}

/// Phi_t(T, dt) = (R, v, p + dt * v) — Eq. 25 Brossard et al. Applied to a
/// 5x5 homogeneous SE_2(3) matrix.
[[maybe_unused]] inline Matrix5 se23_Phi_t(const Matrix5& T, double dt) {
  Matrix5 Tp = T;
  Tp.block<3, 1>(0, 4) = T.block<3, 1>(0, 4) + dt * T.block<3, 1>(0, 3);
  return Tp;
}

/// SE_2(3) state-transition matrix F_dt — Eq. 33 Brossard et al.
inline Matrix9 se23_F_dt(double dt) {
  Matrix9 F = Matrix9::Identity();
  F.block<3, 3>(6, 3) = dt * I_3x3;
  return F;
}

/// SE_2(3) input Jacobian G_j for the SimpleGlobalAcc increment (ref §4b),
/// cols [acc(0-2), gyro(3-5)], rows [theta, nu, rho].
inline Matrix96 se23_G_j_simple(const Vector3& w_hat, const Vector3& /*f_hat*/,
                                double dt) {
  const double dt22 = 0.5 * dt * dt;
  const Matrix3 expminwhatdt = Rot3::Expmap(-w_hat * dt).matrix();
  Matrix96 G = Matrix96::Zero();
  G.block<3, 3>(0, 3) = -so3_J_l_inv(w_hat * dt) * dt;  // theta / gyro
  G.block<3, 3>(3, 0) = -expminwhatdt * dt;             // nu    / acc
  G.block<3, 3>(6, 0) = -expminwhatdt * dt22;           // rho   / acc
  return G;
}

/// One-step ConstantBodyImu increment Yhat = (Exp(w*dt), J_l(w*dt)*f*dt,
/// M(w,dt)*f) — the SAME element update() advances the mean with, so the mean
/// and the gain below stay paired by construction.
inline Se23 se23_full_increment(const Vector3& f_hat,
                                         const Vector3& w_hat, double dt) {
  return Se23(Rot3::Expmap(w_hat * dt),
                       so3_J_l(w_hat * dt) * f_hat * dt,
                       se23_pos_kernel(w_hat, dt) * f_hat);
}

/// SE_2(3) input Jacobian G_j for the ConstantBodyImu increment, cols
/// [acc(0-2), gyro(3-5)], rows [theta, nu, rho]. Computed by central finite
/// differences of the increment tangent d log(Yhat^{-1} Yhat(meas+delta)) — so
/// the (mean, gain) pairing (ref §9, Remark 5) holds exactly, instead of the
/// hand-derived d_v/d_p kernels (whose d_p was tied to the wrong C_hat mean and
/// silently broke when the mean was corrected to M). Bias enters as
/// meas_corrected = meas - bias, so G = -d/d(meas).
inline Matrix96 se23_G_j_full(const Vector3& w_hat, const Vector3& f_hat,
                              double dt) {
  const Se23 Yinv = se23_full_increment(f_hat, w_hat, dt).inverse();
  const double eps = 1e-6, inv2e = 1.0 / (2.0 * eps);
  Matrix96 G;
  for (int j = 0; j < 3; ++j) {
    Vector3 e = Vector3::Zero();
    e(j) = eps;
    const Vector9 dacc =
        Se23::Logmap(
            Yinv.compose(se23_full_increment(f_hat + e, w_hat, dt))) -
        Se23::Logmap(
            Yinv.compose(se23_full_increment(f_hat - e, w_hat, dt)));
    const Vector9 dgyr =
        Se23::Logmap(
            Yinv.compose(se23_full_increment(f_hat, w_hat + e, dt))) -
        Se23::Logmap(
            Yinv.compose(se23_full_increment(f_hat, w_hat - e, dt)));
    G.col(j) = -dacc * inv2e;      // acc column j
    G.col(3 + j) = -dgyr * inv2e;  // gyro column j
  }
  return G;
}

}  // namespace

//------------------------------------------------------------------------------
template <typename Bias>
void ManifoldPreintegrationSE23<Bias>::resetIntegration() {
  deltaTij_ = 0.0;
  deltaXij_ = Se23();
  delRdelBiasOmega_.setZero();
  delNUdelBiasAcc_.setZero();
  delNUdelBiasOmega_.setZero();
  delRHOdelBiasAcc_.setZero();
  delRHOdelBiasOmega_.setZero();
}

//------------------------------------------------------------------------------
template <typename Bias>
bool ManifoldPreintegrationSE23<Bias>::equals(
    const ManifoldPreintegrationSE23& other, double tol) const {
  return p_->equals(*other.p_, tol) &&
         incrementModel_ == other.incrementModel_ &&
         std::abs(deltaTij_ - other.deltaTij_) < tol &&
         biasHat_.equals(other.biasHat_, tol) &&
         deltaXij_.equals(other.deltaXij_, tol) &&
         equal_with_abs_tol(delRdelBiasOmega_, other.delRdelBiasOmega_, tol) &&
         equal_with_abs_tol(delNUdelBiasAcc_, other.delNUdelBiasAcc_, tol) &&
         equal_with_abs_tol(delNUdelBiasOmega_, other.delNUdelBiasOmega_,
                            tol) &&
         equal_with_abs_tol(delRHOdelBiasAcc_, other.delRHOdelBiasAcc_, tol) &&
         equal_with_abs_tol(delRHOdelBiasOmega_, other.delRHOdelBiasOmega_,
                            tol);
}

//------------------------------------------------------------------------------
// update: advance the preintegrated state on SE_2(3) using the Brossard
// left-trivialized propagation
//
//   deltaXij_{k+1}  = deltaXij_k * Ypsilon_hat(w_hat, f_hat, dt)
//   A_k = F_dt(dt)                 (9x9 state-transition, linearized)
//   [B_k | C_k] = G_j(w_hat, f_hat, dt)   (9x3 | 9x3, acc then gyro cols)
//
// The bias Jacobians J_k (9x6, cols [acc, gyro], rows [theta, nu, rho])
// propagate as J_{k+1} = F_dt * J_k + G_j * diag(beta_acc, beta_gyro),
// where beta_* = exp(-t_k / tau_*) for the GaussMarkovBias model and 1 for
// ConstantBias.
//------------------------------------------------------------------------------
template <typename Bias>
void ManifoldPreintegrationSE23<Bias>::update(
    const Vector3& measuredAcc, const Vector3& measuredOmega, const double dt,
    Matrix9* A, Matrix93* B, Matrix93* C, Vector3* correctedAcc,
    Vector3* correctedOmega) {
  // 1. Correct bias to obtain (f_hat, w_hat) using the FROZEN window-start bias
  //    b_i (= biasHat_). We deliberately use the no-dt correct* overloads for
  //    BOTH bias types, i.e. beta = 1: the IMU is debiased with b_i, never with
  //    a mid-window mean-reverted bias. (GM mean-reversion of the *covariance*
  //    and of the between-states residual is handled elsewhere; it must not
  //    enter the in-window correction.)
  //    ALT (disabled): mid-window GM mean reversion, beta = exp(-deltaTij_/tau):
  //      acc   = biasHat_.correctAccelerometer(measuredAcc, deltaTij_);
  //      omega = biasHat_.correctGyroscope(measuredOmega, deltaTij_);
  Vector3 acc = biasHat_.correctAccelerometer(measuredAcc);
  Vector3 omega = biasHat_.correctGyroscope(measuredOmega);

  // 2. Possibly correct for sensor pose.
  Matrix3 D_correctedAcc_acc, D_correctedAcc_omega, D_correctedOmega_omega;
  if (p().body_P_sensor) {
    std::tie(acc, omega) = this->correctMeasurementsBySensorPose(
        acc, omega, D_correctedAcc_acc, D_correctedAcc_omega,
        D_correctedOmega_omega);
  }

  // Expose the corrected (f_hat, w_hat) for the continuous-time covariance path.
  if (correctedAcc) *correctedAcc = acc;
  if (correctedOmega) *correctedOmega = omega;

  // 3. Advance group state on SE_2(3).
  //    Recursion Z_{k+1} = Phi_dt(Z_k) * Ypsilon_hat(w_hat, f_hat, dt) so that
  //    the full predict factors cleanly as
  //      X_j = Gamma_T(g) * Phi_T(X_i) * Z_j.
  //    F_dt (which has dt*I at the rho/nu block) is precisely the tangent-space
  //    counterpart of this recursion.
  //
  //    NOTE: we do NOT compose via a generic 5x5 matrix product + reslice —
  //    doing so lets the rotation block drift off SO(3) under accumulation,
  //    and Se23(Matrix5) slices the 3x3 without re-orthogonalizing.
  //    After ~tens of seconds the non-orthogonality is large enough to flip
  //    the quaternion conversion branch and cause a visible attitude jump.
  //    Instead compose intrinsically: Rot3::Expmap keeps R on SO(3) exactly,
  //    and v/p get the algebraically equivalent updates that Phi_t * Ypsilon
  //    yields for an exact SO(3) rotation block.
  const Rot3 R_i = deltaXij_.rotation();
  const Vector3 v_i = deltaXij_.x(0);
  const Vector3 p_i = deltaXij_.x(1);
  const Rot3 dR = Rot3::Expmap(omega * dt);

  // Body-frame velocity / position increments of the single-step element
  // Ypsilon_hat, chosen by the increment model. Ypsilon = (dR, dv_b, dp_b).
  Vector3 dv_b, dp_b;
  switch (incrementModel_) {
    case SE23IncrementModel::SimpleGlobalAcc:  // constant global acc
      dv_b = acc * dt;
      dp_b = acc * (0.5 * dt * dt);
      break;
    case SE23IncrementModel::ConstantBodyImu:  // constant body specific force
      dv_b = so3_J_l(omega * dt) * acc * dt;
      dp_b = se23_pos_kernel(omega, dt) * acc;
      break;
  }
  const Se23 Yhat(dR, dv_b, dp_b);

  // Advance the mean state intrinsically (keeps R on SO(3) exactly). This is
  // algebraically deltaXij_ = Phi_dt(deltaXij_) * Yhat.
  const Rot3 R_new = R_i * dR;
  const Vector3 v_new = v_i + R_i * dv_b;
  const Vector3 p_new = p_i + v_i * dt + R_i * dp_b;
  deltaXij_ = Se23(R_new, v_new, p_new);

  // 4. Linearized transition + measurement Jacobians.
  //    The FULL one-step transition is A = Ad(Yhat^{-1}) * F_dt (ref §3b),
  //    NOT F_dt alone: its top-left block is Exp(-w_hat*dt), so rotation error
  //    correctly rotates the velocity/position covariance blocks.
  const Matrix9 F = Yhat.inverse().AdjointMap() * se23_F_dt(dt);
  const Matrix96 G = (incrementModel_ == SE23IncrementModel::ConstantBodyImu)
                         ? se23_G_j_full(omega, acc, dt)
                         : se23_G_j_simple(omega, acc, dt);  // cols acc|gyro

  *A = F;
  *B = G.block<9, 3>(0, 0);  // d(xi_inc)/d(acc)
  *C = G.block<9, 3>(0, 3);  // d(xi_inc)/d(omega)

  if (p().body_P_sensor) {
    *C *= D_correctedOmega_omega;
    if (!p().body_P_sensor->translation().isZero()) {
      *C += *B * D_correctedAcc_omega;
    }
    *B *= D_correctedAcc_acc;  // must be last
  }

  // 5. Propagate the bias Jacobian J = d(preint delta)/d(b_i) via the full
  //    transition, at beta = 1 (frozen b_i, ref §7a):  J_{k+1} = A J_k + G.
  //    Assembled/written back through the 5 stored 3x3 blocks (rows theta, nu,
  //    rho ; cols acc, gyro). The theta/acc block is provably zero and stays 0.
  Matrix96 Jbias = Matrix96::Zero();
  Jbias.block<3, 3>(0, 3) = delRdelBiasOmega_;
  Jbias.block<3, 3>(3, 0) = delNUdelBiasAcc_;
  Jbias.block<3, 3>(3, 3) = delNUdelBiasOmega_;
  Jbias.block<3, 3>(6, 0) = delRHOdelBiasAcc_;
  Jbias.block<3, 3>(6, 3) = delRHOdelBiasOmega_;

  Jbias = (*A) * Jbias + G;  // G's sign already encodes d/d(bias) at beta=1

  delRdelBiasOmega_ = Jbias.block<3, 3>(0, 3);
  delNUdelBiasAcc_ = Jbias.block<3, 3>(3, 0);
  delNUdelBiasOmega_ = Jbias.block<3, 3>(3, 3);
  delRHOdelBiasAcc_ = Jbias.block<3, 3>(6, 0);
  delRHOdelBiasOmega_ = Jbias.block<3, 3>(6, 3);

  // 6. Bookkeeping.
  deltaTij_ += dt;
}

//------------------------------------------------------------------------------
// biasCorrectedDelta
//   xi = Log(deltaXij_) + J_bias * (bias_i - biasHat_)
// where J_bias is assembled from the tracked 3x3 blocks with row order
// [theta, nu, rho] and col order [acc, gyro].
//------------------------------------------------------------------------------
template <typename Bias>
Vector9 ManifoldPreintegrationSE23<Bias>::biasCorrectedDelta(
    const Bias& bias_i, OptionalJacobian<9, 6> H) const {
  const Bias biasIncr = bias_i - biasHat_;
  Vector6 db;
  db << biasIncr.accelerometer(), biasIncr.gyroscope();  // cols [acc, gyro]

  // Right-trivialized bias-perturbation Jacobian J_bias (rows [theta, nu, rho],
  // cols [acc, gyro]) as accumulated by update().
  Matrix96 J = Matrix96::Zero();
  J.block<3, 3>(0, 3) = delRdelBiasOmega_;
  J.block<3, 3>(3, 0) = delNUdelBiasAcc_;
  J.block<3, 3>(3, 3) = delNUdelBiasOmega_;
  J.block<3, 3>(6, 0) = delRHOdelBiasAcc_;
  J.block<3, 3>(6, 3) = delRHOdelBiasOmega_;

  // The recursion accumulates the RIGHT-trivialized perturbation, i.e.
  //   deltaXij(b) ~= deltaXij_ * Exp(J * db).
  // So the corrected tangent is Log of that product; the Logmap contributes the
  // J_r^{-1} factor that maps the group perturbation into Log-coordinates
  // (adding J*db directly to Log(deltaXij_) would drop it).
  const Vector9 tanCorr = J * db;
  Matrix9 D_exp_tan, D_comp_corr, D_log;
  const Se23 Corr =
      Se23::Expmap(tanCorr, H ? &D_exp_tan : nullptr);
  const Se23 corrected =
      deltaXij_.compose(Corr, {}, H ? &D_comp_corr : nullptr);
  const Vector9 xi = Se23::Logmap(corrected, H ? &D_log : nullptr);

  if (H) *H = D_log * D_comp_corr * D_exp_tan * J;
  return xi;
}

//------------------------------------------------------------------------------

}  // namespace gtsam

// Explicit instantiations.
template class gtsam::ManifoldPreintegrationSE23<gtsam::imuBias::ConstantBias>;
template class gtsam::ManifoldPreintegrationSE23<
    gtsam::imuBias::GaussMarkovBias>;
