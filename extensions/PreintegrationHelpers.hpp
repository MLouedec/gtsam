#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Rot3.h>

/**
 * @file PreintegrationHelpers.hpp
 * @brief SO(3) and SE_2(3) Lie-group helpers used by preintegration.
 *
 * Organised so that helpers are grouped by the group they belong to:
 *   - parnav::helpers::SO3::*    — Lie group helpers for SO(3)
 *   - parnav::helpers::SE3::*    — Lie group helpers for SE(3) (placeholder)
 *   - parnav::helpers::SE23::*   — Lie group helpers for SE_2(3)
 *   - parnav::helpers::*         — shared utilities (vee, R projection, Q)
 *
 * SE_2(3) elements are represented as homogeneous 5x5 matrices:
 *
 *     T = [ R  v  p ]
 *         [ 0  1  0 ]
 *         [ 0  0  1 ]
 *
 * with tangent ordering xi = [theta(3), nu(3), rho(3)] (rotation, velocity,
 * position) per Brossard et al.
 *
 * References:
 *   - Solà, "A micro Lie theory for state estimation in robotics"
 *   - Brossard et al., "AI-IMU Dead-Reckoning"
 *   - Barfoot, "State Estimation for Robotics"
 */

namespace parnav::helpers {

// Shared utilities ----------------------------------------------------------

/// Project a (possibly non-orthogonal) 3x3 matrix onto SO(3) using Grip et al.
auto force_R_to_SO3(const gtsam::Matrix& R, double mu = 0.5) -> gtsam::Rot3;

/// Inverse of skewSymmetric: extract the 3-vector from a skew-symmetric 3x3.
auto vee_3(const gtsam::Matrix3& theta_x) -> gtsam::Vector3;

/// Q(theta, rho) — coupling block used in SE(3) / SE_2(3) Jacobians.
/// Definition from Barfoot / Solà.
auto Q_theta_rho(const gtsam::Vector3& theta, const gtsam::Vector3& rho)
    -> gtsam::Matrix3;

// SO(3) ---------------------------------------------------------------------

namespace SO3 {

/// Left Jacobian of SO(3).
auto J_l(const gtsam::Vector3& theta) -> gtsam::Matrix3;

/// Inverse of the left Jacobian of SO(3).
auto J_l_inv(const gtsam::Vector3& theta) -> gtsam::Matrix3;

/// Right Jacobian of SO(3).
auto J_r(const gtsam::Vector3& theta) -> gtsam::Matrix3;

/// Inverse of the right Jacobian of SO(3).
auto J_r_inv(const gtsam::Vector3& theta) -> gtsam::Matrix3;

/// SO(3) exponential map theta -> R, with re-orthogonalisation.
auto Expmap(const gtsam::Vector3& theta) -> gtsam::Rot3;

/// SO(3) logarithm map R -> theta.
auto Logmap(const gtsam::Rot3& R) -> gtsam::Vector3;

}  // namespace SO3

// SE(3) ---------------------------------------------------------------------

namespace SE3 {
// Placeholder. Add SE(3)-specific helpers here when needed.
// Q_theta_rho() in the parent namespace is the standard SE(3) coupling term.
}  // namespace SE3

// SE_2(3) -------------------------------------------------------------------

namespace SE23 {

/// Left Jacobian of SE_2(3) for tangent xi = [theta, nu, rho].
auto J_l(const gtsam::Vector9& xi) -> gtsam::Matrix9;

/// Inverse of the left Jacobian of SE_2(3).
auto J_l_inv(const gtsam::Vector9& xi) -> gtsam::Matrix9;

/// Right Jacobian of SE_2(3).
auto J_r(const gtsam::Vector9& xi) -> gtsam::Matrix9;

/// Inverse of the right Jacobian of SE_2(3).
auto J_r_inv(const gtsam::Vector9& xi) -> gtsam::Matrix9;

/// SE_2(3) exponential map xi -> T (5x5 homogeneous matrix).
auto Expmap(const gtsam::Vector9& xi) -> gtsam::Matrix5;

/// SE_2(3) logarithm map T (5x5 homogeneous) -> xi.
auto Logmap(const gtsam::Matrix5& T) -> gtsam::Vector9;

/// Phi_t — Eq. 25 in Brossard et al.
auto Phi_t(const gtsam::Matrix5& T, double dt) -> gtsam::Matrix5;

/// Gamma_t(g, dt) — Eq. 26 in Brossard et al.
auto Gamma_t(const gtsam::Vector3& g, double dt) -> gtsam::Matrix5;

/// State transition matrix for tangent-space propagation — Eq. 33 in Brossard.
auto F_dt(double dt) -> gtsam::Matrix9;

/// Ypsilon_hat(w_hat, f_hat, dt) — Eq. 36 in Brossard et al.
auto Ypsilon_hat(const gtsam::Vector3& w_hat, const gtsam::Vector3& f_hat,
                 double dt) -> gtsam::Matrix5;

/// Input Jacobian G_j(w_hat, f_hat, dt) — Eq. 38 in Brossard.
auto G_j(const gtsam::Vector3& w_hat, const gtsam::Vector3& f_hat, double dt)
    -> gtsam::Matrix96;

/// Build a 5x5 SE_2(3) homogeneous matrix from (R, v, p).
inline auto Make(const gtsam::Rot3& R, const gtsam::Vector3& v,
                 const gtsam::Vector3& p) -> gtsam::Matrix5 {
  gtsam::Matrix5 T = gtsam::Matrix5::Identity();
  T.block<3, 3>(0, 0) = R.matrix();
  T.block<3, 1>(0, 3) = v;
  T.block<3, 1>(0, 4) = p;
  return T;
}

inline auto Rotation(const gtsam::Matrix5& T) -> gtsam::Rot3 {
  return gtsam::Rot3{gtsam::Matrix3{T.block<3, 3>(0, 0)}};
}
inline auto Velocity(const gtsam::Matrix5& T) -> gtsam::Vector3 {
  return T.block<3, 1>(0, 3);
}
inline auto Position(const gtsam::Matrix5& T) -> gtsam::Vector3 {
  return T.block<3, 1>(0, 4);
}

}  // namespace SE23

}  // namespace parnav::helpers
