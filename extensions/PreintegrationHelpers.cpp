#include "PreintegrationHelpers.hpp"

#include <gtsam/base/Matrix.h>

#include <cmath>
#include <limits>
#include <tuple>

namespace parnav::helpers {

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

auto force_R_to_SO3(const gtsam::Matrix& R, double mu) -> gtsam::Rot3 {
  // From Grip et al.
  const gtsam::Vector3& r1 = R.block<3, 1>(0, 0);
  const gtsam::Vector3& r2 = R.block<3, 1>(0, 1);

  const gtsam::Vector3& r1bar = r1 / std::max(r1.norm(), mu);
  const gtsam::Vector3& r2bar =
      ((gtsam::I_3x3 - r1bar * r1bar.transpose()) * r2) /
      std::max(((gtsam::I_3x3 - r1bar * r1bar.transpose()) * r2).norm(), mu);

  const auto& S_r1 = gtsam::skewSymmetric(r1bar);

  gtsam::Matrix3 Rm;
  Rm << r1bar, r2bar, S_r1 * r2bar;

  return gtsam::Rot3{Rm};
}

auto vee_3(const gtsam::Matrix3& theta_x) -> gtsam::Vector3 {
  return (gtsam::Vector(3) << theta_x.coeffRef(2, 1), theta_x.coeffRef(0, 2),
          theta_x.coeffRef(1, 0))
      .finished();
}

auto Q_theta_rho(const gtsam::Vector3& theta, const gtsam::Vector3& rho)
    -> gtsam::Matrix3 {
  // Definition from Barfoot and Solà.
  const double theta_ = theta.norm();
  const double theta_sq = theta_ * theta_;
  const double theta_cu = theta_sq * theta_;
  const double theta_4 = theta_cu * theta_;
  const double theta_5 = theta_4 * theta_;

  const gtsam::Matrix3& theta_x = gtsam::skewSymmetric(theta);
  const gtsam::Matrix3& rho_x = gtsam::skewSymmetric(rho);

  gtsam::Matrix3 Q = 0.5 * rho_x;

  if (std::numeric_limits<double>::epsilon() > theta_cu) {
    return Q;
  }

  const gtsam::Matrix3& second_order_term =
      theta_x * rho_x + rho_x * theta_x + theta_x * rho_x * theta_x;
  Q += ((theta_ - std::sin(theta_)) / theta_cu) * second_order_term;

  if (std::numeric_limits<double>::epsilon() > theta_4) {
    return Q;
  }

  const double a = ((1 - 0.5 * theta_sq - std::cos(theta_)) / theta_4);
  const gtsam::Matrix3& third_order_term = theta_x * theta_x * rho_x +
                                            rho_x * theta_x * theta_x -
                                            3 * theta_x * rho_x * theta_x;
  Q -= a * third_order_term;

  if (std::numeric_limits<double>::epsilon() > theta_5) {
    return Q;
  }

  const double b = 0.5 * (a - 3 * ((theta_ - std::sin(theta_) - (theta_cu / 6)) /
                                    theta_5));
  const gtsam::Matrix3& fourth_order_term =
      theta_x * rho_x * theta_x * theta_x + theta_x * theta_x * rho_x * theta_x;
  Q -= b * fourth_order_term;

  return Q;
}

// ---------------------------------------------------------------------------
// SO(3)
// ---------------------------------------------------------------------------

namespace SO3 {

auto J_l(const gtsam::Vector3& theta) -> gtsam::Matrix3 {
  const double theta_ = theta.norm();
  const double theta_sq = theta_ * theta_;
  const double theta_cu = theta_sq * theta_;

  const auto& theta_x = gtsam::skewSymmetric(theta);
  const auto& theta_x_sq = theta_x * theta_x;

  gtsam::Matrix3 J = gtsam::I_3x3;

  if (std::numeric_limits<double>::epsilon() <= theta_sq) {
    const double a = (1 - std::cos(theta_)) / theta_sq;
    J += a * theta_x;
  }

  if (std::numeric_limits<double>::epsilon() <= theta_cu) {
    const double b = (theta_ - std::sin(theta_)) / theta_cu;
    J += b * theta_x_sq;
  }

  return J;
}

auto J_l_inv(const gtsam::Vector3& theta) -> gtsam::Matrix3 {
  const double theta_ = theta.norm();
  const double theta_sq = theta_ * theta_;
  const double twothetasintheta = 2 * theta_ * std::sin(theta_);

  const auto& theta_x = gtsam::skewSymmetric(theta);
  const auto& theta_x_sq = theta_x * theta_x;

  gtsam::Matrix3 J_inv = gtsam::I_3x3 - 0.5 * theta_x;

  if (std::numeric_limits<double>::epsilon() <= theta_sq &&
      std::numeric_limits<double>::epsilon() <= twothetasintheta) {
    const double a =
        (1 / theta_sq) - ((1 + std::cos(theta_)) / twothetasintheta);
    J_inv += a * theta_x_sq;
  }

  return J_inv;
}

auto J_r(const gtsam::Vector3& theta) -> gtsam::Matrix3 {
  gtsam::Matrix3 J = gtsam::Matrix3::Identity();

  const double theta_ = theta.norm();
  const double theta_sq = theta_ * theta_;
  const double theta_cu = theta_sq * theta_;

  const auto& theta_x = gtsam::skewSymmetric(theta);
  const auto& theta_x_sq = theta_x * theta_x;

  if (std::numeric_limits<double>::epsilon() <= theta_sq) {
    const double a = (1 - std::cos(theta_)) / theta_sq;
    J -= a * theta_x;
  }

  if (std::numeric_limits<double>::epsilon() <= theta_cu) {
    const double b = (theta_ - std::sin(theta_)) / theta_cu;
    J += b * theta_x_sq;
  }

  return J;
}

auto J_r_inv(const gtsam::Vector3& theta) -> gtsam::Matrix3 {
  const double theta_ = theta.norm();
  const double theta_sq = theta_ * theta_;
  const double thetasintheta2 = 2 * theta_ * std::sin(theta_);

  const auto& theta_x = gtsam::skewSymmetric(theta);
  const auto& theta_x_sq = theta_x * theta_x;

  gtsam::Matrix3 J_inv = gtsam::I_3x3 + 0.5 * theta_x;

  if (std::numeric_limits<double>::epsilon() <= theta_sq &&
      std::numeric_limits<double>::epsilon() <= thetasintheta2) {
    const double a =
        (1 / theta_sq) - ((1 + std::cos(theta_)) / thetasintheta2);
    J_inv += a * theta_x_sq;
  }

  return J_inv;
}

auto Expmap(const gtsam::Vector3& theta) -> gtsam::Rot3 {
  const double theta_ = theta.norm();

  const gtsam::Vector3& u = theta.normalized();
  const gtsam::Matrix3& u_x = gtsam::skewSymmetric(u);

  gtsam::Matrix3 Exp_theta =
      gtsam::I_3x3 + u_x * std::sin(theta_) + u_x * u_x * (1 - std::cos(theta_));

  return force_R_to_SO3(Exp_theta);
}

auto Logmap(const gtsam::Rot3& R) -> gtsam::Vector3 {
  const gtsam::Matrix3& Rm = R.matrix();
  const double mu = std::acos(0.5 * (Rm.trace() - 1));
  const gtsam::Vector3& vec = vee_3(Rm - Rm.transpose());

  return mu > 1e-8 ? (0.5 * (mu / (std::sin(mu))) * vec) : 0.5 * vec;
}

}  // namespace SO3

// ---------------------------------------------------------------------------
// SE_2(3)
// ---------------------------------------------------------------------------

namespace SE23 {

auto J_l(const gtsam::Vector9& xi) -> gtsam::Matrix9 {
  const gtsam::Vector3& theta = xi.head<3>();
  const gtsam::Vector3& nu = xi.middleRows<3>(3);
  const gtsam::Vector3& rho = xi.tail<3>();

  const gtsam::Matrix3& J_theta = SO3::J_l(theta);
  const gtsam::Matrix3& Q_pn = Q_theta_rho(theta, nu);
  const gtsam::Matrix3& Q_pr = Q_theta_rho(theta, rho);

  gtsam::Matrix9 J;
  J << J_theta, gtsam::Z_3x3, gtsam::Z_3x3, Q_pn, J_theta, gtsam::Z_3x3, Q_pr,
      gtsam::Z_3x3, J_theta;

  return J;
}

auto J_l_inv(const gtsam::Vector9& xi) -> gtsam::Matrix9 {
  const gtsam::Vector3& theta = xi.head<3>();
  const gtsam::Vector3& nu = xi.middleRows<3>(3);
  const gtsam::Vector3& rho = xi.tail<3>();

  const gtsam::Matrix3& J_inv_theta = SO3::J_l_inv(theta);
  const gtsam::Matrix3& Q_pn = Q_theta_rho(theta, nu);
  const gtsam::Matrix3& Q_pr = Q_theta_rho(theta, rho);

  gtsam::Matrix9 J_inv;
  J_inv << J_inv_theta, gtsam::Z_3x3, gtsam::Z_3x3,
      -J_inv_theta * Q_pn * J_inv_theta, J_inv_theta, gtsam::Z_3x3,
      -J_inv_theta * Q_pr * J_inv_theta, gtsam::Z_3x3, J_inv_theta;
  return J_inv;
}

auto J_r(const gtsam::Vector9& xi) -> gtsam::Matrix9 { return J_l(-xi); }

auto J_r_inv(const gtsam::Vector9& xi) -> gtsam::Matrix9 {
  return J_l_inv(-xi);
}

auto Expmap(const gtsam::Vector9& xi) -> gtsam::Matrix5 {
  const gtsam::Vector3& theta = xi.head<3>();
  const gtsam::Vector3& nu = xi.middleRows<3>(3);
  const gtsam::Vector3& rho = xi.tail<3>();
  const gtsam::Matrix3& J_l_theta = SO3::J_l(theta);

  gtsam::Matrix5 M = gtsam::Matrix5::Identity();
  M.block<3, 3>(0, 0) = SO3::Expmap(theta).matrix();
  M.block<3, 1>(0, 3) = J_l_theta * nu;
  M.block<3, 1>(0, 4) = J_l_theta * rho;

  return M;
}

auto Logmap(const gtsam::Matrix5& T) -> gtsam::Vector9 {
  const gtsam::Vector3& logR = SO3::Logmap(Rotation(T));
  const gtsam::Vector3& nu = Velocity(T);
  const gtsam::Vector3& rho = Position(T);
  const gtsam::Matrix3& J_logR_inv = SO3::J_l_inv(logR);

  return (gtsam::Vector(9) << logR, J_logR_inv * nu, J_logR_inv * rho)
      .finished();
}

auto Phi_t(const gtsam::Matrix5& T, double dt) -> gtsam::Matrix5 {
  // Eq. 25 Brossard et al.
  return Make(Rotation(T), Velocity(T), Position(T) + dt * Velocity(T));
}

auto Gamma_t(const gtsam::Vector3& g, double dt) -> gtsam::Matrix5 {
  // Eq. 26 Brossard et al.
  const double dt22 = 0.5 * dt * dt;
  return Make(gtsam::Rot3::Identity(), g * dt, g * dt22);
}

auto F_dt(double dt) -> gtsam::Matrix9 {
  // Eq. 33 Brossard et al.
  gtsam::Matrix9 F = gtsam::Matrix9::Identity();
  F.block<3, 3>(6, 3) = dt * gtsam::I_3x3;
  return F;
}

auto Ypsilon_hat(const gtsam::Vector3& w_hat, const gtsam::Vector3& f_hat,
                 double dt) -> gtsam::Matrix5 {
  // Eq. 36 Brossard et al.
  const double dt22 = 0.5 * dt * dt;
  return Make(SO3::Expmap(w_hat * dt), f_hat * dt, f_hat * dt22);
}

auto G_j(const gtsam::Vector3& w_hat, const gtsam::Vector3& f_hat, double dt)
    -> gtsam::Matrix96 {
  std::ignore = f_hat;
  // Eq. 38 Brossard.
  const double dt22 = 0.5 * dt * dt;
  const gtsam::Matrix3& expminwhatdt = SO3::Expmap(-w_hat * dt).matrix();

  const gtsam::Matrix3& delRdelw = -SO3::J_l_inv(w_hat * dt) * dt;
  const gtsam::Matrix3& delvdelf = -expminwhatdt * dt;
  const gtsam::Matrix3& delpdelf = -expminwhatdt * dt22;

  gtsam::Matrix96 G = gtsam::Matrix96::Zero();
  G.block<3, 3>(0, 3) = delRdelw;
  G.block<3, 3>(3, 0) = delvdelf;
  G.block<3, 3>(6, 0) = delpdelf;

  return G;
}

}  // namespace SE23

}  // namespace parnav::helpers
