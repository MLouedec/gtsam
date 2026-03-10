/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    SE23CovariancePropagation.cpp
 * @brief   Continuous F_c / Q~ assembly and discrete Q_d (Ours, VanLoan).
 */

#include <gtsam/navigation/SE23CovariancePropagation.h>
#include <gtsam/base/MatrixConstants.h>

#include <unsupported/Eigen/MatrixFunctions>  // Pade matrix exponential (VanLoan)

namespace gtsam {

//------------------------------------------------------------------------------
SE23Covariance se23ContinuousFc(const Vector3& w_hat, const Vector3& f_hat,
                          double biasFcAcc, double biasFcGyro) {
  const Matrix3 Sw = skewSymmetric(w_hat);
  const Matrix3 Sf = skewSymmetric(f_hat);

  SE23Covariance Fc = SE23Covariance::Zero();
  // F_SE23 (rows/cols theta, nu, rho).
  Fc.block<3, 3>(0, 0) = -Sw;      // theta <- theta
  Fc.block<3, 3>(3, 0) = -Sf;      // nu    <- theta
  Fc.block<3, 3>(3, 3) = -Sw;      // nu    <- nu
  Fc.block<3, 3>(6, 3) = I_3x3;    // rho   <- nu
  Fc.block<3, 3>(6, 6) = -Sw;      // rho   <- rho

  // Bias-error -> pose coupling G_SE23 (acc-first: b_acc at 9, b_gyro at 12).
  Fc.block<3, 3>(0, 12) = -I_3x3;  // theta <- b_gyro
  Fc.block<3, 3>(3, 9) = -I_3x3;   // nu    <- b_acc

  // Bias self-dynamics (GM: -1/tau, Wiener: 0).
  Fc.block<3, 3>(9, 9) = biasFcAcc * I_3x3;
  Fc.block<3, 3>(12, 12) = biasFcGyro * I_3x3;
  return Fc;
}

//------------------------------------------------------------------------------
SE23Covariance se23ContinuousQtilde(const Matrix3& accCov, const Matrix3& gyroCov,
                              const Matrix3& intCov, const Matrix3& biasAccCov,
                              const Matrix3& biasGyroCov) {
  SE23Covariance Q = SE23Covariance::Zero();
  Q.block<3, 3>(0, 0) = gyroCov;        // theta driven by gyro white noise
  Q.block<3, 3>(3, 3) = accCov;         // nu    driven by acc white noise
  Q.block<3, 3>(6, 6) = intCov;         // rho   integration random walk
  Q.block<3, 3>(9, 9) = biasAccCov;     // b_acc bias evolution
  Q.block<3, 3>(12, 12) = biasGyroCov;  // b_gyro bias evolution
  return Q;
}

//------------------------------------------------------------------------------
SE23Covariance se23DiscreteQd_Ours(const SE23Covariance& Fc, const SE23Covariance& Q, double dt) {
  // Q_d = sum_{n=1..4} (dt^n/n!) sum_{j=0}^{n-1} C(n-1,j) Fc^j Q (Fc^T)^{n-1-j}.
  // Using Q = Q^T so that (Fc^a Q Fc^Tb)^T = Fc^b Q Fc^Ta.
  const SE23Covariance FQ = Fc * Q;              // Fc Q
  const SE23Covariance F2Q = Fc * FQ;            // Fc^2 Q
  const SE23Covariance F3Q = Fc * F2Q;           // Fc^3 Q
  const SE23Covariance FQFt = FQ * Fc.transpose();    // Fc Q Fc^T (symmetric)
  const SE23Covariance F2QFt = F2Q * Fc.transpose();  // Fc^2 Q Fc^T

  const double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt;
  SE23Covariance Qd = dt * Q;
  Qd += (dt2 / 2.0) * (FQ + FQ.transpose());
  Qd += (dt3 / 6.0) * (F2Q + F2Q.transpose() + 2.0 * FQFt);
  Qd += (dt4 / 24.0) *
        (F3Q + F3Q.transpose() + 3.0 * (F2QFt + F2QFt.transpose()));
  return Qd;
}

//------------------------------------------------------------------------------
SE23Covariance se23DiscreteQd_VanLoan(const SE23Covariance& Fc, const SE23Covariance& Q,
                                double dt) {
  // Van Loan (1978): M = [[-Fc, Q],[0, Fc^T]] * dt, then Q_d = E22^T E12 with
  // E = expm(M) (ref §6a). E22^T is the discrete transition Phi_d.
  Eigen::Matrix<double, 30, 30> M = Eigen::Matrix<double, 30, 30>::Zero();
  M.block<15, 15>(0, 0) = -Fc * dt;
  M.block<15, 15>(0, 15) = Q * dt;
  M.block<15, 15>(15, 15) = Fc.transpose() * dt;

  const Eigen::Matrix<double, 30, 30> E = M.exp();
  const SE23Covariance E12 = E.block<15, 15>(0, 15);
  const SE23Covariance E22 = E.block<15, 15>(15, 15);
  SE23Covariance Qd = E22.transpose() * E12;
  return 0.5 * (Qd + Qd.transpose());  // symmetrize away round-off
}

}  // namespace gtsam
