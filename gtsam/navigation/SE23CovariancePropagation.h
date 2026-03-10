/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    SE23CovariancePropagation.h
 * @brief   Continuous-time error dynamics and discrete process-noise Q_d for
 *          SE_2(3) IMU preintegration.
 *
 * Everything here is in the code's 15-dim acc-first block order
 *   [theta(3), nu(3), rho(3), b_acc(3), b_gyro(3)].
 * (The discrete-time reference derivation uses gyro-first for the bias; the
 * builders below translate into this acc-first layout — keep that in mind when
 * cross-checking against the reference.)
 *
 * Three covariance methods, in increasing cost / accuracy:
 *   - Brossard : discrete gain reconstruction G (Qc/dt) G^T (handled inline in
 *                CombinedImuFactor2; 2nd order, ref §6c).
 *   - Ours     : truncated 4th-order Lyapunov/Taylor series in (F_c, Q~), no
 *                matrix exponential (ref §6b).
 *   - VanLoan  : the exact Van Loan matrix-exponential integral (ref §6a); the
 *                ONLY method that forms a full matrix exponential.
 */

#pragma once

#include <gtsam/base/Matrix.h>

namespace gtsam {

/// 15x15 / 30x30 helpers (not covered by GTSAM's fixed-size matrix macros).
using SE23Covariance = Eigen::Matrix<double, 15, 15>;

/// Discrete process-noise covariance method for SE_2(3) preintegration.
enum class SE23CovarianceMethod { Brossard, Ours, VanLoan };

/**
 * Continuous-time 15x15 error-dynamics matrix F_c (ref §1), acc-first order.
 *   F_c = [ F_SE23   G_SE23 ]      F_SE23 = [ -Sw   0    0 ]   G_SE23 couples
 *         [   0      F_bias ]               [ -Sf  -Sw   0 ]   bias error into
 *                                           [  0    I   -Sw]   the pose block.
 * @param biasFcAcc,biasFcGyro  continuous bias self-rate per axis:
 *        -1/tau (Gauss-Markov) or 0 (Wiener / random walk).
 */
SE23Covariance se23ContinuousFc(const Vector3& w_hat, const Vector3& f_hat,
                          double biasFcAcc, double biasFcGyro);

/**
 * Continuous process-noise density Q~ = G_c Q_c G_c^T (acc-first). Diagonal
 * blocks: theta<-gyroCov, nu<-accCov, rho<-intCov, b_acc<-biasAccCov,
 * b_gyro<-biasGyroCov. All arguments are the continuous-time covariances used
 * by PreintegrationCombinedParams.
 */
SE23Covariance se23ContinuousQtilde(const Matrix3& accCov, const Matrix3& gyroCov,
                              const Matrix3& intCov, const Matrix3& biasAccCov,
                              const Matrix3& biasGyroCov);

/// "Ours": Q_d via the truncated 4th-order Lyapunov series (ref §6b). O(dt^5)
/// error, no matrix exponential.
SE23Covariance se23DiscreteQd_Ours(const SE23Covariance& Fc, const SE23Covariance& Qtilde,
                             double dt);

/// "VanLoan": exact Q_d = E22^T E12 from expm([[-Fc, Q~],[0, Fc^T]]*dt) (ref
/// §6a). Uses a full (Pade) matrix exponential.
SE23Covariance se23DiscreteQd_VanLoan(const SE23Covariance& Fc, const SE23Covariance& Qtilde,
                                double dt);

}  // namespace gtsam
