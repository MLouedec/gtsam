/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    testSE23CovariancePropagation.cpp
 * @brief   Continuous F_c / Q~ assembly and Q_d methods (Ours vs Van Loan).
 */

#include <gtsam/navigation/SE23CovariancePropagation.h>

#include <CppUnitLite/TestHarness.h>

#include <unsupported/Eigen/MatrixFunctions>  // reference expm

using namespace gtsam;

namespace {
const Vector3 kW(0.03, -0.05, 0.09);   // w_hat
const Vector3 kF(0.2, 9.7, -0.3);      // f_hat (specific force)
const Matrix3 kACov = 0.004 * 0.004 * I_3x3;
const Matrix3 kWCov = 0.0007 * 0.0007 * I_3x3;
const Matrix3 kICov = 1e-8 * I_3x3;
const Matrix3 kBA = 1e-4 * I_3x3;
const Matrix3 kBG = 1e-5 * I_3x3;
}  // namespace

/* ************************************************************************* */
// F_c top-left 9x9 must match ref §1 F_SE23 (assertion #1/#2 building block):
// blocks -Sw, -Sf, I, -Sw, and expm(F_c dt) top-left 9x9 == expm(F_SE23 dt).
TEST(SE23CovariancePropagation, ContinuousFcStructure) {
  const SE23Covariance Fc = se23ContinuousFc(kW, kF, -1.0 / 100.0, -1.0 / 200.0);
  const Matrix3 Sw = skewSymmetric(kW), Sf = skewSymmetric(kF);

  EXPECT(assert_equal(Matrix(-Sw), Matrix(Fc.block<3, 3>(0, 0)), 1e-12));
  EXPECT(assert_equal(Matrix(-Sf), Matrix(Fc.block<3, 3>(3, 0)), 1e-12));
  EXPECT(assert_equal(Matrix(-Sw), Matrix(Fc.block<3, 3>(3, 3)), 1e-12));
  EXPECT(assert_equal(Matrix(I_3x3), Matrix(Fc.block<3, 3>(6, 3)), 1e-12));
  EXPECT(assert_equal(Matrix(-Sw), Matrix(Fc.block<3, 3>(6, 6)), 1e-12));
  // bias->pose coupling
  EXPECT(assert_equal(Matrix(-I_3x3), Matrix(Fc.block<3, 3>(0, 12)), 1e-12));
  EXPECT(assert_equal(Matrix(-I_3x3), Matrix(Fc.block<3, 3>(3, 9)), 1e-12));
  // GM bias self-block
  EXPECT(assert_equal(Matrix(-0.01 * I_3x3), Matrix(Fc.block<3, 3>(9, 9)), 1e-12));

  // expm(F_c dt) top-left 9x9 == expm(F_SE23 dt).
  const double dt = 0.01;
  Matrix9 Fse23 = Matrix9::Zero();
  Fse23.block<3, 3>(0, 0) = -Sw;
  Fse23.block<3, 3>(3, 0) = -Sf;
  Fse23.block<3, 3>(3, 3) = -Sw;
  Fse23.block<3, 3>(6, 3) = I_3x3;
  Fse23.block<3, 3>(6, 6) = -Sw;
  const Matrix9 expTopLeft =
      (SE23Covariance(Fc * dt)).exp().block<9, 9>(0, 0);
  EXPECT(assert_equal(Matrix((Matrix9)(Fse23 * dt).exp()), Matrix(expTopLeft),
                      1e-10));
}

/* ************************************************************************* */
// Van Loan Q_d must be symmetric SPD with leading term dt * Q~.
TEST(SE23CovariancePropagation, VanLoanBasics) {
  const SE23Covariance Fc = se23ContinuousFc(kW, kF, -1.0 / 100.0, -1.0 / 200.0);
  const SE23Covariance Qt = se23ContinuousQtilde(kACov, kWCov, kICov, kBA, kBG);
  const double dt = 0.01;
  const SE23Covariance Qd = se23DiscreteQd_VanLoan(Fc, Qt, dt);

  EXPECT(assert_equal(Matrix(Qd), Matrix(Qd.transpose()), 1e-14));
  // leading order dt*Q~ (theta/nu/bias diagonal blocks dominate).
  EXPECT(assert_equal(Matrix(dt * Qt.block<3, 3>(0, 0)),
                      Matrix(Qd.block<3, 3>(0, 0)), 1e-6));
  Eigen::SelfAdjointEigenSolver<SE23Covariance> es(Qd);
  EXPECT(es.eigenvalues()(0) > -1e-14);  // PSD
}

/* ************************************************************************* */
// "Ours" is 4th-order accurate vs the exact Van Loan: ||Q_VL - Q_Ours|| should
// fall ~2^5 = 32x when dt is halved (ref §7/§8). A 2nd-order method (Brossard
// gain) would only fall ~8x.
TEST(SE23CovariancePropagation, OursIsFourthOrderVsVanLoan) {
  const SE23Covariance Fc = se23ContinuousFc(kW, kF, -1.0 / 100.0, -1.0 / 200.0);
  const SE23Covariance Qt = se23ContinuousQtilde(kACov, kWCov, kICov, kBA, kBG);

  auto err = [&](double dt) {
    return (se23DiscreteQd_VanLoan(Fc, Qt, dt) - se23DiscreteQd_Ours(Fc, Qt, dt))
        .norm();
  };
  const double dt = 0.1;
  const double ratio = err(dt) / err(dt / 2.0);
  // Expect ~32; allow a generous band that still excludes 3rd order (~8).
  EXPECT(ratio > 20.0);
  EXPECT(ratio < 50.0);
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
