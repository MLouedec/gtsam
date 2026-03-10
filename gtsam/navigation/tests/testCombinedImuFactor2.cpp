/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    testCombinedImuFactor2.cpp
 * @brief   Unit tests for CombinedImuFactor2 (SE_2(3), ExtendedPose3, bias-templated).
 */

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/navigation/CombinedImuFactor2.h>

#include <CppUnitLite/TestHarness.h>

#include "imuFactorTesting.h"

using namespace std::placeholders;

namespace {

using GMBias = imuBias::GaussMarkovBias;
using PIM_CB = PreintegratedCombinedMeasurements2T<
    ManifoldPreintegrationSE23<imuBias::ConstantBias>, imuBias::ConstantBias>;
using PIM_GM =
    PreintegratedCombinedMeasurements2T<ManifoldPreintegrationSE23<GMBias>,
                                        GMBias>;
using Factor_CB = CombinedImuFactor2T<PIM_CB, imuBias::ConstantBias>;
using Factor_GM = CombinedImuFactor2T<PIM_GM, GMBias>;

std::shared_ptr<PreintegrationCombinedParamsT<imuBias::ConstantBias>>
CombinedParamsCB() {
  auto p = PreintegrationCombinedParamsT<
      imuBias::ConstantBias>::MakeSharedD(kGravity);
  p->gyroscopeCovariance = kGyroSigma * kGyroSigma * I_3x3;
  p->accelerometerCovariance = kAccelSigma * kAccelSigma * I_3x3;
  p->integrationCovariance = 1e-6 * I_3x3;
  p->biasAccCovariance = 1e-6 * I_3x3;
  p->biasOmegaCovariance = 1e-7 * I_3x3;
  return p;
}

std::shared_ptr<PreintegrationCombinedParamsT<GMBias>> CombinedParamsGM() {
  auto p = PreintegrationCombinedParamsT<GMBias>::MakeSharedD(kGravity);
  p->gyroscopeCovariance = kGyroSigma * kGyroSigma * I_3x3;
  p->accelerometerCovariance = kAccelSigma * kAccelSigma * I_3x3;
  p->integrationCovariance = 1e-6 * I_3x3;
  p->biasAccCovariance = 1e-6 * I_3x3;
  p->biasOmegaCovariance = 1e-7 * I_3x3;
  return p;
}

}  // namespace

/* ************************************************************************* */
// The three covariance methods must all yield SPD covariances; Ours and Van
// Loan agree to high order, Brossard is close but measurably different.
TEST(CombinedImuFactor2, CovarianceMethodsAgree) {
  auto p = CombinedParamsCB();
  testing::SomeMeasurements ms;
  auto build = [&](SE23CovarianceMethod m) {
    PIM_CB pim(p, imuBias::ConstantBias(),
               Eigen::Matrix<double, 15, 15>::Zero(),
               SE23IncrementModel::SimpleGlobalAcc, m);
    for (const auto& x : ms) pim.integrateMeasurement(x.acc, x.gyro, x.dt);
    return Matrix(pim.preintMeasCov());
  };
  const Matrix Qb = build(SE23CovarianceMethod::Brossard);
  const Matrix Qo = build(SE23CovarianceMethod::Ours);
  const Matrix Qv = build(SE23CovarianceMethod::VanLoan);

  for (const Matrix* Q : {&Qb, &Qo, &Qv}) {
    Eigen::LLT<Matrix> llt(*Q);
    EXPECT(llt.info() == Eigen::Success);  // SPD
  }
  // Ours ~= VanLoan (both high order) at dt = 0.01.
  EXPECT((Qv - Qo).norm() / Qv.norm() < 1e-6);
  // Brossard in the same ballpark but distinct (Jensen deficit, O(dt^3)).
  const double relB = (Qv - Qb).norm() / Qv.norm();
  EXPECT(relB < 0.05);
  EXPECT(relB > 1e-9);
}

/* ************************************************************************* */
// Zero residual at the predicted state.
TEST(CombinedImuFactor2, ZeroResidualAtPrediction) {
  auto p = CombinedParamsCB();
  PIM_CB pim(p);
  testing::SomeMeasurements measurements;
  for (const auto& m : measurements) pim.integrateMeasurement(m.acc, m.gyro, m.dt);

  const ExtendedPose3 x1(Rot3::Ypr(0.1, -0.2, 0.3), Vector3(0.3, 0.1, -0.05),
                         Point3(0.5, -0.3, 0.2));
  const Bias bias;
  const ExtendedPose3 x2 = pim.predict(x1, bias);

  Factor_CB factor(X(1), X(2), B(1), B(2), pim);
  const Vector r = factor.evaluateError(x1, x2, bias, bias, nullptr, nullptr,
                                        nullptr, nullptr);
  EXPECT_LONGS_EQUAL(15, r.size());
  EXPECT(assert_equal(Vector(Vector::Zero(15)), Vector(r), 1e-8));
}

/* ************************************************************************* */
// Numerical Jacobians — ConstantBias.
TEST(CombinedImuFactor2, JacobiansConstantBias) {
  auto p = CombinedParamsCB();
  PIM_CB pim(p);
  testing::SomeMeasurements measurements;
  for (const auto& m : measurements) pim.integrateMeasurement(m.acc, m.gyro, m.dt);

  Factor_CB factor(X(1), X(2), B(1), B(2), pim);

  const ExtendedPose3 x1(Rot3::Yaw(0.15), Vector3(0.2, -0.1, 0.05),
                         Point3(0.5, -0.2, 0.1));
  const ExtendedPose3 x2 = pim.predict(x1, Bias());
  const Bias bias_i(Vector3(0.01, -0.02, 0.005),
                    Vector3(-0.001, 0.002, 0.003));
  const Bias bias_j(Vector3(0.015, -0.018, 0.004),
                    Vector3(-0.0012, 0.0021, 0.0032));

  Matrix H1, H2, H3, H4;
  factor.evaluateError(x1, x2, bias_i, bias_j, &H1, &H2, &H3, &H4);

  std::function<Vector(const ExtendedPose3&, const ExtendedPose3&, const Bias&,
                       const Bias&)>
      f = [&](const ExtendedPose3& a, const ExtendedPose3& b, const Bias& c,
              const Bias& d) {
        return factor.evaluateError(a, b, c, d, nullptr, nullptr, nullptr,
                                    nullptr);
      };

  EXPECT(assert_equal(numericalDerivative41<Vector, ExtendedPose3, ExtendedPose3,
                                            Bias, Bias>(f, x1, x2, bias_i,
                                                        bias_j),
                      H1, 1e-4));
  EXPECT(assert_equal(numericalDerivative42<Vector, ExtendedPose3, ExtendedPose3,
                                            Bias, Bias>(f, x1, x2, bias_i,
                                                        bias_j),
                      H2, 1e-4));
  EXPECT(assert_equal(numericalDerivative43<Vector, ExtendedPose3, ExtendedPose3,
                                            Bias, Bias>(f, x1, x2, bias_i,
                                                        bias_j),
                      H3, 1e-4));
  EXPECT(assert_equal(numericalDerivative44<Vector, ExtendedPose3, ExtendedPose3,
                                            Bias, Bias>(f, x1, x2, bias_i,
                                                        bias_j),
                      H4, 1e-4));
}

/* ************************************************************************* */
// Numerical Jacobians — GaussMarkovBias.
TEST(CombinedImuFactor2, JacobiansGaussMarkovBias) {
  auto p = CombinedParamsGM();
  const double tauAcc = 600.0, tauGyro = 1200.0;
  GMBias biasHat(Z_3x1, Z_3x1, tauAcc, tauGyro);
  PIM_GM pim(p, biasHat);
  testing::SomeMeasurements measurements;
  for (const auto& m : measurements) pim.integrateMeasurement(m.acc, m.gyro, m.dt);

  Factor_GM factor(X(1), X(2), B(1), B(2), pim);

  const ExtendedPose3 x1(Rot3::Yaw(0.1), Vector3(0.2, -0.1, 0.0),
                         Point3(0.3, -0.1, 0.05));
  const ExtendedPose3 x2 = pim.predict(x1, biasHat);
  const GMBias bias_i(Vector3(0.01, -0.02, 0.005),
                      Vector3(-0.001, 0.002, 0.003), tauAcc, tauGyro);
  const GMBias bias_j(Vector3(0.012, -0.018, 0.006),
                      Vector3(-0.0011, 0.0019, 0.0028), tauAcc, tauGyro);

  Matrix H1, H2, H3, H4;
  factor.evaluateError(x1, x2, bias_i, bias_j, &H1, &H2, &H3, &H4);

  std::function<Vector(const ExtendedPose3&, const ExtendedPose3&, const GMBias&,
                       const GMBias&)>
      f = [&](const ExtendedPose3& a, const ExtendedPose3& b,
              const GMBias& c, const GMBias& d) {
        return factor.evaluateError(a, b, c, d, nullptr, nullptr, nullptr,
                                    nullptr);
      };

  EXPECT(assert_equal(numericalDerivative41<Vector, ExtendedPose3, ExtendedPose3,
                                            GMBias, GMBias>(f, x1, x2, bias_i,
                                                            bias_j),
                      H1, 1e-4));
  EXPECT(assert_equal(numericalDerivative42<Vector, ExtendedPose3, ExtendedPose3,
                                            GMBias, GMBias>(f, x1, x2, bias_i,
                                                            bias_j),
                      H2, 1e-4));
  EXPECT(assert_equal(numericalDerivative43<Vector, ExtendedPose3, ExtendedPose3,
                                            GMBias, GMBias>(f, x1, x2, bias_i,
                                                            bias_j),
                      H3, 1e-4));
  EXPECT(assert_equal(numericalDerivative44<Vector, ExtendedPose3, ExtendedPose3,
                                            GMBias, GMBias>(f, x1, x2, bias_i,
                                                            bias_j),
                      H4, 1e-4));
}

/* ************************************************************************* */
// Covariance block-layout pin-down. After integration the preintMeasCov_ is
// [theta, nu, rho, bias_acc, bias_gyro] (15x15). Verify structural properties:
//   - symmetric, positive-diagonal.
//   - the (rho, rho) block (rows/cols 6..8) is strictly larger in trace than
//     the (bias_gyro, bias_gyro) block because rho has both acc noise AND
//     integration covariance injection.
TEST(CombinedImuFactor2, CovarianceLayout) {
  auto p = CombinedParamsCB();
  PIM_CB pim(p);
  testing::SomeMeasurements measurements;
  for (const auto& m : measurements) pim.integrateMeasurement(m.acc, m.gyro, m.dt);

  const Matrix M = pim.preintMeasCov();
  EXPECT_LONGS_EQUAL(15, M.rows());
  EXPECT_LONGS_EQUAL(15, M.cols());
  EXPECT(assert_equal(Matrix(M), Matrix(M.transpose()), 1e-12));

  for (int i = 0; i < 15; ++i) EXPECT(M(i, i) > 0.0);

  const double trRho = M.block<3, 3>(6, 6).trace();
  const double trBg = M.block<3, 3>(12, 12).trace();
  EXPECT(trRho > trBg);
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
