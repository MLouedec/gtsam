/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    testManifoldPreintegrationSE23.cpp
 * @brief   Unit tests for ManifoldPreintegrationSE23 (Barrau SE_2(3) form).
 */

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/navigation/ManifoldPreintegrationSE23.h>

#include <CppUnitLite/TestHarness.h>

#include "imuFactorTesting.h"

using namespace std::placeholders;

/* ************************************************************************* */
// The class's bias-Jacobian blocks live in the SE_2(3) tangent of the
// *biasCorrectedDelta*, not on the (R, v, p) fields of deltaXij_. Verify by
// comparing H = d(biasCorrectedDelta)/d(bias_i) against a numerical derivative
// around a small nonzero biasHat (so the linearization matches a finite-
// difference probe).
TEST(ManifoldPreintegrationSE23, BiasCorrectedDeltaJacobian) {
  testing::SomeMeasurements measurements;

  const Bias biasHat(Vector3(0.01, -0.02, 0.005),
                     Vector3(-0.003, 0.004, 0.002));
  ManifoldPreintegrationSE23<> pim(testing::Params(), biasHat);
  testing::integrateMeasurements(measurements, &pim);

  std::function<Vector9(const Bias&)> f = [&](const Bias& b) {
    return pim.biasCorrectedDelta(b, {});
  };

  Matrix96 aH;
  pim.biasCorrectedDelta(biasHat, aH);

  EXPECT(assert_equal(numericalDerivative11<Vector9, Bias>(f, biasHat),
                      Matrix(aH), 1e-6));
}

/* ************************************************************************* */
// Check predict() Jacobians with numerical derivatives.
TEST(ManifoldPreintegrationSE23, PredictJacobians) {
  testing::SomeMeasurements measurements;
  ManifoldPreintegrationSE23<> pim(testing::Params());
  testing::integrateMeasurements(measurements, &pim);

  const ExtendedPose3 x1(Rot3::Yaw(0.2), Vector3(0.4, -0.1, 0.05),
                         Point3(0.3, -0.2, 0.1));
  const Bias bias(Vector3(0.01, -0.02, 0.005), Vector3(-0.001, 0.002, 0.003));

  Matrix9 aH1;
  Matrix96 aH2;
  pim.predict(x1, bias, aH1, aH2);

  std::function<ExtendedPose3(const ExtendedPose3&, const Bias&)> f =
      [&](const ExtendedPose3& s, const Bias& b) {
        return pim.predict(s, b);
      };

  EXPECT(assert_equal(numericalDerivative21(f, x1, bias), aH1, 1e-5));
  EXPECT(assert_equal(numericalDerivative22(f, x1, bias), aH2, 1e-5));
}

/* ************************************************************************* */
// Check computeError() Jacobians with numerical derivatives.
TEST(ManifoldPreintegrationSE23, ComputeErrorJacobians) {
  testing::SomeMeasurements measurements;
  ManifoldPreintegrationSE23<> pim(testing::Params());
  testing::integrateMeasurements(measurements, &pim);

  const ExtendedPose3 x1(Rot3::Yaw(0.2), Vector3(0.4, -0.1, 0.05),
                         Point3(0.3, -0.2, 0.1));
  const ExtendedPose3 x2 = pim.predict(x1, Bias());
  const Bias bias(Vector3(0.01, -0.02, 0.005), Vector3(-0.001, 0.002, 0.003));

  Matrix9 aH1, aH2;
  Matrix96 aH3;
  pim.computeError(x1, x2, bias, aH1, aH2, aH3);

  std::function<Vector9(const ExtendedPose3&, const ExtendedPose3&,
                        const Bias&)>
      f = [&](const ExtendedPose3& a, const ExtendedPose3& b, const Bias& c) {
        return pim.computeError(a, b, c, nullptr, nullptr, nullptr);
      };

  EXPECT(assert_equal(numericalDerivative31(f, x1, x2, bias), aH1, 1e-5));
  EXPECT(assert_equal(numericalDerivative32(f, x1, x2, bias), aH2, 1e-5));
  EXPECT(assert_equal(numericalDerivative33(f, x1, x2, bias), aH3, 1e-5));
}

/* ************************************************************************* */
// Sanity: zero gyro, constant acc along +x with zero gravity should give a
// kinematic straight-line trajectory.
TEST(ManifoldPreintegrationSE23, StraightLineZeroGravity) {
  auto p = std::make_shared<PreintegrationParams>(Vector3::Zero());
  p->gyroscopeCovariance = 1e-8 * I_3x3;
  p->accelerometerCovariance = 1e-8 * I_3x3;
  p->integrationCovariance = 1e-8 * I_3x3;

  ManifoldPreintegrationSE23<> pim(p);
  const Vector3 acc(1.0, 0.0, 0.0);
  const Vector3 omega = Vector3::Zero();
  const double dt = 0.01;
  const int N = 100;
  for (int i = 0; i < N; ++i) pim.integrateMeasurement(acc, omega, dt);

  const double T = N * dt;  // 1.0 s
  const ExtendedPose3 x1;   // identity
  const ExtendedPose3 x2 = pim.predict(x1, Bias());

  EXPECT(assert_equal(Rot3(), x2.rotation(), 1e-9));
  // v = a * T = 1.0
  EXPECT(assert_equal(Vector3(1.0, 0, 0), x2.velocity(), 1e-6));
  // p = 0.5 * a * T^2 = 0.5
  EXPECT(assert_equal(Point3(0.5, 0, 0), x2.position(), 1e-6));
  EXPECT_DOUBLES_EQUAL(T, pim.deltaTij(), 1e-9);
}

/* ************************************************************************* */
// Sanity: zero bias -> computeError on predicted state is zero.
TEST(ManifoldPreintegrationSE23, ZeroResidualAtPrediction) {
  testing::SomeMeasurements measurements;
  ManifoldPreintegrationSE23<> pim(testing::Params());
  testing::integrateMeasurements(measurements, &pim);

  const ExtendedPose3 x1(Rot3::Ypr(0.1, -0.2, 0.3), Vector3(0.5, 0.1, -0.2),
                         Point3(1.0, 2.0, -0.5));
  const Bias bias;
  const ExtendedPose3 x2 = pim.predict(x1, bias);

  const Vector9 e =
      pim.computeError(x1, x2, bias, nullptr, nullptr, nullptr);
  EXPECT(assert_equal(Vector(Vector9::Zero()), Vector(e), 1e-8));
}

/* ************************************************************************* */
// Phase-A check: the one-step 9x9 transition A must be Ad(Yhat^{-1})*F_dt, whose
// top-left 3x3 block is Exp(-w_hat*dt) (ref assertion #4). Under the old
// A = F_dt (identity top-left) this fails.
TEST(ManifoldPreintegrationSE23, TransitionTopLeftIsExpNegOmega) {
  ManifoldPreintegrationSE23<> pim(testing::Params());  // zero biasHat
  const Vector3 acc(0.05, 0.09, 0.01), omega(0.03, 0.09, 0.06);
  const double dt = 0.01;
  Matrix9 A;
  Matrix93 B, C;
  pim.update(acc, omega, dt, &A, &B, &C);
  EXPECT(assert_equal(Rot3::Expmap(-omega * dt).matrix(),
                      Matrix3(A.block<3, 3>(0, 0)), 1e-12));
}

/* ************************************************************************* */
// Strong bias-Jacobian check: the accumulated J_bias must equal the first-order
// sensitivity of the *re-integrated* preintegrated delta to the linearization
// bias (ref assertion #9). This exercises the full transition A across the
// whole window; it fails under A = F_dt (missing the rotational coupling).
namespace {
Vector9 reintegratedLog(const testing::SomeMeasurements& ms, const Bias& b,
                        SE23IncrementModel model) {
  ManifoldPreintegrationSE23<> pim(testing::Params(), b, model);
  testing::integrateMeasurements(ms, &pim);
  return ExtendedPose3::Logmap(pim.deltaXij());
}

// Returns {numeric FD, analytic J_bias} for comparison inside a TEST.
std::pair<Matrix, Matrix> biasJacobianNumericVsAnalytic(
    SE23IncrementModel model) {
  testing::SomeMeasurements ms;
  const Bias biasHat(Vector3(0.01, -0.02, 0.005),
                     Vector3(-0.003, 0.004, 0.002));
  ManifoldPreintegrationSE23<> pim(testing::Params(), biasHat, model);
  testing::integrateMeasurements(ms, &pim);

  Matrix96 H;
  pim.biasCorrectedDelta(biasHat, H);

  std::function<Vector9(const Bias&)> f = [&](const Bias& b) {
    return reintegratedLog(ms, b, model);
  };
  return {numericalDerivative11<Vector9, Bias>(f, biasHat), Matrix(H)};
}
}  // namespace

TEST(ManifoldPreintegrationSE23, BiasJacobianVsReintegration_Simple) {
  auto r = biasJacobianNumericVsAnalytic(SE23IncrementModel::SimpleGlobalAcc);
  EXPECT(assert_equal(r.first, r.second, 1e-5));
}

TEST(ManifoldPreintegrationSE23, BiasJacobianVsReintegration_Full) {
  auto r = biasJacobianNumericVsAnalytic(SE23IncrementModel::ConstantBodyImu);
  EXPECT(assert_equal(r.first, r.second, 1e-5));
}

/* ************************************************************************* */
// The two increment models must agree exactly when omega == 0 (J_l(0)=I,
// C_hat(0,dt)=dt^2/2 I), a sanity check on the ConstantBodyImu math.
TEST(ManifoldPreintegrationSE23, SimpleFullAgreeZeroOmega) {
  auto p = std::make_shared<PreintegrationParams>(Vector3(0, 0, 9.81));
  ManifoldPreintegrationSE23<> pimS(p, Bias(), SE23IncrementModel::SimpleGlobalAcc);
  ManifoldPreintegrationSE23<> pimF(p, Bias(), SE23IncrementModel::ConstantBodyImu);
  const Vector3 acc(0.3, -0.2, 0.1), omega = Vector3::Zero();
  for (int i = 0; i < 50; ++i) {
    pimS.integrateMeasurement(acc, omega, 0.01);
    pimF.integrateMeasurement(acc, omega, 0.01);
  }
  EXPECT(assert_equal(pimS.deltaXij(), pimF.deltaXij(), 1e-12));
}

/* ************************************************************************* */
// predict()/computeError() Jacobians for the ConstantBodyImu increment model.
TEST(ManifoldPreintegrationSE23, FullModelPredictJacobians) {
  testing::SomeMeasurements measurements;
  ManifoldPreintegrationSE23<> pim(testing::Params(), Bias(),
                                   SE23IncrementModel::ConstantBodyImu);
  testing::integrateMeasurements(measurements, &pim);

  const ExtendedPose3 x1(Rot3::Yaw(0.2), Vector3(0.4, -0.1, 0.05),
                         Point3(0.3, -0.2, 0.1));
  const Bias bias(Vector3(0.01, -0.02, 0.005), Vector3(-0.001, 0.002, 0.003));

  Matrix9 aH1;
  Matrix96 aH2;
  pim.predict(x1, bias, aH1, aH2);
  std::function<ExtendedPose3(const ExtendedPose3&, const Bias&)> f =
      [&](const ExtendedPose3& s, const Bias& b) { return pim.predict(s, b); };
  EXPECT(assert_equal(numericalDerivative21(f, x1, bias), aH1, 1e-5));
  EXPECT(assert_equal(numericalDerivative22(f, x1, bias), aH2, 1e-5));
}

/* ************************************************************************* */
// The ConstantBodyImu model must reproduce the constant-body-force trajectory
// (its own assumption) at zero noise/bias -- this is the ~1.7 m position bug.
// Truth is an INDEPENDENT fine numerical integration (no kernel helpers).
TEST(ManifoldPreintegrationSE23, FullModelReproducesConstantBodyTruth) {
  auto p = std::make_shared<PreintegrationParams>(Vector3::Zero());  // no gravity
  p->gyroscopeCovariance = 1e-10 * I_3x3;
  p->accelerometerCovariance = 1e-10 * I_3x3;
  p->integrationCovariance = 1e-10 * I_3x3;
  ManifoldPreintegrationSE23<> pim(p, Bias(),
                                   SE23IncrementModel::ConstantBodyImu);
  const Vector3 acc(0.3, -0.2, 0.1), omega(0.4, -0.3, 0.6);  // real rotation
  const double dt = 0.01;
  const int N = 100;
  const double T = N * dt;
  for (int i = 0; i < N; ++i) pim.integrateMeasurement(acc, omega, dt);
  const ExtendedPose3 x = pim.predict(ExtendedPose3(), Bias());

  // v(t) = int_0^t R(s) acc ds, p(t) = int_0^t v ds, R(t) = Exp(omega t).
  const int Nf = 100000;
  const double h = T / Nf;
  Vector3 v_t = Vector3::Zero(), p_t = Vector3::Zero();
  for (int k = 0; k < Nf; ++k) {
    const Vector3 Rf = Rot3::Expmap(omega * ((k + 0.5) * h)) * acc;
    p_t += (v_t + 0.5 * Rf * h) * h;
    v_t += Rf * h;
  }
  EXPECT(assert_equal(Rot3::Expmap(omega * T), x.rotation(), 1e-9));
  EXPECT(assert_equal(v_t, x.velocity(), 1e-4));
  EXPECT(assert_equal(Point3(p_t), x.position(), 1e-4));
}

/* ************************************************************************* */
// Full-model bias Jacobian at LARGE omega (where the old d_p kernel was wrong):
// analytic J_bias vs FD of re-integration.
TEST(ManifoldPreintegrationSE23, FullModelBiasJacobianLargeOmega) {
  const Bias biasHat(Vector3(0.01, -0.02, 0.005),
                     Vector3(-0.003, 0.004, 0.002));
  auto build = [](const Bias& b) {
    ManifoldPreintegrationSE23<> pim(testing::Params(), b,
                                     SE23IncrementModel::ConstantBodyImu);
    const Vector3 acc(0.05, 0.09, 0.01), omega(0.4, -0.3, 0.6);
    for (int i = 0; i < 30; ++i) pim.integrateMeasurement(acc, omega, 0.01);
    return pim;
  };
  auto pim = build(biasHat);
  Matrix96 H;
  pim.biasCorrectedDelta(biasHat, H);
  std::function<Vector9(const Bias&)> f = [&](const Bias& b) {
    return ExtendedPose3::Logmap(build(b).deltaXij());
  };
  EXPECT(assert_equal(numericalDerivative11<Vector9, Bias>(f, biasHat),
                      Matrix(H), 1e-5));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
