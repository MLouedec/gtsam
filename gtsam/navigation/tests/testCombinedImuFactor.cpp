/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    testCombinedImuFactor.cpp
 * @brief   Unit test for Lupton-style combined IMU factor
 * @author  Luca Carlone
 * @author  Frank Dellaert
 * @author  Richard Roberts
 * @author  Stephen Williams
 * @author  Varun Agrawal
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/MatrixConstants.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/VectorConstants.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/ScenarioRunner.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/factorTesting.h>

#include <list>

#include "imuFactorTesting.h"

namespace combined {
// Create default parameters with Z-down and above noise parameters
static std::shared_ptr<PreintegratedCombinedMeasurements::Params> Params(
    const Matrix3& biasAccCovariance = Matrix3::Zero(),
    const Matrix3& biasOmegaCovariance = Matrix3::Zero()) {
  auto p = PreintegratedCombinedMeasurements::Params::MakeSharedD(kGravity);
  p->gyroscopeCovariance = kGyroSigma * kGyroSigma * I_3x3;
  p->accelerometerCovariance = kAccelSigma * kAccelSigma * I_3x3;
  p->integrationCovariance = 0.0001 * I_3x3;
  p->biasAccCovariance = biasAccCovariance;
  p->biasOmegaCovariance = biasOmegaCovariance;
  return p;
}
}  // namespace combined

/* ************************************************************************* */
TEST_PIM_WITH_COMBINED_BACKEND(CombinedImuFactor, PreintegratedMeasurements) {
  // Linearization point
  Bias bias(
      Vector3(0, 0, 0),
      Vector3(
          0, 0,
          0));  ///< Current estimate of acceleration and angular rate biases

  // Measurements
  Vector3 measuredAcc(0.1, 0.0, 0.0);
  Vector3 measuredOmega(M_PI / 100.0, 0.0, 0.0);
  double deltaT = 0.5;
  double tol = 1e-6;

  auto p = combined::Params();

  // Actual preintegrated values
  PIM expected1(p, bias);
  expected1.integrateMeasurement(measuredAcc, measuredOmega, deltaT);

  CombinedPIM actual1(p, bias);

  actual1.integrateMeasurement(measuredAcc, measuredOmega, deltaT);

  EXPECT(assert_equal(Vector(expected1.deltaPij()), actual1.deltaPij(), tol));
  EXPECT(assert_equal(Vector(expected1.deltaVij()), actual1.deltaVij(), tol));
  EXPECT(assert_equal(expected1.deltaRij(), actual1.deltaRij(), tol));
  DOUBLES_EQUAL(expected1.deltaTij(), actual1.deltaTij(), tol);
}

/* ************************************************************************* */
// Checks equivalence between matching three-way and combined PIM backends.
TEST_PIM_WITH_COMBINED_BACKEND(CombinedImuFactor, ErrorWithBiases) {
  Bias bias(Vector3(0.2, 0, 0), Vector3(0, 0, 0.3));     // Biases (acc, rot)
  Bias bias2(Vector3(0.2, 0.2, 0), Vector3(1, 0, 0.3));  // Biases (acc, rot)
  Pose3 x1(Rot3::Expmap(Vector3(0, 0, M_PI / 4.0)), Point3(5.0, 1.0, -50.0));
  Vector3 v1(0.5, 0.0, 0.0);
  Pose3 x2(Rot3::Expmap(Vector3(0, 0, M_PI / 4.0 + M_PI / 10.0)),
           Point3(5.5, 1.0, -50.0));
  Vector3 v2(0.5, 0.0, 0.0);

  auto p = combined::Params();
  p->omegaCoriolis = Vector3(0, 0.1, 0.1);
  PIM pim(p, Bias(Vector3(0.2, 0.0, 0.0), Vector3(0.0, 0.0, 0.0)));

  // Measurements
  Vector3 measuredOmega{0, 0, M_PI / 10.0 + 0.3};
  Vector3 measuredAcc =
      x1.rotation().unrotate(-p->n_gravity) + Vector3(0.2, 0.0, 0.0);
  double deltaT = 1.0;
  double tol = 1e-6;

  pim.integrateMeasurement(measuredAcc, measuredOmega, deltaT);

  CombinedPIM combined_pim(
      p, Bias(Vector3(0.2, 0.0, 0.0), Vector3(0.0, 0.0, 0.0)));

  combined_pim.integrateMeasurement(measuredAcc, measuredOmega, deltaT);

  // Create factor
  ImuFactorT<PIM> imuFactor(X(1), V(1), X(2), V(2), B(1), pim);

  CombinedImuFactorT<CombinedPIM> combinedFactor(X(1), V(1), X(2), V(2), B(1),
                                                 B(2), combined_pim);

  Vector errorExpected = imuFactor.evaluateError(x1, v1, x2, v2, bias);
  Vector errorActual =
      combinedFactor.evaluateError(x1, v1, x2, v2, bias, bias2);
  EXPECT(assert_equal(errorExpected, errorActual.head(9), tol));

  // Expected Jacobians
  Matrix H1e, H2e, H3e, H4e, H5e;
  (void)imuFactor.evaluateError(x1, v1, x2, v2, bias, H1e, H2e, H3e, H4e, H5e);

  // Actual Jacobians
  Matrix H1a, H2a, H3a, H4a, H5a, H6a;
  (void)combinedFactor.evaluateError(x1, v1, x2, v2, bias, bias2, H1a, H2a, H3a,
                                     H4a, H5a, H6a);

  EXPECT(assert_equal(H1e, H1a.topRows(9)));
  EXPECT(assert_equal(H2e, H2a.topRows(9)));
  EXPECT(assert_equal(H3e, H3a.topRows(9)));
  EXPECT(assert_equal(H4e, H4a.topRows(9)));
  EXPECT(assert_equal(H5e, H5a.topRows(9)));
}

/* ************************************************************************* */
// This test only works with tangent preintegration.
TEST(CombinedImuFactor, FirstOrderPreIntegratedMeasurements) {
  auto p = combined::Params();
  testing::SomeMeasurements measurements;

  auto preintegrated = [&](const Vector3& a, const Vector3& w) {
    PreintegratedImuMeasurementsT<TangentPreintegration> pim(p, Bias(a, w));
    testing::integrateMeasurements(measurements, &pim);
    return pim.preintegrated();
  };

  // Actual pre-integrated values
  PreintegratedCombinedMeasurementsT<TangentPreintegration<imuBias::ConstantBias>, imuBias::ConstantBias> pim(p);
  testing::integrateMeasurements(measurements, &pim);

  EXPECT(assert_equal(numericalDerivative21<Vector9, Vector3, Vector3>(
                          preintegrated, Z_3x1, Z_3x1),
                      pim.preintegrated_H_biasAcc()));
  EXPECT(assert_equal(numericalDerivative22<Vector9, Vector3, Vector3>(
                          preintegrated, Z_3x1, Z_3x1),
                      pim.preintegrated_H_biasOmega(), 1e-3));
}

/* ************************************************************************* */
TEST_PIM(CombinedImuFactor, PredictPositionAndVelocity) {
  const Bias bias(Vector3(0, 0.1, 0), Vector3(0, 0.1, 0));  // Biases (acc, rot)

  auto p = combined::Params();

  // Measurements
  const Vector3 measuredOmega(0, 0.1, 0);  // M_PI/10.0+0.3;
  const Vector3 measuredAcc(0, 1.1, -kGravity);
  const double deltaT = 0.01;

  CombinedPIM pim(p, bias);

  for (int i = 0; i < 100; ++i)
    pim.integrateMeasurement(measuredAcc, measuredOmega, deltaT);

  // Create factor
  const noiseModel::Gaussian::shared_ptr combinedmodel =
      noiseModel::Gaussian::Covariance(pim.preintMeasCov());
  const CombinedImuFactorT<CombinedPIM> Combinedfactor(X(1), V(1), X(2), V(2),
                                                       B(1), B(2), pim);

  // Predict
  const NavState actual = pim.predict(NavState(), bias);
  const Pose3 expectedPose(Rot3(), Point3(0, 0.5, 0));
  const Vector3 expectedVelocity(0, 1, 0);
  EXPECT(assert_equal(expectedPose, actual.pose()));
  EXPECT(assert_equal(Vector(expectedVelocity), Vector(actual.velocity())));
}

/* ************************************************************************* */
TEST_PIM(CombinedImuFactor, PredictRotation) {
  const Bias bias(Vector3(0, 0, 0), Vector3(0, 0, 0));  // Biases (acc, rot)
  auto p = combined::Params();
  CombinedPIM pim(p, bias);
  const Vector3 measuredAcc = -kGravityAlongNavZDown;
  const Vector3 measuredOmega(0, 0, M_PI / 10.0);
  const double deltaT = 0.01;
  const double tol = 1e-4;
  for (int i = 0; i < 100; ++i)
    pim.integrateMeasurement(measuredAcc, measuredOmega, deltaT);
  const CombinedImuFactorT<CombinedPIM> Combinedfactor(X(1), V(1), X(2), V(2),
                                                       B(1), B(2), pim);

  // Predict
  const Pose3 x(Rot3::Ypr(0, 0, 0), Point3(0, 0, 0)), x2;
  const Vector3 v(0, 0, 0);
  const NavState actual = pim.predict(NavState(x, v), bias);
  const Pose3 expectedPose(Rot3::Ypr(M_PI / 10, 0, 0), Point3(0, 0, 0));
  EXPECT(assert_equal(expectedPose, actual.pose(), tol));
}

/* ************************************************************************* */
// Testing covariance to check if all the jacobians are accounted for.
TEST(CombinedImuFactor, CheckDefaultCovarianceRegression) {
  auto params = PreintegrationCombinedParams::MakeSharedU(9.81);

  params->setAccelerometerCovariance(pow(0.01, 2) * I_3x3);
  params->setGyroscopeCovariance(pow(1.75e-4, 2) * I_3x3);
  params->setIntegrationCovariance(pow(0.0, 2) * I_3x3);
  params->setOmegaCoriolis(Vector3::Zero());

  imuBias::ConstantBias currentBias;

  PreintegratedCombinedMeasurements actual(params, currentBias);

  // Measurements
  Vector3 measuredAcc(0.1577, -0.8251, 9.6111);
  Vector3 measuredOmega(-0.0210, 0.0311, 0.0145);
  double deltaT = 5;

  actual.integrateMeasurement(measuredAcc, measuredOmega, deltaT);

  Eigen::Matrix<double, 15, 15> expected{
      {1.53125e-07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 1.53125e-07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 1.53125e-07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0.003125, 0, 0, 0.00125, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0.003125, 0, 0, 0.00125, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0.003125, 0, 0, 0.00125, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0.00125, 0, 0, 0.0005, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0.00125, 0, 0, 0.0005, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0.00125, 0, 0, 0.0005, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 5., 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5., 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5., 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5., 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5., 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.}};

  // regression
  EXPECT(assert_equal(expected, actual.preintMeasCov()));
}

/* ************************************************************************* */
// Test that the covariance values for the ImuFactor and the CombinedImuFactor
// (top-left 9x9) are the same
TEST_PIM_WITH_COMBINED_BACKEND(CombinedImuFactor, SameCovariance) {
  // IMU measurements and time delta
  Vector3 accMeas(0.1577, -0.8251, 9.6111);
  Vector3 omegaMeas(-0.0210, 0.0311, 0.0145);
  double deltaT = 0.01;

  // Assume zero bias
  imuBias::ConstantBias currentBias;

  // Define params for ImuFactor
  auto params = PreintegrationParams::MakeSharedU();
  params->setAccelerometerCovariance(pow(0.01, 2) * I_3x3);
  params->setGyroscopeCovariance(pow(1.75e-4, 2) * I_3x3);
  params->setIntegrationCovariance(pow(0, 2) * I_3x3);
  params->setOmegaCoriolis(Vector3::Zero());

  // The IMU preintegration object for ImuFactor
  PIM pim(params, currentBias);
  pim.integrateMeasurement(accMeas, omegaMeas, deltaT);

  // Define params for CombinedImuFactor
  auto combined_params = PreintegrationCombinedParams::MakeSharedU();
  combined_params->setAccelerometerCovariance(pow(0.01, 2) * I_3x3);
  combined_params->setGyroscopeCovariance(pow(1.75e-4, 2) * I_3x3);
  // Set bias integration covariance explicitly to zero
  combined_params->setIntegrationCovariance(Z_3x3);
  combined_params->setOmegaCoriolis(Z_3x1);

  // The IMU preintegration object for CombinedImuFactor
  CombinedPIM cpim(combined_params, currentBias);
  cpim.integrateMeasurement(accMeas, omegaMeas, deltaT);

  // Assert if the noise covariance
  EXPECT(assert_equal<Matrix9>(
      pim.preintMeasCov(), Matrix9(cpim.preintMeasCov().block(0, 0, 9, 9))));
}

/* ************************************************************************* */
// Runners currently still use the integration type based on the compile flag.
// So we use the default CombinedScenarioRunner below.
TEST(CombinedImuFactor, Accelerating) {
  const double a = 0.2, v = 50;

  // Set up body pointing towards y axis, and start at 10,20,0 with velocity
  // going in X The body itself has Z axis pointing down
  const Rot3 nRb(Point3(0, 1, 0), Point3(1, 0, 0), Point3(0, 0, -1));
  const Point3 initial_position(10, 20, 0);
  const Vector3 initial_velocity(v, 0, 0);

  const AcceleratingScenario scenario(nRb, initial_position, initial_velocity,
                                      Vector3(a, 0, 0));

  const double T = 3.0;  // seconds

  CombinedScenarioRunner runner(scenario, combined::Params(), T / 10);

  PreintegratedCombinedMeasurements pim = runner.integrate(T);
  EXPECT(assert_equal(scenario.pose(T), runner.predict(pim).pose(), 1e-9));

  auto estimatedCov = runner.estimateCovariance(T, 100);
  Eigen::Matrix<double, 15, 15> expected = pim.preintMeasCov();
  EXPECT(assert_equal(estimatedCov, expected, 0.1));
}

/* ************************************************************************* */
// Numerical Jacobian test for GaussMarkov CombinedImuFactor
TEST(CombinedImuFactor, GaussMarkovJacobians) {
  using GMBias = imuBias::GaussMarkovBias;
  using GMPreintegration = ManifoldPreintegration<GMBias>;
  using GMPIM =
      PreintegratedCombinedMeasurementsT<GMPreintegration, GMBias>;
  using GMFactor = CombinedImuFactorT<GMPIM, GMBias>;

  // Create params
  auto p = PreintegrationCombinedParamsT<GMBias>::MakeSharedD(kGravity);
  p->gyroscopeCovariance = kGyroSigma * kGyroSigma * I_3x3;
  p->accelerometerCovariance = kAccelSigma * kAccelSigma * I_3x3;
  p->integrationCovariance = 0.0001 * I_3x3;
  p->biasAccCovariance = 0.001 * I_3x3;
  p->biasOmegaCovariance = 0.001 * I_3x3;

  double tauAcc = 300.0, tauGyro = 300.0;
  GMBias biasHat(Vector3(0.1, 0, 0), Vector3(0, 0, 0.01), tauAcc, tauGyro);
  GMPIM pim(p, biasHat);

  // Integrate a few measurements
  Vector3 measuredAcc(0.1, 0.0, -kGravity);
  Vector3 measuredOmega(0.01, 0.0, 0.02);
  double dt = 0.01;
  for (int i = 0; i < 100; i++) {
    pim.integrateMeasurement(measuredAcc, measuredOmega, dt);
  }

  // Linearization point
  Pose3 pose_i(Rot3::Expmap(Vector3(0.01, 0.02, 0.03)),
               Point3(1.0, 2.0, 3.0));
  Vector3 vel_i(0.5, 0.1, -0.1);
  Pose3 pose_j(Rot3::Expmap(Vector3(0.02, 0.03, 0.05)),
               Point3(1.5, 2.1, 2.9));
  Vector3 vel_j(0.6, 0.15, -0.05);
  GMBias bias_i(Vector3(0.12, 0.01, -0.01), Vector3(0.01, -0.01, 0.02),
                tauAcc, tauGyro);
  GMBias bias_j(Vector3(0.11, 0.015, -0.005), Vector3(0.008, -0.012, 0.018),
                tauAcc, tauGyro);

  GMFactor factor(X(1), V(1), X(2), V(2), B(1), B(2), pim);

  // Compute analytical Jacobians
  Matrix H1a, H2a, H3a, H4a, H5a, H6a;
  Vector error = factor.evaluateError(pose_i, vel_i, pose_j, vel_j, bias_i,
                                      bias_j, H1a, H2a, H3a, H4a, H5a, H6a);

  // Numerical Jacobians
  auto f1 = [&](const Pose3& p1) {
    return factor.evaluateError(p1, vel_i, pose_j, vel_j, bias_i, bias_j);
  };
  auto f2 = [&](const Vector3& v1) {
    return factor.evaluateError(pose_i, v1, pose_j, vel_j, bias_i, bias_j);
  };
  auto f3 = [&](const Pose3& p2) {
    return factor.evaluateError(pose_i, vel_i, p2, vel_j, bias_i, bias_j);
  };
  auto f4 = [&](const Vector3& v2) {
    return factor.evaluateError(pose_i, vel_i, pose_j, v2, bias_i, bias_j);
  };
  auto f5 = [&](const GMBias& b1) {
    return factor.evaluateError(pose_i, vel_i, pose_j, vel_j, b1, bias_j);
  };
  auto f6 = [&](const GMBias& b2) {
    return factor.evaluateError(pose_i, vel_i, pose_j, vel_j, bias_i, b2);
  };

  Matrix H1n = numericalDerivative11<Vector, Pose3>(f1, pose_i);
  Matrix H2n = numericalDerivative11<Vector, Vector3>(f2, vel_i);
  Matrix H3n = numericalDerivative11<Vector, Pose3>(f3, pose_j);
  Matrix H4n = numericalDerivative11<Vector, Vector3>(f4, vel_j);
  Matrix H5n = numericalDerivative11<Vector, GMBias>(f5, bias_i);
  Matrix H6n = numericalDerivative11<Vector, GMBias>(f6, bias_j);

  EXPECT(assert_equal(H1n, H1a, 1e-5));
  EXPECT(assert_equal(H2n, H2a, 1e-5));
  EXPECT(assert_equal(H3n, H3a, 1e-5));
  EXPECT(assert_equal(H4n, H4a, 1e-5));
  EXPECT(assert_equal(H5n, H5a, 1e-5));
  EXPECT(assert_equal(H6n, H6a, 1e-5));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
