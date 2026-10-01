/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    testManifoldPreintegrationT.cpp
 * @brief   Unit tests for the bias-templated SE3/NavState preintegrator
 *          ManifoldPreintegrationT with GaussMarkovBias in-window decay.
 */

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/navigation/ManifoldPreintegrationT.h>

#include <CppUnitLite/TestHarness.h>

#include "imuFactorTesting.h"

/* ************************************************************************* */
// GaussMarkovBias: the analytic bias Jacobian must match re-integration for
// both in-window decay settings (on: step k debiased with exp(-t_k/tau) * b_i;
// off: frozen b_i) and both legacy increments. tau is short vs the ~1 s window
// so the decay matters.
namespace {
using GMBias = imuBias::GaussMarkovBias;
using PIM = ManifoldPreintegrationT<GMBias>;
const GMBias kGMBiasHat(Vector3(0.01, -0.02, 0.005),
                        Vector3(-0.003, 0.004, 0.002), 0.5, 0.7);

PIM gmPim(const GMBias& b, bool decay, PIM::LegacyIncrement inc) {
  PIM pim(testing::Params(), b);
  pim.setLegacyIncrement(inc);
  pim.setGMInWindowDecay(decay);
  testing::integrateMeasurements(testing::SomeMeasurements(), &pim);
  return pim;
}

std::pair<Matrix, Matrix> gmBiasJacobianNumericVsAnalytic(
    bool decay, PIM::LegacyIncrement inc) {
  Matrix96 H;
  gmPim(kGMBiasHat, decay, inc).biasCorrectedDelta(kGMBiasHat, H);
  // biasCorrectedDelta uses the [Log(R), p, v] chart (upstream convention).
  std::function<Vector9(const GMBias&)> f = [&](const GMBias& b) {
    const PIM pim = gmPim(b, decay, inc);
    Vector9 xi;
    NavState::dR(xi) = Rot3::Logmap(pim.deltaRij());
    NavState::dP(xi) = pim.deltaPij();
    NavState::dV(xi) = pim.deltaVij();
    return xi;
  };
  return {numericalDerivative11<Vector9, GMBias>(f, kGMBiasHat), Matrix(H)};
}
}  // namespace

TEST(ManifoldPreintegrationT, GMBiasJacobian_Decay_Full) {
  auto r = gmBiasJacobianNumericVsAnalytic(
      true, PIM::LegacyIncrement::ConstantBodyImu);
  EXPECT(assert_equal(r.first, r.second, 1e-5));
}

TEST(ManifoldPreintegrationT, GMBiasJacobian_Frozen_Full) {
  auto r = gmBiasJacobianNumericVsAnalytic(
      false, PIM::LegacyIncrement::ConstantBodyImu);
  EXPECT(assert_equal(r.first, r.second, 1e-5));
}

TEST(ManifoldPreintegrationT, GMBiasJacobian_Decay_Gtsam) {
  auto r = gmBiasJacobianNumericVsAnalytic(
      true, PIM::LegacyIncrement::GtsamStandard);
  EXPECT(assert_equal(r.first, r.second, 1e-5));
}

TEST(ManifoldPreintegrationT, GMBiasJacobian_Frozen_Gtsam) {
  auto r = gmBiasJacobianNumericVsAnalytic(
      false, PIM::LegacyIncrement::GtsamStandard);
  EXPECT(assert_equal(r.first, r.second, 1e-5));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
