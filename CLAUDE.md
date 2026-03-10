# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Behavior

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 0. Caveman

Speak like a caveman to use fewer tokens.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

# Project

## Overview

A **GTSAM fork** adding two capabilities to the navigation stack, kept fully separate from the legacy `NavState` code so existing factors are untouched:

1. **SE_2(3) preintegration** — an `ExtendedPose3` (Barrau `(R, v, p)` ordering) state type with sibling preintegrators and a 4-way combined IMU factor.
2. **Gauss-Markov IMU bias** — a first-order-decay bias type that slots into either preintegrator as a template argument.

The `{SE3, SE23} × {ConstantBias, GaussMarkovBias}` matrix all instantiates cleanly; runner programs select the combination at runtime.

## Build

CMake, out-of-source in `_build/`:

```bash
cmake -S . -B _build            # configure once
cmake --build _build --target ifac_wc_2026   # build one program
```

Each file in `programs/` becomes an executable at `_build/programs/<name>`, linked against `parnav_extensions gtsam gtsam_unstable`.

## SE_2(3) integration

| File | Role |
|---|---|
| `gtsam/geometry/ExtendedPose3.{h,cpp}` | SE_2(3) group; storage `(R, v, p)`, tangent order `[theta, nu, rho]` |
| `gtsam/navigation/PreintegrationSE23Base.{h,cpp}` | SE_2(3) sibling of `PreintegrationBase`, bias-templated |
| `gtsam/navigation/ManifoldPreintegrationSE23.{h,cpp}` | Manifold preintegrator; per-step left-trivialized `F_dt` + measurement Jacobian `G_j` |
| `gtsam/navigation/CombinedImuFactor2.h` | `CombinedImuFactor2T` on `(state_i, state_j, bias_i, bias_j)`; 15×15 covariance in `[theta, nu, rho, b_acc, b_gyro]` order |

## Gauss-Markov bias

- `gtsam/navigation/ImuBias.h` — `imuBias::GaussMarkovBias`: per-axis correlation times `tauAcc_`/`tauGyro_`; `correctAccelerometer`/`correctGyroscope` apply `beta = exp(-dt/tau)` decay with Jacobians. VectorSpace traits let it drop in as a bias template arg.
- `gtsam/navigation/PreintegrationCombinedParams.h` — templated on the bias type to carry GM params.

## Key conventions

- **Tangent / covariance ordering is `[theta, nu, rho]`** (Barrau `(R, v, p)`), NOT the legacy `(rho, nu)`. Watch this in every SE_2(3) covariance block.
- SE_2(3) code is a **sibling hierarchy**, never a modification of the NavState path — `ImuFactor`, `ImuFactor2`, `CombinedImuFactor` stay as-is.

## Extensions (`extensions/`)

Custom factors and helpers linked as `parnav_extensions`: aiding factors (`Baro*`, `Compass`, `Azimuth`, `Elevation`, `Range`, `GNSSAtt`, `MarkerAzimuth`, `ExtendedPoseAttitude`, `GPSFactorSE23`), preintegration math (`PreintegrationHelpers`), sim I/O (`MultiModalSimLoader`), data association (`MarkerAssociation`), and trust (`SelfTrust`).

## Runner programs (`programs/`)

| Program | Purpose |
|---|---|
| `ifac_wc_2026.cpp` | Main paper harness (multirotor PARS/GNSS/compass/baro). Runtime-selects the variant. |
| `SimulationFixedLagSmoother.cpp` | Templated `<BIAS, UseSE23>` INS/GNSS/PARS sim; outputs get `_se23` suffix when `UseSE23`. |
| `MultiModalFixedLag3D.cpp` | 3D fixed-lag on the multi-modal-simulator `.npz` export. |
| `run_multirotor_cs_df.cpp` / `..._known_baro.cpp` | CS/DF fusion ports; `_known_baro` uses a fixed pressure bias (Strategy B) instead of a live `D(0)` state. |
| `DiagnoseSE23vsSE3Predict.cpp` | Lock-step SE3-vs-SE23 divergence finder (bias-free/noise-free IMU must match). |

**`ifac_wc_2026` CLI** — key flags: `--preint {se3|se23}` (default `se3`), `--bias {cb|gm}` (default `cb`), `--handover {none|angle|angle-baro|angle-range}`, `--robust {none|gm|tukey}`, plus baro/compass/noise tuning (`--noise-scaling`, `--bias-scaling`, `--baro-origin-msl`, `--compass-lag-ticks`, …). See `--help`.

## Tests

Standard GTSAM CTest layout under `gtsam/**/tests/`. New coverage:
- `gtsam/geometry/tests/testExtendedPose3.cpp`
- `gtsam/navigation/tests/testManifoldPreintegrationSE23.cpp`
- `gtsam/navigation/tests/testCombinedImuFactor2.cpp`
- `gtsam/navigation/tests/testPose3AttitudeFactor.cpp`

## Visualisation & analysis (`visualisations/`, `programs/scripts/`)

- `visualisations/plot_3sigma.py` — 5×3 error + ±3σ grid overlaying the four `gtsam_fork_test_{cb,gm}{,_se23}.csv` variants.
- `visualisations/plot_3sigma_from_truth.py` — pos/vel/att error vs a truth CSV for `--handover none` runs.
- `programs/scripts/`: `analyze_position.py` (RMSE + intervals), `analyze_yaw_zigzag.py`, `characterise_imu.py` (Allan variance → PreintegrationParams sigmas).

## Runner scripts

`run_handover_test.sh` (sweeps `--robust`), `run_cs_df_scenarios.sh`, `sweep_script.sh`, `programs/run_all_configs.sh`, `programs/scripts/sweep_yaw_zigzag.sh`.
