#pragma once

// Per-sensor self-trust (Beta-reputation with exponential forgetting).
// One-for-one C++ port of `lib/sensor_trust.py::SelfTrust` from the
// multi-modale-simulator. Stateless w.r.t. GTSAM types; the smoother layer
// converts trust to a sigma multiplier via `trustScale()`.

#include <map>
#include <string>
#include <vector>

namespace parnav {

// Subjective-logic opinion derived from the same good_/bad_ Beta counters as
// trust: belief B=a/(a+b+2), disbelief D=b/(a+b+2), uncertainty U=2/(a+b+2)
// (B+D+U=1). trust==tau==B+0.5*U is algebraically identical to get()'s
// (a+1)/(a+b+2) — use get() for the scalar, this struct is belief/disbelief/
// uncertainty only.
struct Opinion {
  double belief{0.0};
  double disbelief{0.0};
  double uncertainty{1.0};
};

class SelfTrust {
 public:
  SelfTrust(double alpha1 = 0.9, double alpha2 = 0.99,
            double default_trust = 0.5);

  // Switch to continuous-time forgetting. forget_good/forget_bad are the
  // fractions of the good/bad counters forgotten per *nominal measurement
  // period* of each sensor (e.g. 0.01 = 1%). Per step the counters are
  // multiplied by (1-forget)^(f_nom * dt_elapsed), so the decay is the same
  // in wall-clock time whatever the smoother step rate, and composes exactly
  // over irregular steps. alpha1/alpha2 are then unused.
  void setForgetting(double forget_good, double forget_bad);
  // Nominal measurement rate (Hz) used by a key (default: default_rate).
  void setNominalRate(const std::string& sensor, double hz);
  void setDefaultRate(double hz) { default_rate_ = hz; }
  // Current smoother time (s); call once per step before any update/decay.
  // Only used in continuous-time mode.
  void setTime(double t) { now_ = t; }
  bool continuousTime() const { return continuous_; }

  void update(const std::string& sensor, bool is_good);
  void decay(const std::string& sensor);  // applied when sensor is silent
  double get(const std::string& sensor) const;
  Opinion opinion(const std::string& sensor) const;

  std::map<std::string, Opinion> snapshot() const;
  void record();  // append snapshot to history
  const std::vector<std::map<std::string, Opinion>>& history() const {
    return history_;
  }

 private:
  void ensure(const std::string& sensor);
  // Continuous-time mode: decay the counters of `sensor` for the time elapsed
  // since it was last advanced (no-op on the first call or within a step).
  void advance(const std::string& sensor);

  double alpha1_;
  double alpha2_;
  double default_;
  bool continuous_{false};
  double forget_good_{0.01};
  double forget_bad_{0.01};
  double default_rate_{1.0};
  double now_{0.0};
  std::map<std::string, double> rate_;
  std::map<std::string, double> last_t_;
  std::map<std::string, double> good_;
  std::map<std::string, double> bad_;
  std::vector<std::map<std::string, Opinion>> history_;
};

// Sigma-scaling laws matching Python `_trust_scale`.
struct TrustScalingCfg {
  std::string scaling{"inverse"};  // inverse | inverse_sqrt | linear | off
  double floor{0.001};
  double linear_k{5.0};
};

double trustScale(double trust, const TrustScalingCfg& cfg);

}  // namespace parnav
