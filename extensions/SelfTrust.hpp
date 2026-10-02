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

  double alpha1_;
  double alpha2_;
  double default_;
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
