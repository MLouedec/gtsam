#include "SelfTrust.hpp"

#include <cmath>
#include <stdexcept>

namespace parnav {

SelfTrust::SelfTrust(double alpha1, double alpha2, double default_trust)
    : alpha1_(alpha1), alpha2_(alpha2), default_(default_trust) {}

void SelfTrust::ensure(const std::string& sensor) {
  if (good_.find(sensor) == good_.end()) {
    good_[sensor] = 0.0;
    bad_[sensor] = 0.0;
  }
}

void SelfTrust::setForgetting(double forget_good, double forget_bad) {
  if (forget_good < 0.0 || forget_good >= 1.0 || forget_bad < 0.0 ||
      forget_bad >= 1.0)
    throw std::runtime_error("trust forgetting must be in [0, 1)");
  continuous_ = true;
  forget_good_ = forget_good;
  forget_bad_ = forget_bad;
}

void SelfTrust::setNominalRate(const std::string& sensor, double hz) {
  if (hz > 0.0) rate_[sensor] = hz;
}

void SelfTrust::advance(const std::string& sensor) {
  auto last = last_t_.find(sensor);
  if (last == last_t_.end()) {
    last_t_[sensor] = now_;  // first sighting: nothing to forget yet
    return;
  }
  const double dt = now_ - last->second;
  if (dt <= 0.0) return;  // already advanced this step
  auto r = rate_.find(sensor);
  const double periods = (r == rate_.end() ? default_rate_ : r->second) * dt;
  good_[sensor] *= std::pow(1.0 - forget_good_, periods);
  bad_[sensor] *= std::pow(1.0 - forget_bad_, periods);
  last->second = now_;
}

void SelfTrust::update(const std::string& sensor, bool is_good) {
  ensure(sensor);
  if (continuous_) {
    advance(sensor);
    good_[sensor] += is_good ? 1.0 : 0.0;
    bad_[sensor] += is_good ? 0.0 : 1.0;
    return;
  }
  good_[sensor] = alpha1_ * good_[sensor] + (is_good ? 1.0 : 0.0);
  bad_[sensor] = alpha2_ * bad_[sensor] + (is_good ? 0.0 : 1.0);
}

void SelfTrust::decay(const std::string& sensor) {
  ensure(sensor);
  if (continuous_) {
    advance(sensor);
    return;
  }
  good_[sensor] *= alpha1_;
  bad_[sensor] *= alpha2_;
}

double SelfTrust::get(const std::string& sensor) const {
  auto it = good_.find(sensor);
  if (it == good_.end()) return default_;
  double g = it->second;
  double b = bad_.at(sensor);
  return (g + 1.0) / (g + b + 2.0);
}

Opinion SelfTrust::opinion(const std::string& sensor) const {
  auto it = good_.find(sensor);
  if (it == good_.end()) return Opinion{0.0, 0.0, 1.0};
  double a = it->second;
  double b = bad_.at(sensor);
  double denom = a + b + 2.0;
  Opinion op;
  op.belief = a / denom;
  op.disbelief = b / denom;
  op.uncertainty = 2.0 / denom;
  return op;
}

std::map<std::string, Opinion> SelfTrust::snapshot() const {
  std::map<std::string, Opinion> out;
  for (const auto& kv : good_) out[kv.first] = opinion(kv.first);
  return out;
}

void SelfTrust::record() { history_.push_back(snapshot()); }

double trustScale(double trust, const TrustScalingCfg& cfg) {
  if (cfg.scaling == "off") return 1.0;
  double t_eff = trust < cfg.floor ? cfg.floor : trust;
  if (cfg.scaling == "inverse") return 1.0 / t_eff;
  if (cfg.scaling == "inverse_sqrt") return 1.0 / std::sqrt(t_eff);
  if (cfg.scaling == "linear") return 1.0 + cfg.linear_k * (1.0 - trust);
  throw std::runtime_error("unknown trust scaling: " + cfg.scaling);
}

}  // namespace parnav
