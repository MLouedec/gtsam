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

void SelfTrust::update(const std::string& sensor, bool is_good) {
  ensure(sensor);
  good_[sensor] = alpha1_ * good_[sensor] + (is_good ? 1.0 : 0.0);
  bad_[sensor] = alpha2_ * bad_[sensor] + (is_good ? 0.0 : 1.0);
}

void SelfTrust::decay(const std::string& sensor) {
  ensure(sensor);
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

std::map<std::string, double> SelfTrust::snapshot() const {
  std::map<std::string, double> out;
  for (const auto& kv : good_) out[kv.first] = get(kv.first);
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
