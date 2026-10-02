#include "GatingWindow.hpp"

namespace parnav {

void GatingWindow::push(const std::string& sensor, double t, int n_total,
                        int n_passed) {
  auto& dq = entries_[sensor];
  dq.push_back({t, n_total, n_passed});
  while (!dq.empty() && dq.front().t < t - window_s_) {
    dq.pop_front();
  }
}

GatingSnapshot GatingWindow::passRatio(const std::string& sensor) const {
  auto it = entries_.find(sensor);
  if (it == entries_.end() || it->second.empty()) return {};
  int total = 0, passed = 0;
  for (const auto& e : it->second) {
    total += e.n_total;
    passed += e.n_passed;
  }
  if (total == 0) return {};
  return {static_cast<double>(passed) / total, total};
}

std::map<std::string, GatingSnapshot> GatingWindow::snapshot() const {
  std::map<std::string, GatingSnapshot> out;
  for (const auto& kv : entries_) out[kv.first] = passRatio(kv.first);
  return out;
}

void GatingWindow::record() { history_.push_back(snapshot()); }

}  // namespace parnav
