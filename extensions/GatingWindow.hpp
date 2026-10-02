#pragma once

// Windowed gating pass-ratio per sensor — an alternative, differently-tuned
// view to SelfTrust's EWMA, not derived from it. Deliberately independent:
// the point is a second, simpler signal (raw pass-ratio over a fixed time
// window) to compare against the EWMA trust value during validation.

#include <deque>
#include <map>
#include <string>
#include <vector>

namespace parnav {

// pass_ratio + the window's total detection count (n_window), so the CSV
// writer can report both.
struct GatingSnapshot {
  double pass_ratio{1.0};
  int n_window{0};
};

class GatingWindow {
 public:
  explicit GatingWindow(double window_s = 10.0) : window_s_(window_s) {}

  // Step-level aggregate for one sensor at time t: n_passed out of n_total
  // detections/checks this step. Evicts entries older than window_s.
  void push(const std::string& sensor, double t, int n_total, int n_passed);

  // Sum of n_passed / n_total over entries within the window ending at the
  // most recently pushed time for this sensor. Returns 1.0 (vacuously
  // "passing") if the sensor has no entries in the window.
  GatingSnapshot passRatio(const std::string& sensor) const;

  std::map<std::string, GatingSnapshot> snapshot() const;
  void record();  // append snapshot to history
  const std::vector<std::map<std::string, GatingSnapshot>>& history() const {
    return history_;
  }

 private:
  struct Entry {
    double t;
    int n_total;
    int n_passed;
  };

  double window_s_;
  std::map<std::string, std::deque<Entry>> entries_;
  std::vector<std::map<std::string, GatingSnapshot>> history_;
};

}  // namespace parnav
