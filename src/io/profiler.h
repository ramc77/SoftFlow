#pragma once
#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace softflow {

// Per-module timing instrumentation for performance profiling
class Profiler {
public:
    struct Timer {
        std::string name;
        double total_seconds = 0.0;
        int call_count = 0;
        std::chrono::high_resolution_clock::time_point start;
    };

    int addTimer(const std::string& name) {
        int id = static_cast<int>(timers_.size());
        timers_.push_back({name, 0.0, 0, {}});
        return id;
    }

    void start(int id) {
        timers_[id].start = std::chrono::high_resolution_clock::now();
    }

    void stop(int id) {
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - timers_[id].start).count();
        timers_[id].total_seconds += elapsed;
        timers_[id].call_count++;
    }

    void reset() {
        for (auto& t : timers_) {
            t.total_seconds = 0.0;
            t.call_count = 0;
        }
    }

    void printReport(int total_steps) const {
        double total_time = 0.0;
        for (const auto& t : timers_) total_time += t.total_seconds;
        if (total_time < 1e-12) return;

        std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
        std::cout << "║             SoftFlow Performance Report                  ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Total steps: " << std::setw(10) << total_steps
                  << "   Total time: " << std::fixed << std::setprecision(3)
                  << std::setw(8) << total_time << " s" << std::setw(9) << "" << "║\n";

        if (total_steps > 0) {
            double us_per_step = total_time / total_steps * 1e6;
            std::cout << "║ Time/step: " << std::setw(10) << std::setprecision(1)
                      << us_per_step << " µs" << std::setw(27) << "" << "║\n";
        }

        std::cout << "╠══════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Module                    Time (s)     %      Calls    ║\n";
        std::cout << "╠──────────────────────────────────────────────────────────╣\n";

        // Sort by time descending for display
        std::vector<int> order(timers_.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return timers_[a].total_seconds > timers_[b].total_seconds;
        });

        for (int idx : order) {
            const auto& t = timers_[idx];
            if (t.call_count == 0) continue;
            double pct = t.total_seconds / total_time * 100.0;
            std::cout << "║  " << std::left << std::setw(24) << t.name
                      << std::right << std::fixed << std::setprecision(4)
                      << std::setw(9) << t.total_seconds
                      << std::setprecision(1) << std::setw(7) << pct << "%"
                      << std::setw(9) << t.call_count << "  ║\n";
        }

        std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    }

private:
    std::vector<Timer> timers_;
};

// RAII scope timer
class ScopeTimer {
public:
    ScopeTimer(Profiler& profiler, int id) : profiler_(profiler), id_(id) {
        profiler_.start(id_);
    }
    ~ScopeTimer() { profiler_.stop(id_); }
private:
    Profiler& profiler_;
    int id_;
};

} // namespace softflow
