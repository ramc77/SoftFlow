#pragma once
#include "surrogate_model.h"
#include <vector>

namespace softflow {

// Ring buffer for collecting (input, target) training pairs on-the-fly
class TrainingData {
public:
    explicit TrainingData(int max_size = 10000);

    void addSample(const MLFeatures& features, const Vec2d& force);
    void clear();
    int size() const;
    bool isFull() const;

    const std::vector<MLFeatures>& getInputs() const { return inputs_; }
    const std::vector<Vec2d>& getTargets() const { return targets_; }

private:
    int max_size_;
    int write_pos_ = 0;
    bool full_ = false;
    std::vector<MLFeatures> inputs_;
    std::vector<Vec2d> targets_;
};

} // namespace softflow
