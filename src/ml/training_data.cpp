#include "training_data.h"

namespace softflow {

TrainingData::TrainingData(int max_size)
    : max_size_(max_size) {
    inputs_.reserve(max_size);
    targets_.reserve(max_size);
}

void TrainingData::addSample(const MLFeatures& features, const Vec2d& force) {
    if (!full_) {
        inputs_.push_back(features);
        targets_.push_back(force);
        if (static_cast<int>(inputs_.size()) >= max_size_) {
            full_ = true;
            write_pos_ = 0;
        }
    } else {
        // Ring buffer: overwrite oldest
        inputs_[write_pos_] = features;
        targets_[write_pos_] = force;
        write_pos_ = (write_pos_ + 1) % max_size_;
    }
}

void TrainingData::clear() {
    inputs_.clear();
    targets_.clear();
    write_pos_ = 0;
    full_ = false;
}

int TrainingData::size() const {
    return static_cast<int>(inputs_.size());
}

bool TrainingData::isFull() const {
    return full_;
}

} // namespace softflow
