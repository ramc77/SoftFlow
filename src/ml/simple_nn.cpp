#include "simple_nn.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>

namespace softflow {

SimpleNN::SimpleNN(const MLParams& params)
    : input_size_(MLFeatures::SIZE),
      hidden_size_(params.hidden_size),
      learning_rate_(params.learning_rate),
      epochs_(params.training_epochs) {
    std::mt19937 rng(42);
    initializeWeights(rng);
}

void SimpleNN::initializeWeights(std::mt19937& rng) {
    // He initialization: std = sqrt(2/fan_in)
    auto he_init = [&](int fan_in) {
        std::normal_distribution<Real> dist(0.0, std::sqrt(2.0 / fan_in));
        return [&rng, dist]() mutable { return dist(rng); };
    };

    // Layer 1
    auto gen1 = he_init(input_size_);
    W1_.resize(hidden_size_, std::vector<Real>(input_size_));
    b1_.resize(hidden_size_, 0.0);
    for (auto& row : W1_) for (auto& w : row) w = gen1();

    // Layer 2
    auto gen2 = he_init(hidden_size_);
    W2_.resize(hidden_size_, std::vector<Real>(hidden_size_));
    b2_.resize(hidden_size_, 0.0);
    for (auto& row : W2_) for (auto& w : row) w = gen2();

    // Layer 3
    auto gen3 = he_init(hidden_size_);
    W3_.resize(output_size_, std::vector<Real>(hidden_size_));
    b3_.resize(output_size_, 0.0);
    for (auto& row : W3_) for (auto& w : row) w = gen3();
}

std::vector<Real> SimpleNN::forward(const std::vector<Real>& input,
                                     std::vector<Real>& h1,
                                     std::vector<Real>& h2) const {
    // Layer 1: h1 = ReLU(W1 * input + b1)
    h1.resize(hidden_size_);
    for (int i = 0; i < hidden_size_; ++i) {
        Real sum = b1_[i];
        for (int j = 0; j < input_size_; ++j) {
            sum += W1_[i][j] * input[j];
        }
        h1[i] = std::max(Real(0), sum);  // ReLU
    }

    // Layer 2: h2 = ReLU(W2 * h1 + b2)
    h2.resize(hidden_size_);
    for (int i = 0; i < hidden_size_; ++i) {
        Real sum = b2_[i];
        for (int j = 0; j < hidden_size_; ++j) {
            sum += W2_[i][j] * h1[j];
        }
        h2[i] = std::max(Real(0), sum);
    }

    // Layer 3: output = W3 * h2 + b3 (linear)
    std::vector<Real> output(output_size_);
    for (int i = 0; i < output_size_; ++i) {
        Real sum = b3_[i];
        for (int j = 0; j < hidden_size_; ++j) {
            sum += W3_[i][j] * h2[j];
        }
        output[i] = sum;
    }

    return output;
}

Vec2d SimpleNN::predictForce(const MLFeatures& features) {
    auto input = features.toVector();
    if (has_normalization_) {
        input = normalize(input, input_mean_, input_std_);
    }
    std::vector<Real> h1, h2;
    auto output = forward(input, h1, h2);

    // Denormalize output
    Real fx = output[0], fy = output[1];
    if (has_normalization_) {
        fx = fx * output_std_[0] + output_mean_[0];
        fy = fy * output_std_[1] + output_mean_[1];
    }
    return {fx, fy};
}

void SimpleNN::computeNormalization(const std::vector<MLFeatures>& inputs,
                                     const std::vector<Vec2d>& targets) {
    int N = static_cast<int>(inputs.size());
    if (N == 0) return;

    input_mean_.assign(input_size_, 0.0);
    input_std_.assign(input_size_, 0.0);
    output_mean_ = {0.0, 0.0};
    output_std_ = {0.0, 0.0};

    for (int n = 0; n < N; ++n) {
        auto v = inputs[n].toVector();
        for (int j = 0; j < input_size_; ++j) input_mean_[j] += v[j];
        output_mean_[0] += targets[n].x;
        output_mean_[1] += targets[n].y;
    }
    for (int j = 0; j < input_size_; ++j) input_mean_[j] /= N;
    output_mean_[0] /= N;
    output_mean_[1] /= N;

    for (int n = 0; n < N; ++n) {
        auto v = inputs[n].toVector();
        for (int j = 0; j < input_size_; ++j) {
            Real d = v[j] - input_mean_[j];
            input_std_[j] += d * d;
        }
        Real d0 = targets[n].x - output_mean_[0];
        Real d1 = targets[n].y - output_mean_[1];
        output_std_[0] += d0 * d0;
        output_std_[1] += d1 * d1;
    }
    for (int j = 0; j < input_size_; ++j) {
        input_std_[j] = std::sqrt(input_std_[j] / N);
        if (input_std_[j] < 1e-10) input_std_[j] = 1.0;
    }
    output_std_[0] = std::sqrt(output_std_[0] / N);
    output_std_[1] = std::sqrt(output_std_[1] / N);
    if (output_std_[0] < 1e-10) output_std_[0] = 1.0;
    if (output_std_[1] < 1e-10) output_std_[1] = 1.0;

    has_normalization_ = true;
}

std::vector<Real> SimpleNN::normalize(const std::vector<Real>& v,
                                       const std::vector<Real>& mean,
                                       const std::vector<Real>& std_dev) const {
    std::vector<Real> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = (v[i] - mean[i]) / std_dev[i];
    }
    return out;
}

void SimpleNN::train(const std::vector<MLFeatures>& inputs,
                      const std::vector<Vec2d>& targets) {
    int N = static_cast<int>(inputs.size());
    if (N < 10) return;

    computeNormalization(inputs, targets);

    // Prepare normalized data
    std::vector<std::vector<Real>> X(N);
    std::vector<std::vector<Real>> Y(N);
    for (int n = 0; n < N; ++n) {
        X[n] = normalize(inputs[n].toVector(), input_mean_, input_std_);
        Y[n] = {(targets[n].x - output_mean_[0]) / output_std_[0],
                (targets[n].y - output_mean_[1]) / output_std_[1]};
    }

    // Mini-batch SGD with backpropagation
    int batch_size = std::min(64, N);
    std::mt19937 rng(123);

    for (int epoch = 0; epoch < epochs_; ++epoch) {
        // Shuffle indices
        std::vector<int> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        Real total_loss = 0.0;

        for (int b = 0; b < N; b += batch_size) {
            int bsize = std::min(batch_size, N - b);

            // Accumulate gradients
            auto dW1 = W1_; for (auto& r : dW1) std::fill(r.begin(), r.end(), 0);
            auto dW2 = W2_; for (auto& r : dW2) std::fill(r.begin(), r.end(), 0);
            auto dW3 = W3_; for (auto& r : dW3) std::fill(r.begin(), r.end(), 0);
            std::vector<Real> db1(hidden_size_, 0), db2(hidden_size_, 0), db3(output_size_, 0);

            for (int s = 0; s < bsize; ++s) {
                int idx = indices[b + s];

                // Forward pass
                std::vector<Real> h1, h2;
                auto out = forward(X[idx], h1, h2);

                // Loss = MSE
                std::vector<Real> dout(output_size_);
                for (int i = 0; i < output_size_; ++i) {
                    dout[i] = 2.0 * (out[i] - Y[idx][i]) / output_size_;
                    total_loss += (out[i] - Y[idx][i]) * (out[i] - Y[idx][i]);
                }

                // Backprop layer 3
                std::vector<Real> dh2(hidden_size_, 0);
                for (int i = 0; i < output_size_; ++i) {
                    for (int j = 0; j < hidden_size_; ++j) {
                        dW3[i][j] += dout[i] * h2[j];
                        dh2[j] += W3_[i][j] * dout[i];
                    }
                    db3[i] += dout[i];
                }

                // ReLU derivative for h2
                for (int i = 0; i < hidden_size_; ++i) {
                    if (h2[i] <= 0) dh2[i] = 0;
                }

                // Backprop layer 2
                std::vector<Real> dh1(hidden_size_, 0);
                for (int i = 0; i < hidden_size_; ++i) {
                    for (int j = 0; j < hidden_size_; ++j) {
                        dW2[i][j] += dh2[i] * h1[j];
                        dh1[j] += W2_[i][j] * dh2[i];
                    }
                    db2[i] += dh2[i];
                }

                // ReLU derivative for h1
                for (int i = 0; i < hidden_size_; ++i) {
                    if (h1[i] <= 0) dh1[i] = 0;
                }

                // Backprop layer 1
                for (int i = 0; i < hidden_size_; ++i) {
                    for (int j = 0; j < input_size_; ++j) {
                        dW1[i][j] += dh1[i] * X[idx][j];
                    }
                    db1[i] += dh1[i];
                }
            }

            // Update weights
            Real lr = learning_rate_ / bsize;
            for (int i = 0; i < hidden_size_; ++i) {
                for (int j = 0; j < input_size_; ++j) W1_[i][j] -= lr * dW1[i][j];
                b1_[i] -= lr * db1[i];
            }
            for (int i = 0; i < hidden_size_; ++i) {
                for (int j = 0; j < hidden_size_; ++j) W2_[i][j] -= lr * dW2[i][j];
                b2_[i] -= lr * db2[i];
            }
            for (int i = 0; i < output_size_; ++i) {
                for (int j = 0; j < hidden_size_; ++j) W3_[i][j] -= lr * dW3[i][j];
                b3_[i] -= lr * db3[i];
            }
        }

        if (epoch % 10 == 0) {
            std::cout << "  ML epoch " << epoch << " loss: " << total_loss / N << std::endl;
        }
    }

    trained_ = true;
    std::cout << "ML surrogate trained on " << N << " samples" << std::endl;
}

void SimpleNN::saveWeights(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    auto writeVec = [&](const std::vector<Real>& v) {
        int sz = static_cast<int>(v.size());
        f.write(reinterpret_cast<const char*>(&sz), sizeof(int));
        f.write(reinterpret_cast<const char*>(v.data()), sz * sizeof(Real));
    };
    auto writeMat = [&](const std::vector<std::vector<Real>>& m) {
        int rows = static_cast<int>(m.size());
        f.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        for (const auto& row : m) writeVec(row);
    };
    writeMat(W1_); writeVec(b1_);
    writeMat(W2_); writeVec(b2_);
    writeMat(W3_); writeVec(b3_);
    writeVec(input_mean_); writeVec(input_std_);
    writeVec(output_mean_); writeVec(output_std_);
}

void SimpleNN::loadWeights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return;
    auto readVec = [&](std::vector<Real>& v) {
        int sz; f.read(reinterpret_cast<char*>(&sz), sizeof(int));
        v.resize(sz);
        f.read(reinterpret_cast<char*>(v.data()), sz * sizeof(Real));
    };
    auto readMat = [&](std::vector<std::vector<Real>>& m) {
        int rows; f.read(reinterpret_cast<char*>(&rows), sizeof(int));
        m.resize(rows);
        for (auto& row : m) readVec(row);
    };
    readMat(W1_); readVec(b1_);
    readMat(W2_); readVec(b2_);
    readMat(W3_); readVec(b3_);
    readVec(input_mean_); readVec(input_std_);
    readVec(output_mean_); readVec(output_std_);
    has_normalization_ = true;
    trained_ = true;
}

} // namespace softflow
