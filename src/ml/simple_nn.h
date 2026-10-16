#pragma once
#include "surrogate_model.h"
#include "../core/parameters.h"
#include <vector>
#include <random>

namespace softflow {

// Simple feedforward neural network (no external dependencies)
// Architecture: input -> hidden1 -> hidden2 -> output
// Activation: ReLU for hidden layers, linear for output
// Training: SGD with backpropagation
class SimpleNN : public SurrogateModel {
public:
    explicit SimpleNN(const MLParams& params);

    Vec2d predictForce(const MLFeatures& features) override;
    void train(const std::vector<MLFeatures>& inputs,
               const std::vector<Vec2d>& targets) override;
    bool isReady() const override { return trained_; }

    void saveWeights(const std::string& path) const;
    void loadWeights(const std::string& path);

private:
    int input_size_;
    int hidden_size_;
    int output_size_ = 2;
    Real learning_rate_;
    int epochs_;
    bool trained_ = false;

    // Weights and biases
    // Layer 1: input -> hidden1
    std::vector<std::vector<Real>> W1_;  // [hidden_size x input_size]
    std::vector<Real> b1_;               // [hidden_size]
    // Layer 2: hidden1 -> hidden2
    std::vector<std::vector<Real>> W2_;  // [hidden_size x hidden_size]
    std::vector<Real> b2_;               // [hidden_size]
    // Layer 3: hidden2 -> output
    std::vector<std::vector<Real>> W3_;  // [output_size x hidden_size]
    std::vector<Real> b3_;               // [output_size]

    // Normalization parameters
    std::vector<Real> input_mean_, input_std_;
    std::vector<Real> output_mean_, output_std_;
    bool has_normalization_ = false;

    void initializeWeights(std::mt19937& rng);
    std::vector<Real> forward(const std::vector<Real>& input,
                              std::vector<Real>& h1, std::vector<Real>& h2) const;
    void computeNormalization(const std::vector<MLFeatures>& inputs,
                              const std::vector<Vec2d>& targets);
    std::vector<Real> normalize(const std::vector<Real>& v,
                                const std::vector<Real>& mean,
                                const std::vector<Real>& std_dev) const;
};

} // namespace softflow
