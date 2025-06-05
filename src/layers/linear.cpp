#include "resnet/layers/linear.h"
#include <random>
#include <cmath>

Linear::Linear(int in_features, int out_features, bool use_bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(use_bias),
      weights_(std::vector<int>{out_features, in_features}),
      bias_(use_bias_ ? Tensor(std::vector<int>{out_features}) : Tensor(std::vector<int>{})) {
    init_parameters();
}

void Linear::init_parameters() {
    // Xavier/Glorot initialization
    float std = std::sqrt(2.0f / (in_features_ + out_features_));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std);
    
    // Initialize weights
    auto weight_data = weights_.data();
    int weight_size = weights_.size();
    for (int i = 0; i < weight_size; ++i) {
        weight_data[i] = dist(gen);
    }
    
    // Initialize bias to zero if used
    if (use_bias_) {
        bias_.zero();
    }
}

Tensor Linear::forward(const Tensor& input) {
    const auto& input_shape = input.shape();
    int batch_size = input_shape[0];
    
    Tensor output(std::vector<int>{batch_size, out_features_});
    output.zero();
    
    // Perform matrix multiplication: output = input * weight^T + bias
    for (int b = 0; b < batch_size; ++b) {
        for (int out_f = 0; out_f < out_features_; ++out_f) {
            float sum = 0.0f;
            for (int in_f = 0; in_f < in_features_; ++in_f) {
                sum += input({b, in_f}) * weights_({out_f, in_f});
            }
            output({b, out_f}) = sum;
            
            // Add bias if used
            if (use_bias_) {
                output({b, out_f}) += bias_({out_f});
            }
        }
    }
    
    return output;
}