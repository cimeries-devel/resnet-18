#include "resnet/layers/batch_norm.h"
#include <cmath>

BatchNorm2D::BatchNorm2D(int num_features, float eps, float momentum)
    : num_features_(num_features), eps_(eps), momentum_(momentum),
      weight_(std::vector<int>{num_features}), bias_(std::vector<int>{num_features}),
      running_mean_(std::vector<int>{num_features}), running_var_(std::vector<int>{num_features}) {
    init_parameters();
}

void BatchNorm2D::init_parameters() {
    weight_.fill(1.0f);
    bias_.zero();
    running_mean_.zero();
    running_var_.fill(1.0f);
}

Tensor BatchNorm2D::forward(const Tensor& input, bool training) {
    const auto& input_shape = input.shape();
    int batch_size = input_shape[0];
    int channels = input_shape[1];
    int height = input_shape[2];
    int width = input_shape[3];
    
    Tensor output = input.clone();
    
    if (training) {
        // Compute batch statistics
        Tensor batch_mean(std::vector<int>{channels});
        Tensor batch_var(std::vector<int>{channels});
        batch_mean.zero();
        batch_var.zero();
        
        // Calculate mean
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        sum += input(b, c, h, w);
                    }
                }
            }
            batch_mean({c}) = sum / (batch_size * height * width);
        }
        
        // Calculate variance
        for (int c = 0; c < channels; ++c) {
            float sum_sq_diff = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        float diff = input(b, c, h, w) - batch_mean({c});
                        sum_sq_diff += diff * diff;
                    }
                }
            }
            batch_var({c}) = sum_sq_diff / (batch_size * height * width);
        }
        
        // Update running statistics
        for (int c = 0; c < channels; ++c) {
            running_mean_({c}) = (1.0f - momentum_) * running_mean_({c}) + 
                                 momentum_ * batch_mean({c});
            running_var_({c}) = (1.0f - momentum_) * running_var_({c}) + 
                               momentum_ * batch_var({c});
        }
        
        // Normalize using batch statistics
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        float normalized = (input(b, c, h, w) - batch_mean({c})) / 
                                         std::sqrt(batch_var({c}) + eps_);
                        output(b, c, h, w) = weight_({c}) * normalized + bias_({c});
                    }
                }
            }
        }
    } else {
        // Use running statistics for inference
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        float normalized = (input(b, c, h, w) - running_mean_({c})) / 
                                         std::sqrt(running_var_({c}) + eps_);
                        output(b, c, h, w) = weight_({c}) * normalized + bias_({c});
                    }
                }
            }
        }
    }
    
    return output;
}