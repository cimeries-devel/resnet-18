#include "resnet/utils/activation.h"
#include <algorithm>
#include <cmath>
#include <numeric>

Tensor Activation::relu(const Tensor& input) {
    Tensor output = input.clone();
    float* data = output.data();
    int size = output.size();
    
    for (int i = 0; i < size; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
    
    return output;
}

Tensor Activation::relu_inplace(Tensor& input) {
    float* data = input.data();
    int size = input.size();
    
    for (int i = 0; i < size; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
    
    return input;
}

Tensor Activation::softmax(const Tensor& input, int dim) {
    Tensor output = input.clone();
    const auto& shape = input.shape();
    
    if (dim == -1) {
        dim = shape.size() - 1;
    }
    
    // For simplicity, implement softmax for 2D tensors (batch_size, num_classes)
    if (shape.size() == 2 && dim == 1) {
        int batch_size = shape[0];
        int num_classes = shape[1];
        
        for (int b = 0; b < batch_size; ++b) {
            // Find max for numerical stability
            float max_val = output({b, 0});
            for (int c = 1; c < num_classes; ++c) {
                max_val = std::max(max_val, output({b, c}));
            }
            
            // Compute exp and sum
            float sum = 0.0f;
            for (int c = 0; c < num_classes; ++c) {
                output({b, c}) = std::exp(output({b, c}) - max_val);
                sum += output({b, c});
            }
            
            // Normalize
            for (int c = 0; c < num_classes; ++c) {
                output({b, c}) /= sum;
            }
        }
    }
    
    return output;
}

Tensor Activation::sigmoid(const Tensor& input) {
    Tensor output = input.clone();
    float* data = output.data();
    int size = output.size();
    
    for (int i = 0; i < size; ++i) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
    
    return output;
}

Tensor Activation::tanh(const Tensor& input) {
    Tensor output = input.clone();
    float* data = output.data();
    int size = output.size();
    
    for (int i = 0; i < size; ++i) {
        data[i] = std::tanh(data[i]);
    }
    
    return output;
}