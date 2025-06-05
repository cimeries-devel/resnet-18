#ifndef RESNET_LINEAR_H
#define RESNET_LINEAR_H

#include "../utils/tensor.h"

class Linear {
public:
    Linear(int in_features, int out_features, bool bias = true);
    
    Tensor forward(const Tensor& input);
    void init_parameters();
    void init_weights();
    
    // Getters
    const Tensor& get_weights() const { return weights_; }
    const Tensor& get_bias() const { return bias_; }
    
    // Setters for loading pretrained weights
    void set_weights(const Tensor& weights) { weights_ = weights; }
    void set_bias(const Tensor& bias) { bias_ = bias; }

private:
    int in_features_;
    int out_features_;
    bool use_bias_;
    
    Tensor weights_;
    Tensor bias_;
};

#endif // RESNET_LINEAR_H