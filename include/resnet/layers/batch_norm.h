#ifndef RESNET_BATCH_NORM_H
#define RESNET_BATCH_NORM_H

#include "../utils/tensor.h"

class BatchNorm2D {
public:
    BatchNorm2D(int num_features, float eps = 1e-5, float momentum = 0.1);
    
    Tensor forward(const Tensor& input, bool training = false);
    void init_parameters();
    
    // Getters
    const Tensor& get_weight() const { return weight_; }
    const Tensor& get_bias() const { return bias_; }
    const Tensor& get_running_mean() const { return running_mean_; }
    const Tensor& get_running_var() const { return running_var_; }
    
    // Setters for loading pretrained weights
    void set_weight(const Tensor& weight) { weight_ = weight; }
    void set_bias(const Tensor& bias) { bias_ = bias; }
    void set_running_mean(const Tensor& mean) { running_mean_ = mean; }
    void set_running_var(const Tensor& var) { running_var_ = var; }

private:
    int num_features_;
    float eps_;
    float momentum_;
    
    Tensor weight_;
    Tensor bias_;
    Tensor running_mean_;
    Tensor running_var_;
};

#endif // RESNET_BATCH_NORM_H