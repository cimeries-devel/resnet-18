#ifndef RESNET_CONV2D_H
#define RESNET_CONV2D_H

#include "../utils/tensor.h"

class Conv2D {
public:
    Conv2D(int in_channels, int out_channels, int kernel_size, 
           int stride = 1, int padding = 0, bool bias = true);
    
    Tensor forward(const Tensor& input);
    void init_weights();
    
    // Getters
    const Tensor& get_weights() const { return weights_; }
    const Tensor& get_bias() const { return bias_; }
    
    // Setters for loading pretrained weights
    void set_weights(const Tensor& weights) { weights_ = weights; }
    void set_bias(const Tensor& bias) { bias_ = bias; }

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    bool use_bias_;
    
    Tensor weights_;
    Tensor bias_;
    
    Tensor apply_padding(const Tensor& input) const;
    float compute_convolution(const Tensor& input, int batch, int out_ch, int out_h, int out_w) const;
};

#endif // RESNET_CONV2D_H