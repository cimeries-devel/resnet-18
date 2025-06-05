#include "resnet/layers/pooling.h"
#include <algorithm>
#include <limits>

MaxPool2D::MaxPool2D(int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size), stride_(stride == -1 ? kernel_size : stride), padding_(padding) {}

Tensor MaxPool2D::forward(const Tensor& input) {
    const auto& input_shape = input.shape();
    int batch_size = input_shape[0];
    int channels = input_shape[1];
    int input_height = input_shape[2];
    int input_width = input_shape[3];
    
    int output_height = (input_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    int output_width = (input_width + 2 * padding_ - kernel_size_) / stride_ + 1;
    
    Tensor output(batch_size, channels, output_height, output_width);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int out_h = 0; out_h < output_height; ++out_h) {
                for (int out_w = 0; out_w < output_width; ++out_w) {
                    float max_val = std::numeric_limits<float>::lowest();
                    
                    for (int kh = 0; kh < kernel_size_; ++kh) {
                        for (int kw = 0; kw < kernel_size_; ++kw) {
                            int input_h = out_h * stride_ + kh - padding_;
                            int input_w = out_w * stride_ + kw - padding_;
                            
                            if (input_h >= 0 && input_h < input_height &&
                                input_w >= 0 && input_w < input_width) {
                                max_val = std::max(max_val, input(b, c, input_h, input_w));
                            }
                        }
                    }
                    
                    output(b, c, out_h, out_w) = max_val;
                }
            }
        }
    }
    
    return output;
}

AdaptiveAvgPool2D::AdaptiveAvgPool2D(int output_size) : output_size_(output_size) {}

Tensor AdaptiveAvgPool2D::forward(const Tensor& input) {
    const auto& input_shape = input.shape();
    int batch_size = input_shape[0];
    int channels = input_shape[1];
    int input_height = input_shape[2];
    int input_width = input_shape[3];
    
    Tensor output(batch_size, channels, output_size_, output_size_);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int out_h = 0; out_h < output_size_; ++out_h) {
                for (int out_w = 0; out_w < output_size_; ++out_w) {
                    // Calculate the input region for this output pixel
                    int start_h = (out_h * input_height) / output_size_;
                    int end_h = ((out_h + 1) * input_height) / output_size_;
                    int start_w = (out_w * input_width) / output_size_;
                    int end_w = ((out_w + 1) * input_width) / output_size_;
                    
                    float sum = 0.0f;
                    int count = 0;
                    
                    for (int h = start_h; h < end_h; ++h) {
                        for (int w = start_w; w < end_w; ++w) {
                            sum += input(b, c, h, w);
                            count++;
                        }
                    }
                    
                    output(b, c, out_h, out_w) = sum / count;
                }
            }
        }
    }
    
    return output;
}