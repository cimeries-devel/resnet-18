#include "resnet/layers/conv2d.h"
#include <random>
#include <cmath>

Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool bias)
    : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size),
      stride_(stride), padding_(padding), use_bias_(bias),
      weights_(std::vector<int>{out_channels, in_channels, kernel_size, kernel_size}),
      bias_(use_bias_ ? Tensor(std::vector<int>{out_channels}) : Tensor(std::vector<int>{1})) {
    init_weights();
}

void Conv2D::init_weights() {
    // Xavier/Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    float fan_in = in_channels_ * kernel_size_ * kernel_size_;
    float fan_out = out_channels_ * kernel_size_ * kernel_size_;
    float std_dev = std::sqrt(2.0f / (fan_in + fan_out));
    std::normal_distribution<float> dist(0.0f, std_dev);

    float* weight_data = weights_.data();
    for (int i = 0; i < weights_.size(); ++i) {
        weight_data[i] = dist(gen);
    }

    if (use_bias_) {
        bias_.zero();
    }
}

Tensor Conv2D::forward(const Tensor& input) {
    const auto& input_shape = input.shape();
    int batch_size = input_shape[0];
    int input_height = input_shape[2];
    int input_width = input_shape[3];

    // Calculate output dimensions
    int output_height = (input_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    int output_width = (input_width + 2 * padding_ - kernel_size_) / stride_ + 1;

    Tensor output(std::vector<int>{batch_size, out_channels_, output_height, output_width});

    // Apply padding if needed
    Tensor padded_input = (padding_ > 0) ? apply_padding(input) : input;

    // Perform convolution
    for (int b = 0; b < batch_size; ++b) {
        for (int out_ch = 0; out_ch < out_channels_; ++out_ch) {
            for (int out_h = 0; out_h < output_height; ++out_h) {
                for (int out_w = 0; out_w < output_width; ++out_w) {
                    float value = compute_convolution(padded_input, b, out_ch, out_h, out_w);

                    // Add bias if enabled
                    if (use_bias_) {
                        value += bias_({out_ch});
                    }

                    output(b, out_ch, out_h, out_w) = value;
                }
            }
        }
    }

    return output;
}

Tensor Conv2D::apply_padding(const Tensor& input) const {
    const auto& input_shape = input.shape();
    int batch_size = input_shape[0];
    int channels = input_shape[1];
    int height = input_shape[2];
    int width = input_shape[3];

    int padded_height = height + 2 * padding_;
    int padded_width = width + 2 * padding_;

    Tensor padded(batch_size, channels, padded_height, padded_width);
    padded.zero();

    // Copy input to center of padded tensor
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    padded(b, c, h + padding_, w + padding_) = input(b, c, h, w);
                }
            }
        }
    }

    return padded;
}

float Conv2D::compute_convolution(const Tensor& input, int batch, int out_ch, int out_h, int out_w) const {
    float result = 0.0f;

    for (int in_ch = 0; in_ch < in_channels_; ++in_ch) {
        for (int kh = 0; kh < kernel_size_; ++kh) {
            for (int kw = 0; kw < kernel_size_; ++kw) {
                int input_h = out_h * stride_ + kh;
                int input_w = out_w * stride_ + kw;

                result += input(batch, in_ch, input_h, input_w) *
                         weights_(out_ch, in_ch, kh, kw);
            }
        }
    }

    return result;
}