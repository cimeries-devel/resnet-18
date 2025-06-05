#ifndef RESNET_POOLING_H
#define RESNET_POOLING_H

#include "../utils/tensor.h"

class MaxPool2D {
public:
    MaxPool2D(int kernel_size, int stride = -1, int padding = 0);
    Tensor forward(const Tensor& input);

private:
    int kernel_size_;
    int stride_;
    int padding_;
};

class AdaptiveAvgPool2D {
public:
    AdaptiveAvgPool2D(int output_size);
    Tensor forward(const Tensor& input);

private:
    int output_size_;
};

#endif // RESNET_POOLING_H