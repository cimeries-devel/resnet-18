#ifndef RESNET_RELU_H
#define RESNET_RELU_H

#include "../utils/tensor.h"

class ReLU {
public:
    ReLU(bool inplace = false);
    Tensor forward(const Tensor& input);

private:
    bool inplace_;
};

#endif // RESNET_RELU_H