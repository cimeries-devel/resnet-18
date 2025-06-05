#ifndef RESNET_ACTIVATION_H
#define RESNET_ACTIVATION_H

#include "tensor.h"

class Activation {
public:
    static Tensor relu(const Tensor& input);
    static Tensor relu_inplace(Tensor& input);
    static Tensor softmax(const Tensor& input, int dim = -1);
    static Tensor sigmoid(const Tensor& input);
    static Tensor tanh(const Tensor& input);
};

#endif // RESNET_ACTIVATION_H