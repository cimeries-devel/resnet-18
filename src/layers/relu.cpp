#include "resnet/layers/relu.h"
#include "resnet/utils/activation.h"

ReLU::ReLU(bool inplace) : inplace_(inplace) {}

Tensor ReLU::forward(const Tensor& input) {
    if (inplace_) {
        Tensor mutable_input = input.clone();
        return Activation::relu_inplace(mutable_input);
    } else {
        return Activation::relu(input);
    }
}