#ifndef RESNET_BASIC_BLOCK_H
#define RESNET_BASIC_BLOCK_H

#include "../layers/conv2d.h"
#include "../layers/batch_norm.h"
#include "../layers/relu.h"
#include "../utils/tensor.h"

class BasicBlock {
public:
    BasicBlock(int inplanes, int planes, int stride = 1, bool downsample = false);
    
    Tensor forward(const Tensor& input);
    
    // For loading pretrained weights
    Conv2D& get_conv1() { return conv1_; }
    BatchNorm2D& get_bn1() { return bn1_; }
    Conv2D& get_conv2() { return conv2_; }
    BatchNorm2D& get_bn2() { return bn2_; }
    Conv2D& get_downsample_conv() { return downsample_conv_; }
    BatchNorm2D& get_downsample_bn() { return downsample_bn_; }

private:
    Conv2D conv1_;
    BatchNorm2D bn1_;
    ReLU relu_;
    Conv2D conv2_;
    BatchNorm2D bn2_;
    
    bool has_downsample_;
    Conv2D downsample_conv_;
    BatchNorm2D downsample_bn_;
    
    int stride_;
};

#endif // RESNET_BASIC_BLOCK_H