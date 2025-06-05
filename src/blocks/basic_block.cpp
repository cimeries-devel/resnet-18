#include "resnet/blocks/basic_block.h"

BasicBlock::BasicBlock(int inplanes, int planes, int stride, bool downsample)
    : conv1_(inplanes, planes, 3, stride, 1),
      bn1_(planes),
      relu_(true),
      conv2_(planes, planes, 3, 1, 1),
      bn2_(planes),
      has_downsample_(downsample),
      downsample_conv_(downsample ? Conv2D(inplanes, planes, 1, stride, 0) : Conv2D(1, 1, 1)),
      downsample_bn_(downsample ? BatchNorm2D(planes) : BatchNorm2D(1)),
      stride_(stride) {}

Tensor BasicBlock::forward(const Tensor& input) {
    Tensor identity = input.clone();
    
    // First convolution block
    Tensor out = conv1_.forward(input);
    out = bn1_.forward(out);
    out = relu_.forward(out);
    
    // Second convolution block
    out = conv2_.forward(out);
    out = bn2_.forward(out);
    
    // Downsample identity if needed
    if (has_downsample_) {
        identity = downsample_conv_.forward(identity);
        identity = downsample_bn_.forward(identity);
    }
    
    // Add residual connection
    out += identity;
    out = relu_.forward(out);
    
    return out;
}