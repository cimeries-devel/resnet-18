#ifndef RESNET_RESNET18_H
#define RESNET_RESNET18_H

#include "../blocks/basic_block.h"
#include "../layers/conv2d.h"
#include "../layers/batch_norm.h"
#include "../layers/relu.h"
#include "../layers/pooling.h"
#include "../layers/linear.h"
#include "../utils/tensor.h"
#include <vector>
#include <memory>

class ResNet18 {
public:
    ResNet18(int num_classes = 1000);
    
    Tensor forward(const Tensor& input);
    void load_pretrained_weights(const std::string& weights_path);
    void save_weights(const std::string& weights_path);
    
    // For fine-tuning
    void freeze_features();
    void unfreeze_all();
    void replace_classifier(int new_num_classes);
    
    // Inference
    std::vector<float> predict_probabilities(const Tensor& input);
    int predict_class(const Tensor& input);
    
    void set_training(bool training) { training_ = training; }
    bool is_training() const { return training_; }

private:
    Conv2D conv1_;
    BatchNorm2D bn1_;
    ReLU relu_;
    MaxPool2D maxpool_;
    
    std::vector<std::unique_ptr<BasicBlock>> layer1_;
    std::vector<std::unique_ptr<BasicBlock>> layer2_;
    std::vector<std::unique_ptr<BasicBlock>> layer3_;
    std::vector<std::unique_ptr<BasicBlock>> layer4_;
    
    AdaptiveAvgPool2D avgpool_;
    Linear fc_;
    
    int num_classes_;
    bool training_;
    
    std::vector<std::unique_ptr<BasicBlock>> make_layer(int planes, int blocks, int stride = 1);
    int inplanes_;
};

#endif // RESNET_RESNET18_H