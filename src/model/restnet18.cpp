#include "resnet/model/resnet18.h"
#include "resnet/utils/activation.h"
#include <iostream>
#include <fstream>
#include <algorithm>

ResNet18::ResNet18(int num_classes)
    : conv1_(3, 64, 7, 2, 3),
      bn1_(64),
      relu_(true),
      maxpool_(3, 2, 1),
      avgpool_(1),
      fc_(512, num_classes),
      num_classes_(num_classes),
      training_(false),
      inplanes_(64) {
    
    // Create layers
    layer1_ = make_layer(64, 2, 1);
    layer2_ = make_layer(128, 2, 2);
    layer3_ = make_layer(256, 2, 2);
    layer4_ = make_layer(512, 2, 2);
}

std::vector<std::unique_ptr<BasicBlock>> ResNet18::make_layer(int planes, int blocks, int stride) {
    std::vector<std::unique_ptr<BasicBlock>> layers;
    
    bool downsample = (stride != 1 || inplanes_ != planes);
    layers.push_back(std::make_unique<BasicBlock>(inplanes_, planes, stride, downsample));
    inplanes_ = planes;
    
    for (int i = 1; i < blocks; ++i) {
        layers.push_back(std::make_unique<BasicBlock>(inplanes_, planes));
    }
    
    return layers;
}

Tensor ResNet18::forward(const Tensor& input) {
    // Initial convolution
    Tensor x = conv1_.forward(input);
    x = bn1_.forward(x, training_);
    x = relu_.forward(x);
    x = maxpool_.forward(x);
    
    // ResNet layers
    for (auto& block : layer1_) {
        x = block->forward(x);
    }
    
    for (auto& block : layer2_) {
        x = block->forward(x);
    }
    
    for (auto& block : layer3_) {
        x = block->forward(x);
    }
    
    for (auto& block : layer4_) {
        x = block->forward(x);
    }
    
    // Global average pooling and classifier
    x = avgpool_.forward(x);
    x = fc_.forward(x);
    
    return x;
}

std::vector<float> ResNet18::predict_probabilities(const Tensor& input) {
    set_training(false);
    Tensor logits = forward(input);
    Tensor probs = Activation::softmax(logits);
    
    std::vector<float> result;
    const auto& shape = probs.shape();
    int num_classes = shape[1];
    
    for (int i = 0; i < num_classes; ++i) {
        result.push_back(probs({0, i}));
    }
    
    return result;
}

int ResNet18::predict_class(const Tensor& input) {
    auto probs = predict_probabilities(input);
    return std::max_element(probs.begin(), probs.end()) - probs.begin();
}

void ResNet18::freeze_features() {
    // In a real implementation, this would disable gradient computation
    // for feature layers. For now, it's a placeholder.
    std::cout << "Feature layers frozen for fine-tuning." << std::endl;
}

void ResNet18::unfreeze_all() {
    std::cout << "All layers unfrozen." << std::endl;
}

void ResNet18::replace_classifier(int new_num_classes) {
    fc_ = Linear(512, new_num_classes);
    num_classes_ = new_num_classes;
    std::cout << "Classifier replaced with " << new_num_classes << " classes." << std::endl;
}

void ResNet18::load_pretrained_weights(const std::string& weights_path) {
    std::cout << "Loading pretrained weights from: " << weights_path << std::endl;
    // Placeholder for weight loading implementation
    // In practice, this would load weights from a file format like PyTorch .pth or ONNX
}

void ResNet18::save_weights(const std::string& weights_path) {
    std::cout << "Saving weights to: " << weights_path << std::endl;
    // Placeholder for weight saving implementation
}