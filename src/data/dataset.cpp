#include "resnet/data/dataset.h"
#include <experimental/filesystem>
#include <iostream>
#include <algorithm>
#include <random>

namespace fs = std::experimental::filesystem;

AnimalDataset::AnimalDataset(const std::string& data_root, const std::string& split)
    : data_root_(data_root), split_(split), image_width_(224), image_height_(224),
      normalize_(true), augment_(false) {
    
    discover_classes();
    load_dataset();
    
    std::cout << "Loaded " << samples_.size() << " samples from " << split_ << " split." << std::endl;
}

void AnimalDataset::discover_classes() {
    std::string split_path = data_root_ + "/" + split_;
    
    if (!fs::exists(split_path)) {
        throw std::runtime_error("Dataset path does not exist: " + split_path);
    }
    
    for (const auto& entry : fs::directory_iterator(split_path)) {
        if (is_directory(entry)) {
            class_names_.push_back(entry.path().filename().string());
        }
    }
    
    std::sort(class_names_.begin(), class_names_.end());
    
    for (size_t i = 0; i < class_names_.size(); ++i) {
        class_to_idx_[class_names_[i]] = i;
    }
    
    std::cout << "Found " << class_names_.size() << " classes: ";
    for (const auto& name : class_names_) {
        std::cout << name << " ";
    }
    std::cout << std::endl;
}

void AnimalDataset::load_dataset() {
    std::string split_path = data_root_ + "/" + split_;
    
    for (const auto& class_name : class_names_) {
        std::string class_path = split_path + "/" + class_name;
        int class_idx = class_to_idx_[class_name];
        
        for (const auto& entry : fs::directory_iterator(class_path)) {
            if (std::experimental::filesystem::is_regular_file(entry)) {
                std::string file_path = entry.path().string();
                std::string extension = entry.path().extension().string();
                
                // Check if it's an image file
                if (extension == ".jpg" || extension == ".jpeg" || 
                    extension == ".png" || extension == ".bmp") {
                    
                    // Load and preprocess image
                    cv::Mat image = cv::imread(file_path);
                    if (!image.empty()) {
                        Tensor tensor_image = preprocess_image(image);
                        
                        Sample sample{
                        std::move(tensor_image), class_idx, class_name, file_path};
                        
                        samples_.push_back(std::move(sample));
                    }
                }
            }
        }
    }
}

Tensor AnimalDataset::preprocess_image(const cv::Mat& image) const {
    cv::Mat processed = image.clone();
    
    // Resize to target size
    cv::resize(processed, processed, cv::Size(image_width_, image_height_));
    
    // Convert to tensor
    Tensor tensor_image(processed);
    
    // Apply augmentation if enabled and in training mode
    if (augment_ && split_ == "train") {
        tensor_image = apply_augmentation(tensor_image);
    }
    
    // Normalize if enabled
    if (normalize_) {
        tensor_image = normalize_image(tensor_image);
    }
    
    return tensor_image;
}

Tensor AnimalDataset::apply_augmentation(const Tensor& image) const {
    // Simple augmentation: random horizontal flip
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    Tensor augmented = image.clone();
    
    // Random horizontal flip with 50% probability
    if (dis(gen) > 0.5) {
        const auto& shape = image.shape();
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];
        
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    augmented(0, c, h, w) = image(0, c, h, width - 1 - w);
                }
            }
        }
    }
    
    return augmented;
}

Tensor AnimalDataset::normalize_image(const Tensor& image) const {
    // ImageNet normalization
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};
    
    Tensor normalized = image.clone();
    const auto& shape = image.shape();
    int channels = shape[1];
    int height = shape[2];
    int width = shape[3];
    
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                normalized(0, c, h, w) = (image(0, c, h, w) - mean[c]) / std[c];
            }
        }
    }
    
    return normalized;
}

Sample AnimalDataset::get_sample(size_t index) const {
    if (index >= samples_.size()) {
        throw std::out_of_range("Sample index out of range");
    }
    return samples_[index];
}

void AnimalDataset::set_image_size(int width, int height) {
    image_width_ = width;
    image_height_ = height;
}

void AnimalDataset::print_dataset_info() const {
    std::cout << "\n=== Dataset Information ===" << std::endl;
    std::cout << "Split: " << split_ << std::endl;
    std::cout << "Total samples: " << samples_.size() << std::endl;
    std::cout << "Number of classes: " << class_names_.size() << std::endl;
    std::cout << "Image size: " << image_width_ << "x" << image_height_ << std::endl;
    std::cout << "Normalization: " << (normalize_ ? "enabled" : "disabled") << std::endl;
    std::cout << "Augmentation: " << (augment_ ? "enabled" : "disabled") << std::endl;
    
    auto class_dist = get_class_distribution();
    std::cout << "\nClass distribution:" << std::endl;
    for (const auto& pair : class_dist) {
        std::cout << "  " << pair.first << ": " << pair.second << " samples" << std::endl;
    }
}

std::map<std::string, int> AnimalDataset::get_class_distribution() const {
    std::map<std::string, int> distribution;
    
    for (const auto& sample : samples_) {
        distribution[sample.class_name]++;
    }
    
    return distribution;
}