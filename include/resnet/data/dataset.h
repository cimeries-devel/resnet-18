#ifndef RESNET_DATASET_H
#define RESNET_DATASET_H

#include "../utils/tensor.h"
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>

struct Sample {
    Tensor image;
    int label;
    std::string class_name;
    std::string file_path;
};

class AnimalDataset {
public:
    AnimalDataset(const std::string& data_root, const std::string& split = "train");
    
    size_t size() const { return samples_.size(); }
    Sample get_sample(size_t index) const;
    
    const std::vector<std::string>& get_class_names() const { return class_names_; }
    int get_num_classes() const { return class_names_.size(); }
    
    // Data preprocessing
    void set_image_size(int width, int height);
    void set_normalize(bool normalize) { normalize_ = normalize; }
    void set_augmentation(bool augment) { augment_ = augment; }
    
    // Statistics
    void print_dataset_info() const;
    std::map<std::string, int> get_class_distribution() const;

private:
    std::vector<Sample> samples_;
    std::vector<std::string> class_names_;
    std::map<std::string, int> class_to_idx_;
    
    std::string data_root_;
    std::string split_;
    
    int image_width_;
    int image_height_;
    bool normalize_;
    bool augment_;
    
    void load_dataset();
    void discover_classes();
    Tensor preprocess_image(const cv::Mat& image) const;
    Tensor apply_augmentation(const Tensor& image) const;
    Tensor normalize_image(const Tensor& image) const;
};

#endif // RESNET_DATASET_H