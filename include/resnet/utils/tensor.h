#ifndef RESNET_TENSOR_H
#define RESNET_TENSOR_H

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

class Tensor {
public:
    Tensor();
    Tensor(const std::vector<int>& shape);
    Tensor(const cv::Mat& image);
    Tensor(int batch_size, int channels, int height, int width);
    
    // Copy constructor and assignment
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    
    // Move constructor and assignment
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    ~Tensor() = default;
    
    // Accessors
    float& operator()(int b, int c, int h, int w);
    const float& operator()(int b, int c, int h, int w) const;
    
    float& operator()(const std::vector<int>& indices);
    const float& operator()(const std::vector<int>& indices) const;
    
    // Properties
    const std::vector<int>& shape() const { return shape_; }
    int size() const { return size_; }
    float* data() { return data_.get(); }
    const float* data() const { return data_.get(); }
    
    // Operations
    void zero();
    void fill(float value);
    Tensor clone() const;
    void reshape(const std::vector<int>& new_shape);
    
    // Mathematical operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor& operator+=(const Tensor& other);
    
    // Utility functions
    void print_shape() const;
    cv::Mat to_image() const;

private:
    std::vector<int> shape_;
    std::vector<int> strides_;
    int size_;
    std::unique_ptr<float[]> data_;
    
    void compute_strides();
    int compute_index(const std::vector<int>& indices) const;
};

#endif // RESNET_TENSOR_H