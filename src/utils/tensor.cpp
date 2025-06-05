#include "resnet/utils/tensor.h"
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cstring>
#include <cmath>

Tensor::Tensor() : shape_({}), size_(0), data_(nullptr) {
    // Default constructor creates an empty tensor
}


Tensor::Tensor(const std::vector<int>& shape) : shape_(shape) {
    size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    data_ = std::make_unique<float[]>(size_);
    compute_strides();
    zero();
}

Tensor::Tensor(const cv::Mat& image) {
    if (image.channels() == 3) {
        shape_ = {1, 3, image.rows, image.cols};
    } else if (image.channels() == 1) {
        shape_ = {1, 1, image.rows, image.cols};
    } else {
        throw std::runtime_error("Unsupported number of channels");
    }
    
    size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    data_ = std::make_unique<float[]>(size_);
    compute_strides();
    
    // Convert CV_8UC3 to float tensor with CHW layout
    cv::Mat float_image;
    image.convertTo(float_image, CV_32F, 1.0/255.0);
    
    if (image.channels() == 3) {
        std::vector<cv::Mat> channels;
        cv::split(float_image, channels);
        
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < image.rows; ++h) {
                for (int w = 0; w < image.cols; ++w) {
                    (*this)(0, c, h, w) = channels[c].at<float>(h, w);
                }
            }
        }
    } else {
        for (int h = 0; h < image.rows; ++h) {
            for (int w = 0; w < image.cols; ++w) {
                (*this)(0, 0, h, w) = float_image.at<float>(h, w);
            }
        }
    }
}

Tensor::Tensor(int batch_size, int channels, int height, int width) 
    : shape_({batch_size, channels, height, width}) {
    size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    data_ = std::make_unique<float[]>(size_);
    compute_strides();
    zero();
}

Tensor::Tensor(const Tensor& other) : shape_(other.shape_), strides_(other.strides_), size_(other.size_) {
    data_ = std::make_unique<float[]>(size_);
    std::memcpy(data_.get(), other.data_.get(), size_ * sizeof(float));
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        strides_ = other.strides_;
        size_ = other.size_;
        data_ = std::make_unique<float[]>(size_);
        std::memcpy(data_.get(), other.data_.get(), size_ * sizeof(float));
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept 
    : shape_(std::move(other.shape_)), strides_(std::move(other.strides_)), 
      size_(other.size_), data_(std::move(other.data_)) {
    other.size_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        size_ = other.size_;
        data_ = std::move(other.data_);
        other.size_ = 0;
    }
    return *this;
}

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    if (!shape_.empty()) {
        strides_[shape_.size() - 1] = 1;
        for (int i = shape_.size() - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }
}

int Tensor::compute_index(const std::vector<int>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::runtime_error("Index dimension mismatch");
    }
    
    int index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            throw std::runtime_error("Index out of bounds");
        }
        index += indices[i] * strides_[i];
    }
    return index;
}

float& Tensor::operator()(int b, int c, int h, int w) {
    return (*this)({b, c, h, w});
}

const float& Tensor::operator()(int b, int c, int h, int w) const {
    return (*this)({b, c, h, w});
}

float& Tensor::operator()(const std::vector<int>& indices) {
    return data_[compute_index(indices)];
}

const float& Tensor::operator()(const std::vector<int>& indices) const {
    return data_[compute_index(indices)];
}

void Tensor::zero() {
    std::fill(data_.get(), data_.get() + size_, 0.0f);
}

void Tensor::fill(float value) {
    std::fill(data_.get(), data_.get() + size_, value);
}

Tensor Tensor::clone() const {
    return Tensor(*this);
}

void Tensor::reshape(const std::vector<int>& new_shape) {
    int new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
    if (new_size != size_) {
        throw std::runtime_error("Cannot reshape tensor: size mismatch");
    }
    shape_ = new_shape;
    compute_strides();
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for addition");
    }
    
    Tensor result(shape_);
    for (int i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for subtraction");
    }
    
    Tensor result(shape_);
    for (int i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for multiplication");
    }
    
    Tensor result(shape_);
    for (int i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for addition");
    }
    
    for (int i = 0; i < size_; ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

void Tensor::print_shape() const {
    std::cout << "Shape: (";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i < shape_.size() - 1) std::cout << ", ";
    }
    std::cout << ")" << std::endl;
}

cv::Mat Tensor::to_image() const {
    if (shape_.size() != 4 || shape_[0] != 1) {
        throw std::runtime_error("Can only convert 4D tensor with batch size 1 to image");
    }
    
    int channels = shape_[1];
    int height = shape_[2];
    int width = shape_[3];
    
    cv::Mat image;
    
    if (channels == 3) {
        std::vector<cv::Mat> channel_mats(3);
        for (int c = 0; c < 3; ++c) {
            channel_mats[c] = cv::Mat(height, width, CV_32F);
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    channel_mats[c].at<float>(h, w) = (*this)(0, c, h, w);
                }
            }
        }
        cv::merge(channel_mats, image);
    } else if (channels == 1) {
        image = cv::Mat(height, width, CV_32F);
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                image.at<float>(h, w) = (*this)(0, 0, h, w);
            }
        }
    } else {
        throw std::runtime_error("Unsupported number of channels for image conversion");
    }
    
    // Convert back to 8-bit
    cv::Mat result;
    image.convertTo(result, CV_8U, 255.0);
    return result;
}