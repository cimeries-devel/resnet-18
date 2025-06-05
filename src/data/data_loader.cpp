#include "resnet/data/data_loader.h"
#include <algorithm>

DataLoader::DataLoader(const AnimalDataset& dataset, int batch_size, bool shuffle)
    : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle),
      current_index_(0), current_batch_(0), rng_(std::random_device{}()) {
    
    // Initialize indices
    indices_.resize(dataset_.size());
    std::iota(indices_.begin(), indices_.end(), 0);
    
    if (shuffle_) {
        shuffle_indices();
    }
}

void DataLoader::shuffle_indices() {
    std::shuffle(indices_.begin(), indices_.end(), rng_);
}

bool DataLoader::has_next() const {
    return current_index_ < dataset_.size();
}

Batch DataLoader::get_next_batch() {
    if (!has_next()) {
        throw std::runtime_error("No more batches available");
    }
    
    size_t actual_batch_size = std::min(static_cast<size_t>(batch_size_), 
                                       dataset_.size() - current_index_);
    
    // Get first sample to determine image dimensions
    Sample first_sample = dataset_.get_sample(indices_[current_index_]);
    const auto& img_shape = first_sample.image.shape();
    
    // Create batch tensors
    Tensor batch_images(actual_batch_size, img_shape[1], img_shape[2], img_shape[3]);
    std::vector<int> batch_labels;
    batch_labels.reserve(actual_batch_size);
    
    // Fill batch
    for (size_t i = 0; i < actual_batch_size; ++i) {
        Sample sample = dataset_.get_sample(indices_[current_index_ + i]);
        
        // Copy image data
        const auto& sample_shape = sample.image.shape();
        for (int c = 0; c < sample_shape[1]; ++c) {
            for (int h = 0; h < sample_shape[2]; ++h) {
                for (int w = 0; w < sample_shape[3]; ++w) {
                    batch_images(i, c, h, w) = sample.image(0, c, h, w);
                }
            }
        }
        
        batch_labels.push_back(sample.label);
    }
    
    current_index_ += actual_batch_size;
    current_batch_++;
    
    Batch batch{
    std::move(batch_images),
    std::move(batch_labels),
    actual_batch_size};
    
    return batch;
}

void DataLoader::reset() {
    current_index_ = 0;
    current_batch_ = 0;
    
    if (shuffle_) {
        shuffle_indices();
    }
}

size_t DataLoader::num_batches() const {
    return (dataset_.size() + batch_size_ - 1) / batch_size_;
}