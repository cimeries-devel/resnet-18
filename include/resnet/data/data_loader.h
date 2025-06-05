#ifndef RESNET_DATA_LOADER_H
#define RESNET_DATA_LOADER_H

#include "dataset.h"
#include "../utils/tensor.h"
#include <vector>
#include <random>

struct Batch {
    Tensor images;
    std::vector<int> labels;
    size_t batch_size;
};

class DataLoader {
public:
    DataLoader(const AnimalDataset& dataset, int batch_size, bool shuffle = true);
    
    bool has_next() const;
    Batch get_next_batch();
    void reset();
    
    size_t num_batches() const;
    size_t current_batch() const { return current_batch_; }

private:
    const AnimalDataset& dataset_;
    int batch_size_;
    bool shuffle_;
    
    std::vector<size_t> indices_;
    size_t current_index_;
    size_t current_batch_;
    
    std::mt19937 rng_;
    
    void shuffle_indices();
};

#endif // RESNET_DATA_LOADER_H