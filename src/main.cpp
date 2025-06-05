#include <iostream>
#include <chrono>
#include <iomanip>
#include <experimental/filesystem>
#include "resnet/model/resnet18.h"
#include "resnet/data/dataset.h"
#include "resnet/data/data_loader.h"
#include <opencv2/opencv.hpp>

namespace fs = std::experimental::filesystem;

void print_banner() {
    std::cout << "========================================" << std::endl;
    std::cout << "         ResNet-18 Animal Classifier   " << std::endl;
    std::cout << "========================================" << std::endl;
}

void check_directory_structure(const std::string& data_root) {
    std::cout << "\n--- Checking Directory Structure ---" << std::endl;
    std::cout << "Current working directory: " << fs::current_path() << std::endl;
    std::cout << "Data root: " << data_root << std::endl;

    if (!fs::exists(data_root)) {
        std::cout << "❌ ERROR: Data root directory does not exist!" << std::endl;
        std::cout << "Please create: " << data_root << std::endl;
        std::cout << "From your project directory, run:" << std::endl;
        std::cout << "mkdir -p " << data_root << "/train/cats" << std::endl;
        std::cout << "mkdir -p " << data_root << "/train/dogs" << std::endl;
        std::cout << "mkdir -p " << data_root << "/test/cats" << std::endl;
        std::cout << "mkdir -p " << data_root << "/test/dogs" << std::endl;
        return;
    }
    std::cout << "✅ Data root exists" << std::endl;

    // Check train directory
    std::string train_path = data_root + "/train";
    if (!fs::exists(train_path)) {
        std::cout << "❌ ERROR: Train directory does not exist!" << std::endl;
        std::cout << "Please create: " << train_path << std::endl;
        return;
    }
    std::cout << "✅ Train directory exists" << std::endl;

    // Check test directory
    std::string test_path = data_root + "/test";
    if (!fs::exists(test_path)) {
        std::cout << "❌ ERROR: Test directory does not exist!" << std::endl;
        std::cout << "Please create: " << test_path << std::endl;
        return;
    }
    std::cout << "✅ Test directory exists" << std::endl;

    // Check class directories
    std::vector<std::string> expected_classes = {"cats", "dogs"};

    for (const auto& split : {"train", "test"}) {
        std::string split_path = data_root + "/" + split;
        std::cout << "\nChecking " << split << " directory:" << std::endl;

        for (const auto& class_name : expected_classes) {
            std::string class_path = split_path + "/" + class_name;
            if (!fs::exists(class_path)) {
                std::cout << "❌ ERROR: " << class_path << " does not exist!" << std::endl;
                std::cout << "Please create: mkdir -p " << class_path << std::endl;
            } else {
                // Count images
                int image_count = 0;
                try {
                    for (const auto& entry : fs::directory_iterator(class_path)) {
                        if (fs::is_regular_file(entry)) {
                            std::string ext = entry.path().extension().string();
                            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                                image_count++;
                            }
                        }
                    }
                    std::cout << "✅ " << class_path << " - " << image_count << " images" << std::endl;

                    if (image_count == 0) {
                        std::cout << "⚠️  Warning: No images found in " << class_path << std::endl;
                        std::cout << "   Please add .jpg, .jpeg, .png, or .bmp images to this folder" << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cout << "❌ ERROR reading " << class_path << ": " << e.what() << std::endl;
                }
            }
        }
    }
}

int main() {
    print_banner();

    try {
        // **CAMBIO IMPORTANTE**: Solo 2 clases para cats vs dogs
        std::vector<std::string> animal_classes = {"cats", "dogs"};

        // **CAMBIO**: Verificar ruta absoluta si es necesario
        std::string data_root = "data/animals";

        // Si el programa se ejecuta desde build/, ajustar la ruta
        if (!fs::exists(data_root)) {
            data_root = "../data/animals";
            std::cout << "Trying alternative path: " << data_root << std::endl;
        }

        // Verificar estructura de directorios ANTES de crear el modelo
        check_directory_structure(data_root);

        // Inicializar modelo con el número correcto de clases
        ResNet18 model(animal_classes.size());

        std::cout << "\nResNet-18 model initialized with " << animal_classes.size()
                  << " classes: ";
        for (const auto& cls : animal_classes) {
            std::cout << cls << " ";
        }
        std::cout << std::endl;

        // Solo intentar cargar datos si las carpetas existen
        if (fs::exists(data_root + "/train")) {
            std::cout << "\n--- Testing Data Loading ---" << std::endl;

            try {
                std::cout << "Attempting to load training dataset..." << std::endl;
                AnimalDataset train_dataset(data_root, "train");
                train_dataset.set_augmentation(true);
                train_dataset.print_dataset_info();

                if (train_dataset.size() == 0) {
                    std::cout << "❌ ERROR: No images found in training dataset!" << std::endl;
                    std::cout << "Please add images to:" << std::endl;
                    std::cout << "  " << data_root << "/train/cats/" << std::endl;
                    std::cout << "  " << data_root << "/train/dogs/" << std::endl;
                } else {
                    std::cout << "✅ Dataset loaded successfully!" << std::endl;

                    // Test a batch
                    DataLoader train_loader(train_dataset, 2, true);
                    if (train_loader.has_next()) {
                        Batch batch = train_loader.get_next_batch();
                        std::cout << "✅ Successfully loaded first batch:" << std::endl;
                        std::cout << "   Batch size: " << batch.batch_size << std::endl;
                        std::cout << "   Image shape: ";
                        batch.images.print_shape();

                        // Test model forward pass
                        std::cout << "\n--- Testing Model Forward Pass ---" << std::endl;
                        model.set_training(false);

                        // Extract first image
                        Tensor single_image(1, batch.images.shape()[1],
                                          batch.images.shape()[2], batch.images.shape()[3]);

                        for (int c = 0; c < batch.images.shape()[1]; ++c) {
                            for (int h = 0; h < batch.images.shape()[2]; ++h) {
                                for (int w = 0; w < batch.images.shape()[3]; ++w) {
                                    single_image(0, c, h, w) = batch.images(0, c, h, w);
                                }
                            }
                        }

                        std::cout << "Running forward pass..." << std::endl;
                        Tensor output = model.forward(single_image);
                        std::cout << "✅ Forward pass successful!" << std::endl;
                        std::cout << "Output shape: ";
                        output.print_shape();

                        // Test prediction
                        int predicted_class = model.predict_class(single_image);
                        auto probabilities = model.predict_probabilities(single_image);

                        std::cout << "Prediction results:" << std::endl;
                        std::cout << "  Predicted class: " << animal_classes[predicted_class] << std::endl;
                        std::cout << "  Confidence: " << std::fixed << std::setprecision(3)
                                  << probabilities[predicted_class] * 100 << "%" << std::endl;

                        std::cout << "\n🎉 Everything working correctly!" << std::endl;
                    }
                }

            } catch (const std::exception& e) {
                std::cout << "❌ ERROR during data loading: " << e.what() << std::endl;
            }
        } else {
            std::cout << "\n❌ Cannot test data loading - directories don't exist" << std::endl;
        }

        std::cout << "\n--- Setup Instructions ---" << std::endl;
        std::cout << "To get started:" << std::endl;
        std::cout << "1. Create the directory structure:" << std::endl;
        std::cout << "   mkdir -p " << data_root << "/train/cats" << std::endl;
        std::cout << "   mkdir -p " << data_root << "/train/dogs" << std::endl;
        std::cout << "   mkdir -p " << data_root << "/test/cats" << std::endl;
        std::cout << "   mkdir -p " << data_root << "/test/dogs" << std::endl;
        std::cout << "2. Add images to each folder" << std::endl;
        std::cout << "3. Run the program again" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}