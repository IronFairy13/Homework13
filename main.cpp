#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kNumClasses = 10;
constexpr std::size_t kNumPixels = 784;
constexpr std::size_t kModelRowSize = kNumPixels + 1;  

std::vector<double> loadWeights(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open model file: " + path);
    }

    std::vector<double> weights;
    weights.reserve(kNumClasses * kModelRowSize);

    double value = 0.0;
    while (input >> value) {
        weights.push_back(value);
    }

    if (weights.size() != kNumClasses * kModelRowSize) {
        throw std::runtime_error("Unexpected number of coefficients in model file: " + path);
    }

    return weights;
}

int predictClass(const std::vector<double>& weights, const std::vector<double>& features) {
    const double* row = weights.data();
    double bestScore = -std::numeric_limits<double>::infinity();
    int bestClass = 0;

    for (std::size_t cls = 0; cls < kNumClasses; ++cls) {
        double score = row[0];  
        for (std::size_t i = 0; i < kNumPixels; ++i) {
            score += row[i + 1] * features[i];
        }

        if (score > bestScore) {
            bestScore = score;
            bestClass = static_cast<int>(cls);
        }

        row += kModelRowSize;
    }

    return bestClass;
}

bool parseCsvLine(const std::string& line, int& label, std::vector<double>& features) {
    const char* ptr = line.c_str();
    char* endPtr = nullptr;

    label = static_cast<int>(std::strtol(ptr, &endPtr, 10));
    if (endPtr == ptr) {
        return false;  
    }

    ptr = endPtr;
    if (*ptr == ',') {
        ++ptr;
    } else if (*ptr != '\0') {
        return false;  
    }

    constexpr double inv255 = 1.0 / 255.0;
    for (std::size_t i = 0; i < kNumPixels; ++i) {
        long value = std::strtol(ptr, &endPtr, 10);
        if (endPtr == ptr) {
            return false;  
        }

        features[i] = static_cast<double>(value) * inv255;

        ptr = endPtr;
        if (i + 1 < kNumPixels) {
            if (*ptr != ',') {
                return false;
            }
            ++ptr;
        } else {
            break;
        }
    }

    return true;
}

} 

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <test.csv> <logreg_coef.txt>\n";
        return 1;
    }

    const std::string testPath = argv[1];
    const std::string modelPath = argv[2];

    std::vector<double> weights;
    try {
        weights = loadWeights(modelPath);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }

    std::ifstream testFile(testPath);
    if (!testFile) {
        std::cerr << "Failed to open test data file: " << testPath << '\n';
        return 1;
    }

    std::vector<double> features(kNumPixels, 0.0);
    std::string line;
    std::size_t total = 0;
    std::size_t correct = 0;

    while (std::getline(testFile, line)) {
        if (line.empty()) {
            continue;
        }

        int label = 0;
        if (!parseCsvLine(line, label, features)) {
            std::cerr << "Malformed CSV line at sample " << (total + 1) << '\n';
            return 1;
        }

        const int predicted = predictClass(weights, features);
        if (predicted == label) {
            ++correct;
        }
        ++total;
    }

    if (total == 0) {
        std::cerr << "Test file does not contain any samples\n";
        return 1;
    }

    const double accuracy = static_cast<double>(correct) / static_cast<double>(total);
    std::cout << std::fixed << std::setprecision(3) << accuracy << '\n';
    return 0;
}
