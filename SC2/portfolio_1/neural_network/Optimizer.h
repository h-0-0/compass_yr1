// Optimizer.h
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <numeric>
#include <random>
#include "Neural_Net.h"

// Optimizer class
template <typename DT, typename LT>
class Optimizer {
private:
    NeuralNetwork nn;
    const std::vector<DT> trainingData;
    const std::vector<LT> trainingLabels;
    const std::vector<DT> testData;
    const std::vector<LT> testLabels;
    const std::vector<DT> validationData;
    const std::vector<LT> validationLabels;
    const std::string opt_method;
    const double learningRate;
    const int batchSize;
    const int numEpochs;
    const int numBatches;
    std::vector<int> trainingInd;

public:
    Optimizer(NeuralNetwork& nn,
            const std::vector<DT>& trainingData,
            const std::vector<LT>& trainingLabels,
            const std::vector<DT>& testData,
            const std::vector<LT>& testLabels,
            const std::vector<DT>& validationData,
            const std::vector<LT>& validationLabels,
            const std::string opt_method = "SGD",
            const double learningRate = 0.01,
            const int batchSize = 32,
            const int numEpochs = 10);

    Optimizer(NeuralNetwork& nn,
            const std::vector<DT>& trainingData,
            const std::vector<LT>& trainingLabels,
            const std::vector<DT>& testData,
            const std::vector<LT>& testLabels,
            const std::string opt_method = "SGD",
            const double learningRate = 0.01,
            const int batchSize = 32,
            const int numEpochs = 10);

    ~Optimizer();

    // Gets a batch of data and labels from the training data
    std::pair<std::vector<DT>, std::vector<LT>> getTrainBatch(const int& batchNum) const;

    // Gets all the training data
    std::pair<std::vector<DT>, std::vector<LT>> getTrainData() const;

    // Gets a batch of data and labels from the validation data
    std::pair<std::vector<DT>, std::vector<LT>> getValidation() const;

    // Gets a batch of data and labels from the test data
    std::pair<std::vector<DT>, std::vector<LT>> getTest() const;

    // Shuffles the training data and labels
    void shuffleTrainData();

    // Performs one epochs worth of training, returns the average loss on each batch
    std::vector<double> singleTrain();

    // Performs a single testing step, ie. forward pass and loss calculation for whole test set and returns the average loss
    double singleTest();

    // Iteratively conducts training step for every batch and for number of specified epochs
    void train();

    NeuralNetwork getNn() const;

    std::vector<int> getTrainingInd() const;

};

#include "Optimizer.tpp"

#endif // OPTIMIZER_H
