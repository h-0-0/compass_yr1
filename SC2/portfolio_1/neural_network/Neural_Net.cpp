#include "Print_Utils.h"
#include "Utils.h"
#include "Neural_Net.h"
#include <iostream>
#include <random>
#include <string>
#include <memory>
#include <numeric>

// Implementations for initialization related functions
// ----------------------------------------------------------------------

// Function for computing the MSE between the output and target vectors
double computeMSE(const std::vector<double>& output, const std::vector<double>& target) {
    if (output.size() != target.size()) {
        throw std::runtime_error("Vector dimensions do not match.");
    }

    size_t size = output.size();
    double sumSquaredError = 0.0;

    for (size_t i = 0; i < size; i++) {
        double error = output[i] - target[i];
        sumSquaredError += error * error;
    }

    return sumSquaredError / static_cast<double>(size);
}

// Function to generate a random double value within a specified range
double randomDouble(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

// Function to initialize a matrix of weights with random values
std::vector<std::vector<double>> initializeWeights(size_t numRows, size_t numCols, double min, double max) {
    // Check that the number of rows and columns is greater than 0
    if (numRows == 0 || numCols == 0) {
        throw std::invalid_argument("Number of rows and columns must be greater than 0");
    }
    // Check that the min value is less than the max value
    if (min >= max) {
        throw std::invalid_argument("Min value must be less than max value");
    }
    // Initialize the matrix of weights with random values
    std::vector<std::vector<double>> weights(numRows, std::vector<double>(numCols));
    for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < numCols; ++j) {
            weights[i][j] = randomDouble(min, max);
        }
    }
    return weights;
}

// Function to initialize a vector of random biases according to arguments
std::vector<double> initializeBiases(size_t numBiases, double min, double max){
    // Check that the number of biases is greater than 0
    if (numBiases == 0) {
        throw std::invalid_argument("Number of biases must be greater than 0");
    }
    // Check that the min value is less than the max value
    if (min >= max) {
        throw std::invalid_argument("Min value must be less than max value");
    }
    // Initialize the vector of biases with random values
    std::vector<double> biases(numBiases);
    for (size_t i = 0; i < numBiases; ++i) {
        biases[i] = randomDouble(min, max);
    }
    return biases;
}

// Implementations for ActivationFunction virtual class
// ----------------------------------------------------------------------

// Class for the ReLU activation function, it inherits from the virtual ActivationFunction class
double ReLU::activate(double x) const {
    double out ;
    if(x>0){
        out = x;
    }
    else{
        out = 0;
    }
    return out;
}

double ReLU::derivative(double x) const {
    double out ;
    if(x>0){
        out = 1;
    }
    else{
        out = 0;
    }
    return out;
}

std::string ReLU::get_name() const{
    return "ReLU";
}

// Implementation for Layer class
// ----------------------------------------------------------------------

// Constructor for the Layer class when we have a matrix of weights we want to use
// Layer::Layer(const std::vector<std::vector<double>>& initialWeights, const std::shared_ptr<ActivationFunction>& actFun):
//     weights(initialWeights), actFun(actFun) {}

Layer::Layer(const std::vector<std::vector<double>>& initialWeights, const std::vector<double>& initialBiases, const std::shared_ptr<ActivationFunction>& actFun)
    : weights(initialWeights), biases(initialBiases), actFun(actFun), preActivationValues(initialWeights.size())
{
    if (isMatrix(weights) == false) {
        throw std::invalid_argument("Weights must be a matrix");
    }
    if (isVector(biases) == false) {
        throw std::invalid_argument("Biases must be a vector");
    }
    // Check that the number of biases is the same as the number of rows in the weights matrix
    if (biases.size() != weights.size()) {
        throw std::invalid_argument("Number of biases must be the same as the number of rows in the weights matrix");
    }
}

// Constructor for the Layer class when we want to randomly initialize the weights
Layer::Layer(int n_in, int n_out, const std::shared_ptr<ActivationFunction>& actFun): 
    weights(initializeWeights(n_out, n_in, -1, 1)), biases(initializeBiases(n_out, -1, 1)), actFun(actFun), preActivationValues(n_out) {}

// Deconstructor for the Layer class
Layer::~Layer(){}

// Computes the product of the input with the matrix of weights and adds the bias for the layer and puts through activation function
std::vector<double> Layer::layer_pass(const std::vector<double>& inputs) {
    // Check the inputs is a vector and not empty or a matrix
    if(isVector(inputs) == false){
        throw std::invalid_argument("Inputs must be a vector");
    }
    std::vector<double> out(weights.size());
    std::vector<double> out_pre(weights.size()) ;
    for (size_t i = 0; i < weights.size(); ++i){
        double sum = 0;
        for (size_t j = 0; j < inputs.size(); ++j){
            sum += weights[i][j] * inputs[j];
        }
        sum += biases[i];
        preActivationValues[i] = sum;
        out[i] = actFun->activate(sum); 
    }
    return out;
};

// Computes the derivative of the activation function for the layer
std::vector<double> Layer::get_activation_derivative() const{
    std::vector<double> out ;
    for (size_t i = 0; i < preActivationValues.size(); ++i){
        out.push_back(actFun->derivative(preActivationValues[i])); 
    }
    return out;
};

// Returns the weights for the layer
std::vector<std::vector<double>> Layer::get_weights() const{
    return weights;
};

// Returns the biases for the layer
std::vector<double> Layer::get_biases() const{
    return biases;
};

// Returns the activation function for the layer
std::shared_ptr<ActivationFunction> Layer::get_actFun() const{
    return actFun;
};

// Returns the values of neurons before being passed through the activation function
std::vector<double> Layer::get_preActivationValues() const{
    return preActivationValues;
};

// Returns all stored weight gradients for the layer
std::vector<std::vector<std::vector<double>>> Layer::get_weightGradients() const{
    return weightGradients;
};

// Returns all stored bias gradients for the layer
std::vector<std::vector<double>> Layer::get_biasGradients() const{
    return biasGradients;
};

// Sets the weights for the layer
void Layer::set_weights(const std::vector<std::vector<double>>& newWeights){
    // Check that the new weights matrix has the same dimensions as the current weights matrix
    if (newWeights.size() != weights.size() || newWeights[0].size() != weights[0].size()) {
        throw std::invalid_argument("New weights matrix must have the same dimensions as the current weights matrix");
    }

    weights = newWeights;
};

// Sets the biases for the layer
void Layer::set_biases(const std::vector<double>& newBiases){
    // Check that the new biases vector has the same dimensions as the current biases vector
    if (newBiases.size() != biases.size()) {
        throw std::invalid_argument("New biases vector must have the same dimensions as the current biases vector");
    }

    biases = newBiases;
};

// Add a new set of weight gradients
void Layer::add_weightGradients(const std::vector<std::vector<double>>& newWeightGradients){
    // Check that the new weight gradients matrix has the same dimensions as the current weight gradients matrix
    if (newWeightGradients.size() != weightGradients.size() || newWeightGradients[0].size() != weightGradients[0].size()) {
        throw std::invalid_argument("New weight gradients matrix must have the same dimensions as the current weight gradients matrix");
    }

    weightGradients.push_back(newWeightGradients);
};

// Add a new set of bias gradients
void Layer::add_biasGradients(const std::vector<double>& newBiasGradients){
    // Check that the new bias gradients vector has the same dimensions as the current bias gradients vector
    if (newBiasGradients.size() != biasGradients.size()) {
        throw std::invalid_argument("New bias gradients vector must have the same dimensions as the current bias gradients vector");
    }

    biasGradients.push_back(newBiasGradients);
};

// Clears all stored weight gradients for the layer
void Layer::clear_weightGradients(){
    weightGradients.clear();
};

// Clears all stored bias gradients for the layer
void Layer::clear_biasGradients(){
    biasGradients.clear();
};

// Prints the weights and activation function for the layer
void Layer::print() const {
    std::cout << "Weights:" << std::endl;
    printMatrix(weights);

    std::cout << "Activation Function: ";
    if (actFun) {
        std::cout << actFun->get_name() << std::endl;
    } else {
        std::cout << "None" << std::endl;
    }
};

// Implementation for Neural Network class
// ----------------------------------------------------------------------
// Constructor for the Neural Network class
NeuralNetwork::NeuralNetwork(const std::vector<Layer>& layers): layers(layers) {
    // Check the dimension of layer sizes match
    for (size_t i = 0; i < layers.size() - 1; ++i) {
        if (layers[i].get_weights().size() != layers[i + 1].get_weights()[0].size()) {
            throw std::invalid_argument("Layer sizes must match");
        }
    }
}

// Deconstructor for the Neural Network class
NeuralNetwork::~NeuralNetwork(){}

// Computes a forward pass through the network
std::vector<double> NeuralNetwork::forwardPass(const std::vector<double>& input){
    std::vector<double> out = input;
    for(size_t i=0; i < layers.size(); ++i){
        out = layers[i].layer_pass(out);
    }
    return out;
};

// Computes the gradient of the loss of the output of the network 
std::vector<double> NeuralNetwork::computeLossGradient(const std::vector<double>& output, const std::vector<double>& target) const {
    // Check that the output and target vectors have the same size
    if (output.size() != target.size()) {
        throw std::invalid_argument("Output and target sizes must match");
    }

    std::vector<double> lossGradient(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        // Compute the derivative of the MSE loss with respect to the output
        lossGradient[i] = 2.0 * (output[i] - target[i]);
    }

    return lossGradient;
};

// Computes the gradients of the weights for a layer
std::vector<std::vector<double>> NeuralNetwork::computeWeightGradients(const std::vector<double>& delta, const Layer& layer) const {
    // Check that the delta vector and the weights matrix have the same number of rows
    if (delta.size() != layer.get_weights().size()) {
        throw std::invalid_argument("Delta and weights sizes must match");
    }

    // Compute the gradients for the weights
    std::vector<std::vector<double>> weightGradients(layer.get_weights().size(), std::vector<double>(layer.get_weights()[0].size()));
    for (size_t i = 0; i < weightGradients.size(); ++i) {
        for (size_t j = 0; j < weightGradients[0].size(); ++j) {
            weightGradients[i][j] = delta[i] * layer.get_weights()[i][j];
        }
    }

    return weightGradients;
};

// Computes the gradients of the biases for a layer
std::vector<double> NeuralNetwork::computeBiasGradients(const std::vector<double>& delta) const {
    // Compute the gradients for the biases
    std::vector<double> biasGradients(delta.size());
    for (size_t i = 0; i < biasGradients.size(); ++i) {
        biasGradients[i] = delta[i];
    }

    return biasGradients;
};

// Updates the weights and biases for a layer
void NeuralNetwork::updateWeightsAndBiases(Layer& layer, const std::vector<std::vector<double>>& weightGradients, const std::vector<double>& biasGradients, double learningRate) {
    // Check that the weight gradients and the weights matrix have the same dimensions
    if (weightGradients.size() != layer.get_weights().size() || weightGradients[0].size() != layer.get_weights()[0].size()) {
        throw std::invalid_argument("Weight gradients and weights sizes must match");
    }

    // Check that the bias gradients and the weights matrix have matching dimensions
    if (biasGradients.size() != layer.get_weights().size()) {
        throw std::invalid_argument("Bias gradients and weights sizes must match");
    }
    // Set new weights
    auto new_weights = layer.get_weights();
    for (size_t i = 0; i < layer.get_weights().size(); ++i) {
        for (size_t j = 0; j < layer.get_weights()[0].size(); ++j) {
            new_weights[i][j] -= learningRate * weightGradients[i][j];
        }
    }
    layer.set_weights(new_weights);
    // Set new biases
    auto new_biases = layer.get_biases();
    for (size_t i = 0; i < layer.get_biases().size(); ++i) {
        new_biases[i] -= learningRate * biasGradients[i];
    }
    layer.set_biases(new_biases);
};

// Computes the delta for a layer
std::vector<double> NeuralNetwork::computeDelta(const std::vector<double>& delta, const Layer& layer) const {
    // Check that the delta vector and the weights matrix have the same number of rows
    if (delta.size() != layer.get_weights().size()) {
        throw std::invalid_argument("Delta and weights sizes must match");
    }

    // Compute the delta for the previous layer
    std::vector<double> previousDelta(layer.get_weights()[0].size());
    std::vector<double> act_deriv = layer.get_activation_derivative();
    for (size_t i = 0; i < previousDelta.size(); ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < delta.size(); ++j) {
            sum += delta[j] * layer.get_weights()[j][i];
        }
        previousDelta[i] = sum * act_deriv[i];
    }

    return previousDelta;
};

// For a mini-batch of data-points backpropagates the loss and updates the weights and biases accordingly using gradient descent
double NeuralNetwork::backpropagation(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, double learningRate) {
    // Initialize containers for the gradients of the weights and biases
    std::vector<std::vector<std::vector<std::vector<double>>>> minibatchWeightGradients(inputs.size(), std::vector<std::vector<std::vector<double>>>(layers.size()));
    std::vector<std::vector<std::vector<double>>> minibatchBiasGradients(inputs.size(), std::vector<std::vector<double>>(layers.size()));
    // Initialize container for the loss
    double avg_loss = 0.0;
    // Loop to calculate the loss and gradients for each data point in the batch
    for(size_t i=0; i < inputs.size(); ++i){
        // Get the input and target for the current data point
        std::vector<double> input = inputs[i];
        std::vector<double> target = targets[i];

        // Perform forward pass to get the output
        std::vector<double> output = forwardPass(input);

        // Compute the Loss Gradient
        std::vector<double> delta = computeLossGradient(output, target);

        // Keep track of the loss
        avg_loss += computeMSE(output, target);

        // Loop to calculate the gradients for each layer
        for (int j = layers.size() - 1; j >= 0; --j) {
            Layer& layer = layers[j];

            // Compute gradients for weights and biases
            minibatchWeightGradients[i][j] = computeWeightGradients(delta, layer);
            minibatchBiasGradients[i][j] = computeBiasGradients(delta);

            // Compute gradients for previous layer
            delta = computeDelta(delta, layer);
        }
    }

    for (int i = layers.size() - 1; i >= 0; --i) {
        std::vector<std::vector<double>> avgWeightGradients(minibatchWeightGradients[0][i].size(),std::vector<double>(minibatchWeightGradients[0][i][0].size(), 0));
        std::vector<double> avgBiasGradients(minibatchBiasGradients[0][i].size(), 0);
        // For current layer i, get the gradients for the weights and biases for all the data points in the batch and average them
        for (size_t j = 0; j < minibatchWeightGradients.size(); ++j) {
            avgWeightGradients = matrixAddition(avgWeightGradients, minibatchWeightGradients[j][i]);
            avgBiasGradients = vectorAddition(avgBiasGradients, minibatchBiasGradients[j][i]);
        }
        matrixScalarMultiplication(avgWeightGradients, 1.0 / minibatchWeightGradients.size());
        vectorScalarMultiplication(avgBiasGradients, 1.0 / minibatchBiasGradients.size());
        // Get the layer
        Layer& layer = layers[i];
        // Update weights and biases
        updateWeightsAndBiases(layer, avgWeightGradients, avgBiasGradients, learningRate);
    }
    return avg_loss / inputs.size();
};

// Prints information on all the layers in the network
void NeuralNetwork::print() const {
    std::cout << "Neural Network with following layers{" << std::endl;
    for(const auto& l : layers){
        std::cout << "Layer:" << std::endl;
        l.print();
        std::cout << "\n" << std::endl;
    }
    std::cout << "}" << std::endl;
};


