#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <string>
#include <memory>

// Function for computing the MSE between the output and target vectors
double computeMSE(const std::vector<double>& output, const std::vector<double>& target);

// Function for creating a random double number between min and max
double randomDouble(double min, double max);

// Function to initialize a matrix of random weights according to arguments
std::vector<std::vector<double>> initializeWeights(size_t numRows, size_t numCols, double min, double max);

// Function to initialize a vector of random biases according to arguments
std::vector<double> initializeBiases(size_t numBiases, double min, double max);

// Virtual class which all activation function classes should implement
class ActivationFunction {
public:
    virtual double activate(double x) const = 0;
    virtual double derivative(double x) const = 0;
    virtual std::string get_name() const = 0;
};

// Class for the ReLU activation function, it inherits from the virtual ActivationFunction class
class ReLU : public ActivationFunction {
public:
    double activate(double x) const override;
    double derivative(double x) const override;
    std::string get_name() const override;
};

// Layer class
class Layer {
private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    const std::shared_ptr<ActivationFunction> actFun;
    std::vector<double> preActivationValues;
    // Store the gradients of the weights and biases for all the data points in the current batch
    std::vector<std::vector<std::vector<double>>> weightGradients;
    std::vector<std::vector<double>> biasGradients;

public:
    Layer(const std::vector<std::vector<double>>& initialWeights,
            const std::vector<double>& initialBiases,
            const std::shared_ptr<ActivationFunction>& actFun
        );
    Layer(int n_in,
            int n_out,
            const std::shared_ptr<ActivationFunction>& actFun
        );
    ~Layer();

    std::vector<double> layer_pass(const std::vector<double>& inputs);

    std::vector<double> get_activation_derivative() const;

    std::vector<std::vector<double>> get_weights() const;

    std::vector<double> get_biases() const;

    std::shared_ptr<ActivationFunction> get_actFun() const;

    std::vector<double> get_preActivationValues() const;

    std::vector<std::vector<std::vector<double>>> get_weightGradients() const;

    std::vector<std::vector<double>> get_biasGradients() const;

    void set_weights(const std::vector<std::vector<double>>& newWeights);

    void set_biases(const std::vector<double>& newBiases);

    void add_weightGradients(const std::vector<std::vector<double>>& newWeightGradients);

    void add_biasGradients(const std::vector<double>& newBiasGradients);

    void clear_weightGradients();

    void clear_biasGradients();

    void print() const;
};

// Neural Network class
class NeuralNetwork {
private:
    std::vector<Layer> layers;

public:
    NeuralNetwork(const std::vector<Layer>& layers);
    ~NeuralNetwork();

    std::vector<double> forwardPass(const std::vector<double>& input);

    std::vector<double> computeLossGradient(const std::vector<double>& output, const std::vector<double>& target) const;

    std::vector<std::vector<double>> computeWeightGradients(const std::vector<double>& delta, const Layer& layer) const;

    std::vector<double> computeBiasGradients(const std::vector<double>& delta) const;

    void updateWeightsAndBiases(Layer& layer, const std::vector<std::vector<double>>& weightGradients, const std::vector<double>& biasGradients, double learningRate);

    std::vector<double> computeDelta(const std::vector<double>& delta, const Layer& layer) const;

    double backpropagation(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& target, double learningRate);

    void print() const;
};

#endif // NEURALNETWORK_H
