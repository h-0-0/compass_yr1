#include "Neural_Net.h"
#include "Print_Utils.h"
#include <iostream>
#include <random>
#include <string>
#include<memory>

int main()
{
    std::shared_ptr<ActivationFunction> actFun = std::make_shared<ReLU>(); // Create activation function
    std::vector<double> input = {1.0, 2.0}; // Create input vector

    std::vector<Layer> layers;  // Declaration of vector of layers

    // Create individual layers and add them to the vector
    Layer layer1(2, 4, actFun);
    Layer layer2(4, 2, actFun);
    layers.push_back(layer1);
    layers.push_back(layer2);
    std::cout << "Layer 0:" << std::endl;
    printMatrix(layers[0].get_weights());
    std::cout << "Layer 1:" << std::endl;
    printMatrix(layers[1].get_weights());
    
    // Create neural network with layers
    NeuralNetwork nn(layers);

    std::vector<double> result = nn.forwardPass(input);
    printVector(result);
    return 0;
};


// TODO: create optimization file that carries out training
// TODO: get it working on some regression problem