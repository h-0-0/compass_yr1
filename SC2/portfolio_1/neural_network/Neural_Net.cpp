# include "Neural_Net.h"


NeuralNetwork::NeuralNetwork(std::vector<uint> layer_sizes, Scalar learning_rate){
    std::vector< Matrix<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime> > Net;
    Matrix<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
    for(uint i = 0; i < layer_sizes.size(); i++){

        // We initialize layer in neuron_layers
        neuron_layers.push_back(new RowVector(layer_sizes[i+1]) )

        // We initialize layer in unactivated_layers
        unactivated_layers.push_back(new RowVector(layer_sizes[i+1]))

        // We initialize layer in errors
        errors.push_back

    }
}