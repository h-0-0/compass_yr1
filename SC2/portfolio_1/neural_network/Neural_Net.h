#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>

// use typedefs for future ease for changing data types like : float to double
typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

/*! @brief Neural Network Class
*
* Add some text
*
* @class this is a class
*/
class NeuralNetwork {
public:
    // constructor
    NeuralNetwork(std::vector<uint> layer_sizes, Scalar learning_rate = Scalar(0.005));

    // function for forward propagation of data
    void ForwardPass(RowVector& input);

    // function for backward propagation of errors made by neurons
    void BackwardPass(RowVector& output);

    // function to calculate errors made by neurons in each layer
    void CalcErrors(RowVector& output);

    // function to update the weights of connections
    void UpdateWeights();

    // function to train the neural network give an array of data points
    void Train(std::vector<RowVector*> data);

    // storage objects for working of neural network
    /*
          use pointers when using std::vector<Class> as std::vector<Class> calls destructor of
          Class as soon as it is pushed back! when we use pointers it can't do that, besides
          it also makes our neural network class less heavy!! It would be nice if you can use
          smart pointers instead of usual ones like this
        */
    std::vector<RowVector*> neuron_layers; // stores the different layers of our network
    std::vector<RowVector*> unactivated_layers; // stores the unactivated (activation fn not yet applied) values of layers
    std::vector<RowVector*> errors; // stores the error contribution of each neurons
    std::vector<Matrix*> weights; // the connection weights itself
    Scalar learning_rate;
};