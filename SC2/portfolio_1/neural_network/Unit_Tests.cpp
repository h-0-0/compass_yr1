#include "gtest/gtest.h"
#include "Print_Utils.h"
#include "Neural_Net.h"
#include "Optimizer.h"

// Test randomDouble function with positive and negative values
TEST(RandomDoubleTest, PositiveValues) {
    double result = randomDouble(1.0, 2.0);
    EXPECT_TRUE(result >= 1.0 && result <= 2.0);
}
TEST(RandomDoubleTest, NegativeValues) {
    double result = randomDouble(-2.0, -1.0);
    EXPECT_TRUE(result >= -2.0 && result <= -1.0);
}

 // Test initializeWeights function
TEST(InitializeWeightsTest, DifferentSizes) {
    std::vector<std::vector<double>> weights1 = initializeWeights(2, 3, 0.0, 1.0);
    EXPECT_EQ(weights1.size(), 2);
    EXPECT_EQ(weights1[0].size(), 3);
    std::vector<std::vector<double>> weights2 = initializeWeights(4, 5, -1.0, 0.0);
    EXPECT_EQ(weights2.size(), 4);
    EXPECT_EQ(weights2[0].size(), 5);
}

// Test initializeBiases function
TEST(initializeBiasesTest, PositiveValues){
    std::vector<double> biases = initializeBiases(5, 1.0, 2.0);
    EXPECT_EQ(biases.size(), 5);
    for (double bias : biases) {
        EXPECT_TRUE(bias >= 1.0 && bias <= 2.0);
    }
}
TEST(initializeBiasesTest, ZeroValues){
    EXPECT_THROW(initializeBiases(0, 1.0, 2.0), std::invalid_argument);
}
TEST(initializeBiasesTest, InvalidValues){
    EXPECT_THROW(initializeBiases(5, 2.0, 1.0), std::invalid_argument);
}

// Test ReLU class
TEST(ReLUClassTest, Activate) {
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    double result = relu->activate(1.0);
    EXPECT_DOUBLE_EQ(result, 1.0);
}
TEST(ReLUClassTest, Derivative) {
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    double result = relu->derivative(1.0);
    EXPECT_DOUBLE_EQ(result, 1.0);
}
TEST(ReLUClassTest, GetName) {
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    std::string result = relu->get_name();
    EXPECT_EQ(result, "ReLU");
}

 // Test Layer class
TEST(LayerTest, Constructor) {
    std::vector<std::vector<double>> weights = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<double> biases = {1.0, 2.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer layer(weights, biases, relu);
    std::vector<std::vector<double>> result = layer.get_weights();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].size(), 2);
    EXPECT_DOUBLE_EQ(result[0][0], 1.0);
    EXPECT_DOUBLE_EQ(result[0][1], 2.0);
    EXPECT_DOUBLE_EQ(result[1][0], 3.0);
    EXPECT_DOUBLE_EQ(result[1][1], 4.0);
}
TEST(LayerTest, ConstructorWithValidInputs) { 
    std::vector<std::vector<double>> weights = {{1,2,3},{4,5,6}}; 
    std::vector<double> biases = {1,2}; 
    auto actFun = std::make_shared<ReLU>(); 
    Layer layer(weights, biases, actFun); 
    ASSERT_EQ(layer.get_weights(), weights); 
    ASSERT_EQ(layer.get_biases(), biases); 
    ASSERT_EQ(layer.get_actFun(), actFun); 
} 
TEST(LayerTest, LayerPass) {
    std::vector<std::vector<double>> weights = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<double> biases = {1.0, 2.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer layer(weights, biases, relu);
    std::vector<double> inputs = {0.5, -0.5};
    std::vector<double> result = layer.layer_pass(inputs);
    EXPECT_EQ(result.size(), 2);
    EXPECT_DOUBLE_EQ(result[0], 0.5);
    EXPECT_DOUBLE_EQ(result[1], 1.5);
}

TEST(LayerTest, SetWeightsPositiveTestCase) {
    std::vector<std::vector<double>> weights = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<double> biases = {1.0, 2.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer layer(weights, biases, relu);
    std::vector<std::vector<double>> newWeights {{1, 2}, {3, 4}};
    layer.set_weights(newWeights);
    // Check that the weights were set correctly
    EXPECT_EQ(layer.get_weights(), newWeights);
}
TEST(LayerTest, SetWeightsNegativeTestCase) {
    std::vector<std::vector<double>> weights = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<double> biases = {1.0, 2.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer layer(weights, biases, relu);
    std::vector<std::vector<double>> newWeights {{1, 2}, {3, 4}};
    // Try to set the weights with a matrix of different dimensions
    std::vector<std::vector<double>> wrongDimensions {{1, 2}, {3, 4}, {5, 6}};
    EXPECT_THROW(layer.set_weights(wrongDimensions), std::invalid_argument);
    // Check that the weights were not set to the wrong dimensions
    EXPECT_NE(layer.get_weights(), wrongDimensions);
}

TEST(LayerTest, GetBiases) {
    std::vector<std::vector<double>> weights = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<double> biases = {1.2, 3.4};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer layer(weights, biases, relu);
    std::vector<double> expected_biases = {1.2, 3.4};
    layer.set_biases(expected_biases);
    std::vector<double> actual_biases = layer.get_biases();
    ASSERT_EQ(actual_biases, expected_biases);
}

TEST(LayerTest, GetBiasesEmpty) {
    std::vector<std::vector<double>> weights = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<double> biases = {1.0, 2.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer layer(weights, biases, relu);
    std::vector<double> expected_biases = {};
     std::vector<double> actual_biases = layer.get_biases();
     ASSERT_NE(actual_biases, expected_biases);
}
TEST(LayerTest, SetBiasesPositive) {
    std::vector<std::vector<double>> weights = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<double> biases = {1.0, 2.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer layer(weights, biases, relu);
    std::vector<double> newBiases = {-0.5, 0.5};
    layer.set_biases(newBiases);
    EXPECT_EQ(layer.get_biases(), newBiases);
}
 TEST(LayerTest, SetBiasesNegative) {
    std::vector<std::vector<double>> weights = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<double> biases = {1.0, 2.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer layer(weights, biases, relu);
    std::vector<double> newBiases = {-0.5, 0.0, 2.0};  // wrong size
    EXPECT_THROW(layer.set_biases(newBiases), std::invalid_argument);
    EXPECT_NE(layer.get_biases(), newBiases);
}
TEST(LayerTest, ActivationDerivativeTest) {
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer layer({{1.0, 2.0}, {3.0, 4.0}}, {1.0, 1.0}, relu);
    std::vector<double> input = {0.5, -0.5};
    layer.layer_pass(input);
    std::vector<double> expectedDerivative = {1.0, 1.0};
    EXPECT_EQ(layer.get_activation_derivative(), expectedDerivative);
}


 // Test NeuralNetwork class
TEST(NeuralNetworkTest, ForwardPass) {
    std::vector<std::vector<double>> weights1 = {{1.0,-1.0}, {-1.0, 2.0}};
    std::vector<double> biases1 = {1.0, 1.0};
    std::vector<std::vector<double>> weights2 = {{1.0, -1.0}, {-1.0, 2.0}};
    std::vector<double> biases2 = {1.0, 0.5};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer layer1(weights1, biases1, relu);
    Layer layer2(weights2, biases2, relu);
    std::vector<Layer> layers = {layer1, layer2};
    NeuralNetwork nn(layers);
    std::vector<double> inputs = {0.5, -0.5};
    std::vector<double> result = nn.forwardPass(inputs);
    EXPECT_EQ(result.size(), 2);
    EXPECT_DOUBLE_EQ(result[0], 3);
    EXPECT_DOUBLE_EQ(result[1], 0);
}

 // Negative test cases
TEST(InitializeWeightsTest, MinBiggerThanMax) {
  EXPECT_THROW(initializeWeights(5, 5, 5.0, 2.0), std::invalid_argument);
}
TEST(LayerTest, InvalidWeightsSize) {
    std::vector<std::vector<double>> weights = {{1.0}, {2.0}};
    std::vector<double> biases = {1.0, 2.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    EXPECT_THROW(Layer layer(weights, biases, relu), std::invalid_argument);
}
TEST(LayerTest, InvalidInputsSize) {
    std::vector<std::vector<double>> weights = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<double> biases = {1.0, 2.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer layer(weights, biases, relu);
    std::vector<double> inputs = {0.5};
    EXPECT_THROW(layer.layer_pass(inputs), std::invalid_argument);
}
TEST(NeuralNetworkTest, InvalidLayersSize) {
    std::vector<std::vector<double>> weights1 = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<double> biases1 = {1.0, 2.0};
    std::vector<std::vector<double>> weights2 = {{5.0, 6.0, 7.0}, {8.0, 9.0, 10.0}};
    std::vector<double> biases2 = {1.0, 2.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer layer1(weights1, biases1, relu);
    Layer layer2(weights2, biases2, relu);
    std::vector<Layer> layers = {layer1, layer2};
    EXPECT_THROW(NeuralNetwork nn(layers), std::invalid_argument);
}

// Test computeLossGradient function
 TEST(NeuralNetworkTest, ComputeLossPositive) {
    std::vector<std::vector<double>> weights1 = {{1.0,-1.0}, {-1.0, 2.0}};
    std::vector<double> biases1 = {1.0, 1.0};
    std::vector<std::vector<double>> weights2 = {{1.0, -1.0}, {-1.0, 2.0}};
    std::vector<double> biases2 = {1.0, 0.5};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer layer1(weights1, biases1, relu);
    Layer layer2(weights2, biases2, relu);
    std::vector<Layer> layers = {layer1, layer2};
    NeuralNetwork net(layers);
    std::vector<double> input = {0.5, -0.5};
    std::vector<double> output = net.forwardPass(input);
    std::vector<double> target = {2.0, 1.0};
    std::vector<double> expectedLossGradient = {2.0, -2.0};
    std::vector<double> actualLossGradient = net.computeLossGradient(output, target);
    EXPECT_EQ(expectedLossGradient, actualLossGradient);
}
TEST(NeuralNetworkTest, ComputeLossNegative) {
    std::vector<std::vector<double>> weights1 = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<double> biases1 = {1.0, 2.0};
    std::vector<std::vector<double>> weights2 = {{5.0, 6.0}, {8.0, 9.0}};
    std::vector<double> biases2 = {1.0, 2.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer layer1(weights1, biases1, relu);
    Layer layer2(weights2, biases2, relu);
    std::vector<Layer> layers = {layer1, layer2};
    NeuralNetwork net(layers);
    std::vector<double> input = {1.0, 1.0};
    std::vector<double> output = net.forwardPass(input);
    std::vector<double> target = {1.0, 2.0, 3.0}; // Different size than output
    EXPECT_THROW(net.computeLossGradient(output, target), std::invalid_argument);
}

//  Test computeWeightGradients function
TEST(NeuralNetworkTest, ComputeWeightGradients_PositiveTest) {
    // Test delta input
    std::vector<double> delta = {0.1, 0.2, 0.3};
     // Test layer weights
    std::vector<std::vector<double>> weights = {{0.5, 0.3, 0.1}, {0.2, 0.4, 0.6}, {0.8, 0.9, 0.3}};
    std::vector<double> biases = {10.0, 0.0, 2.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer testLayer(weights, biases, relu);
    std::vector<Layer> layers = {testLayer};
    NeuralNetwork net(layers);
    // Test expected output
    std::vector<std::vector<double>> expectedGradients = {{0.05, 0.03, 0.01}, {0.04, 0.08, 0.12}, {0.24, 0.27, 0.09}};
    // Compute weight gradients
    std::vector<std::vector<double>> actualGradients = net.computeWeightGradients(delta, testLayer);
    // Check that actual gradients match expected gradients to within 1e-6 tolerance
    for (size_t i = 0; i < expectedGradients.size(); ++i) {
        for (size_t j = 0; j < expectedGradients[0].size(); ++j) {
            EXPECT_NEAR(actualGradients[i][j], expectedGradients[i][j], 1e-6);
        }
    }
}
TEST(NeuralNetworkTest, ComputeWeightGradients_NegativeTest) {
    // Test delta input with wrong size
    std::vector<double> delta = {0.1, 0.2, 0.3, 0.4};
    // Test layer weights
    std::vector<std::vector<double>> weights = {{0.5, 0.3, 0.1}, {0.2, 0.4, 0.6}, {0.8, 0.9, 0.3}};
    std::vector<double> biases = {0.0, 0.0, 2.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer testLayer(weights, biases, relu);
    std::vector<Layer> layers = {testLayer};
    NeuralNetwork net(layers);
    // Check that an invalid argument exception is thrown
    EXPECT_THROW(net.computeWeightGradients(delta, testLayer), std::invalid_argument);
}
//  Test computeBiasGradients function
TEST(NeuralNetworkTest, ComputeBiasGradients) {
    std::vector<std::vector<double>> weights = {{0.5, 0.3, 0.1}, {0.2, 0.4, 0.6}, {0.8, 0.9, 0.3}};
    std::vector<double> biases = {5.0, 0.0, 1.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer testLayer(weights, biases, relu);
    std::vector<Layer> layers = {testLayer};
    NeuralNetwork net(layers);
    std::vector<double> delta = {0.1, -0.2, 0.3};
    std::vector<double> expected_output = {0.1, -0.2, 0.3};
    std::vector<double> output = net.computeBiasGradients(delta);
    // Check that the output is the expected size
    ASSERT_EQ(output.size(), expected_output.size());
    // Check that each element of the output is equal to the expected output
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_EQ(output[i], expected_output[i]);
    }
}
// Test the updateWeightsAndBiases function
TEST(NeuralNetworkTest, UpdateWeightsAndBiases_ValidInput) {
    std::vector<std::vector<double>> weights = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}};
    std::vector<double> biases = {1.0, 2.0, 3.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer testLayer(weights, biases, relu);
    std::vector<Layer> layers = {testLayer};
    NeuralNetwork net(layers);
    std::vector<std::vector<double>> weightGradients = {{2, 4, 6}, {8, 10, 12}, {14, 16, 18}};
    std::vector<double> biasGradients = {3, 6, 9};
    double learningRate = 0.1;
    net.updateWeightsAndBiases(testLayer, weightGradients, biasGradients, learningRate);
    std::vector<std::vector<double>> expected_weights = {{-0.1, -0.2, -0.3}, {-0.4, -0.5, -0.6}, {-0.7, -0.8, -0.9}};
    std::vector<double> expected_biases = {0.7, 1.4, 2.1};
    for (size_t i = 0; i < expected_weights.size(); ++i) {
        for (size_t j = 0; j < expected_weights[0].size(); ++j) {
            EXPECT_NEAR(testLayer.get_weights()[i][j], expected_weights[i][j], 1e-6);
        }
    }
    EXPECT_EQ(testLayer.get_biases(), expected_biases);
}
TEST(NeuralNetworkTest, updateWeightsAndBiasesNegativeTest) {
    std::vector<std::vector<double>> weights = {{0.5, 0.3, 0.1}, {0.2, 0.4, 0.6}, {0.8, 0.9, 0.3}};
    std::vector<double> biases = {5.0, 0.0, 1.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer testLayer(weights, biases, relu);
    std::vector<Layer> layers = {testLayer};
    NeuralNetwork net(layers);
    // Set up test input
    std::vector<double> input{ 1.0, 2.0, 3.0 };
    // Set up test gradients with incorrect dimensions
    std::vector<std::vector<double>> weightGradients{ {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6} };
    std::vector<double> biasGradients{ 0.1 };
    // Call updateWeightsAndBiases function with incorrect dimensions of gradients
    EXPECT_THROW(net.updateWeightsAndBiases(testLayer, weightGradients, biasGradients, 0.1), std::invalid_argument);
}
// Test the computeDelta function
TEST(NeuralNetworkTest, ComputeDeltaTest) {
    std::vector<std::vector<double>> weights = {{0.5, 0.3}, {0.2, 0.4}};
    std::vector<double> biases = {1.0, -1.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer testLayer(weights, biases, relu);
    testLayer.layer_pass({1.0, 2.0});
    std::vector<Layer> layers = {testLayer};
    NeuralNetwork net(layers);
    std::vector<double> delta = {1.0, 2.0};
    std::vector<double> previousDelta = net.computeDelta(delta, testLayer);
    ASSERT_EQ(previousDelta.size(), 2);
    EXPECT_DOUBLE_EQ(previousDelta[0], 0.9);
    EXPECT_DOUBLE_EQ(previousDelta[1], 0);
}
TEST(NeuralNetworkTest, ComputeDeltaNegativeTest) {
    std::vector<std::vector<double>> weights = {{0.5, 0.3, 0.1}, {0.2, 0.4, 0.6}, {0.8, 0.9, 0.3}};
    std::vector<double> biases = {5.0, 0.0, 1.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer testLayer(weights, biases, relu);
    std::vector<Layer> layers = {testLayer};
    NeuralNetwork net(layers);
    std::vector<double> delta = {1, 0}; // Delta size does not match the number of output neurons
    ASSERT_THROW(net.computeDelta(delta, testLayer), std::invalid_argument);
}



// Tests for Optimizer.h
// ---------------------------------------------------------------------------------------------
// Test for getTrainBatch
TEST(OptimizerTest, getTrainBatchTest) {
    std::vector<std::vector<double>> trainData = {{1, 2}, {4, 5}, {7, 8}};
    std::vector<double> trainLabels = {3, 9, 15};

    std::vector<std::vector<double>> valData = {{-3, 4}, {2, 5}, {6, 3}};
    std::vector<double> valLabels = {1, 7, 9};

    std::vector<std::vector<double>> testData = {{10, 11}, {13, 14}};
    std::vector<double> testLabels = {21, 27};
    int batchSize = 2;

    std::vector<std::vector<double>> expectedTrainBatchData = {{1, 2}, {4, 5}};
    std::vector<double> expectedTrainBatchLabels = {3, 9};

    NeuralNetwork nn({Layer(2, 2, std::make_shared<ReLU>())});
    Optimizer<std::vector<double>, double> optimizer(nn, trainData, trainLabels, testData, testLabels, "SGD", 0.01, 2, 2);
    auto result = optimizer.getTrainBatch(0);
    EXPECT_EQ(result.first, expectedTrainBatchData);
    EXPECT_EQ(result.second, expectedTrainBatchLabels);
}
// Test for getValidation 
TEST(OptimizerTest, getValidationTest) {
    std::vector<std::vector<double>> trainData = {{1, 2}, {4, 5}, {7, 8}};
    std::vector<double> trainLabels = {3, 9, 15};

    std::vector<std::vector<double>> valData = {{-3, 4}, {2, 5}, {6, 3}};
    std::vector<double> valLabels = {1, 7, 9};

    std::vector<std::vector<double>> testData = {{10, 11}, {13, 14}};
    std::vector<double> testLabels = {21, 27};
    int batchSize = 2;

    std::vector<std::vector<double>> expectedValData = {{-3, 4}, {2, 5}, {6, 3}};
    std::vector<double> expectedValLabels = {1, 7, 9};

    NeuralNetwork nn({Layer(2, 2, std::make_shared<ReLU>())});
    Optimizer<std::vector<double>, double> optimizer(nn, trainData, trainLabels, testData, testLabels, valData, valLabels, "SGD", 0.01, 2, 2);
    auto result = optimizer.getValidation();
    EXPECT_EQ(result.first, expectedValData);
    EXPECT_EQ(result.second, expectedValLabels);
}
// Test for getTest
TEST(OptimizerTest, getTestTest) {
    std::vector<std::vector<double>> trainData = {{1, 2}, {4, 5}, {7, 8}};
    std::vector<double> trainLabels = {3, 9, 15};

    std::vector<std::vector<double>> valData = {{-3, 4}, {2, 5}, {6, 3}};
    std::vector<double> valLabels = {1, 7, 9};

    std::vector<std::vector<double>> testData = {{10, 11}, {13, 14}};
    std::vector<double> testLabels = {21, 27};
    int batchSize = 2;

    std::vector<std::vector<double>> expectedTestData = {{10, 11}, {13, 14}};
    std::vector<double> expectedTestLabels = {21, 27};

    NeuralNetwork nn({Layer(2, 2, std::make_shared<ReLU>())});
    Optimizer<std::vector<double>, double> optimizer(nn, trainData, trainLabels, testData, testLabels, valData, valLabels, "SGD", 0.01, 2, 2);
    auto result = optimizer.getTest();
    EXPECT_EQ(result.first, expectedTestData);
    EXPECT_EQ(result.second, expectedTestLabels);
}
// Test for shuffleTrainData
TEST(OptimizerTest, shuffleTrainData) { 
    std::vector<std::vector<double>> trainData = {{1, 2}, {4, 5}, {7, 8}};
    std::vector<double> trainLabels = {3, 9, 15};

    std::vector<std::vector<double>> testData = {{10, 11}, {13, 14}};
    std::vector<double> testLabels = {21, 27};
    int batchSize = 2;
    NeuralNetwork nn({Layer(2, 2, std::make_shared<ReLU>())});
    Optimizer<std::vector<double>, double> optimizer(nn, trainData, trainLabels, testData, testLabels, "SGD", 0.01, 2, 2);
    
    std::vector<int> preShuffledInd = optimizer.getTrainingInd();
    optimizer.shuffleTrainData(); 
    std::vector<int> shuffledInd = optimizer.getTrainingInd();
    EXPECT_NE(shuffledInd, preShuffledInd);
} 

// Test for singleTrain 
TEST(OptimizerTest, singleTrainTest) {
    std::vector<std::vector<double>> trainData = {{0, 0}, {0, 0}, {0, 0}};
    std::vector<std::vector<double>> trainLabels = {{1,0}, {0,1}, {1, 0}};

    std::vector<std::vector<double>> testData = {{10, 11}, {13, 14}};
    std::vector<std::vector<double>> testLabels = {{1, 0}, {0, 1}};
    int batchSize = 2;

    std::vector<std::vector<double>> weights = {{1.0,-1.0}, {-1.0, 2.0}};
    std::vector<double> biases = {0.0, 0.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer layer(weights, biases, relu);
    std::vector<Layer> layers = {layer};
    NeuralNetwork nn(layers);

    Optimizer<std::vector<double>, std::vector<double>> optimizer(nn, trainData, trainLabels, testData, testLabels, "SGD", 0.01, 2, 2);
    std::vector<double> loss = optimizer.singleTrain();
    EXPECT_EQ(loss.size(), 1);
    EXPECT_NEAR(loss[0], 0.5, 1e-6);
}

// Test for singleTest
TEST(OptimizerTest, singleTestTest) {
    std::vector<std::vector<double>> trainData = {{0, 0}, {0, 0}, {0, 0}};
    std::vector<std::vector<double>> trainLabels = {{1,0}, {0,1}, {1, 0}};

    std::vector<std::vector<double>> testData = {{0, 0}, {0, 0}};
    std::vector<std::vector<double>> testLabels = {{1, 0}, {0, 1}};
    int batchSize = 2;

    std::vector<std::vector<double>> weights = {{1.0,-1.0}, {-1.0, 2.0}};
    std::vector<double> biases = {0.0, 0.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer layer(weights, biases, relu);
    std::vector<Layer> layers = {layer};
    NeuralNetwork nn(layers);

    Optimizer<std::vector<double>, std::vector<double>> optimizer(nn, trainData, trainLabels, testData, testLabels, "SGD", 0.01, 2, 2);
    double loss = optimizer.singleTest();
    EXPECT_NEAR(loss, 0.5, 1e-6);
}

// Test for train
TEST(OptimizerTest, trainTest) {
    std::vector<std::vector<double>> trainData = {{0, 0}, {0, 0}, {0, 0}};
    std::vector<std::vector<double>> trainLabels = {{1,0}, {0,1}, {1, 0}};

    std::vector<std::vector<double>> testData = {{0, 0}, {0, 0}};
    std::vector<std::vector<double>> testLabels = {{1, 0}, {0, 1}};
    int batchSize = 2;

    std::vector<std::vector<double>> weights = {{1.0,-1.0}, {-1.0, 2.0}};
    std::vector<double> biases = {0.0, 0.0};
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLU>();
    Layer layer(weights, biases, relu);
    std::vector<Layer> layers = {layer};
    NeuralNetwork nn(layers);

    Optimizer<std::vector<double>, std::vector<double>> optimizer(nn, trainData, trainLabels, testData, testLabels, "SGD", 0.01, 2, 2);
    optimizer.train();
    // EXPECT_EQ(loss.size(), 1);
    // EXPECT_NEAR(loss[0], 0.5, 1e-6);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}