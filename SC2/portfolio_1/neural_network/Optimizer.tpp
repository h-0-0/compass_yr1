// Optimizer constructor and destructor
// ----------------------------------------------------------------------

// Constructor with validation data
template <typename DT, typename LT>
Optimizer<DT, LT>::Optimizer(NeuralNetwork& nn,
        const std::vector<DT>& trainingData,
        const std::vector<LT>& trainingLabels,
        const std::vector<DT>& testData,
        const std::vector<LT>& testLabels,
        const std::vector<DT>& validationData,
        const std::vector<LT>& validationLabels,
        const std::string opt_method,
        const double learningRate,
        const int batchSize,
        const int numEpochs
    ) : nn(nn),
        trainingData(trainingData),
        trainingLabels(trainingLabels),
        testData(testData),
        testLabels(testLabels),
        validationData(validationData),
        validationLabels(validationLabels),
        opt_method(opt_method),
        learningRate(learningRate),
        batchSize(batchSize),
        numEpochs(numEpochs),
        numBatches(trainingData.size() / batchSize),
        trainingInd(trainingData.size())
    {
        if (trainingData.size() != trainingLabels.size()) {
            throw std::invalid_argument("Training data and labels must be the same size");
        }
        if (testData.size() != testLabels.size()) {
            throw std::invalid_argument("Test data and labels must be the same size");
        }
        if (validationData.size() != validationLabels.size()) {
            throw std::invalid_argument("Validation data and labels must be the same size");
        }
        std::iota(trainingInd.begin(), trainingInd.end(), 0);
};

// Constructor without validation data
template <typename DT, typename LT>
Optimizer<DT, LT>::Optimizer(NeuralNetwork& nn,
        const std::vector<DT>& trainingData,
        const std::vector<LT>& trainingLabels,
        const std::vector<DT>& testData,
        const std::vector<LT>& testLabels,
        const std::string opt_method,
        const double learningRate,
        const int batchSize,
        const int numEpochs
    ) : nn(nn),
        trainingData(trainingData),
        trainingLabels(trainingLabels),
        testData(testData),
        testLabels(testLabels),
        opt_method(opt_method),
        learningRate(learningRate),
        batchSize(batchSize),
        numEpochs(numEpochs),
        numBatches(trainingData.size() / batchSize),
        trainingInd(trainingData.size())
    {
        if (trainingData.size() != trainingLabels.size()) {
            throw std::invalid_argument("Training data and labels must be the same size");
        }
        if (testData.size() != testLabels.size()) {
            throw std::invalid_argument("Test data and labels must be the same size");
        }
        std::iota(trainingInd.begin(), trainingInd.end(), 0);
};

// Destructor
template <typename DT, typename LT>
Optimizer<DT, LT>::~Optimizer() {};


// Optimizer member functions
// ----------------------------------------------------------------------

// Gets a batch of data and labels from the training data
template <typename DT, typename LT>
std::pair<std::vector<DT>, std::vector<LT>> Optimizer<DT, LT>::getTrainBatch(const int& batchNum) const {
    if (batchNum >= numBatches) {
        throw std::invalid_argument("Batch number must be less than the number of batches");
    }
    std::vector<DT> batchData(batchSize);
    std::vector<LT> batchLabels(batchSize);
    for (int i = 0; i < batchSize; i++) {
        batchData[i] = trainingData[trainingInd[batchNum * batchSize + i]];
        batchLabels[i] = trainingLabels[trainingInd[batchNum * batchSize + i]];
    }
    return std::make_pair(batchData, batchLabels);
};

// Gets all the training data
template <typename DT, typename LT>
std::pair<std::vector<DT>, std::vector<LT>> Optimizer<DT, LT>::getTrainData() const {
    return std::make_pair(trainingData, trainingLabels);
};

// Gets all validation data and labels
template <typename DT, typename LT>
std::pair<std::vector<DT>, std::vector<LT>> Optimizer<DT, LT>::getValidation() const {
    return std::make_pair(validationData, validationLabels);
};

// Gets a testing data and labels
template <typename DT, typename LT>
std::pair<std::vector<DT>, std::vector<LT>> Optimizer<DT, LT>::getTest() const {
    return std::make_pair(testData, testLabels);
};

// Shuffles the training data and labels
template <typename DT, typename LT>
void Optimizer<DT, LT>::shuffleTrainData() {
    std::random_device rd;  // Obtain a random seed from the hardware
    std::mt19937 eng(rd()); // Seed the random number engine
    std::shuffle(trainingInd.begin(), trainingInd.end(), eng); // Shuffle the vector of indices
};

// Performs one epochs worth of training
template <typename DT, typename LT>
std::vector<double> Optimizer<DT, LT>::singleTrain() {
    // Shuffle the indices we use to access training data
    shuffleTrainData();
    // Container for the average training loss for each batch
    std::vector<double> avg_train_loss(numBatches);
    // Perform one epoch of training
    for(int i = 0 ; i < numBatches ; ++i){
        std::cout << "Batch " << i << std::endl;
        // Get a batch of data and labels
        std::pair<std::vector<DT>, std::vector<LT>> batch = getTrainBatch(i);

        // Carry out mini-batch SGD on the batch
        avg_train_loss[i] = nn.backpropagation(batch.first, batch.second, learningRate);
    }
    return avg_train_loss;
};

// Performs a single testing step, ie. forward pass and loss calculation for whole test set
template <typename DT, typename LT>
double Optimizer<DT, LT>::singleTest(){
    double loss = 0.0;
    for(int i = 0 ; i < testData.size() ; ++i){
        // Carry out forward pass on the batch
        std::vector<double> out = nn.forwardPass(testData[i]);
        loss += computeMSE(out, testLabels[i]);
    }
    std::cout << "Test loss: " << loss << std::endl;
    return loss / testData.size();
};

// Iteratively conducts training step for every batch and for number of specified epochs
template <typename DT, typename LT>
void Optimizer<DT, LT>::train(){
    // Container for the average training loss for each batch
    std::vector<std::vector<double>> avg_train_loss(numEpochs, std::vector<double>(numBatches));
    // Container for the average test loss for each epoch
    std::vector<double> avg_test_loss(numEpochs);
    // Perform the specified number of epochs
    for(int i = 0 ; i < numEpochs ; ++i){
        std::cout << "Epoch " << i << std::endl;
        // Perform one epoch of training
        avg_train_loss[i] = singleTrain();
        // Perform one epoch of testing
        avg_test_loss[i] = singleTest();
    }
    // Print the average training loss for each batch
    std::cout << "Average training loss for each batch:" << std::endl;
    printMatrix(avg_train_loss);
    // Print the average test loss for each epoch
    std::cout << "Average test loss for each epoch:" << std::endl;
    printVector(avg_test_loss);
};

// Returns the neural network
template <typename DT, typename LT>
NeuralNetwork Optimizer<DT, LT>::getNn() const {
    return nn;
};

// Returns the training indices
template <typename DT, typename LT>
std::vector<int> Optimizer<DT, LT>::getTrainingInd() const {
    return trainingInd;
};

