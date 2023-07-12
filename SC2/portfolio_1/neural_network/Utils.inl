template<typename T>
bool isVector(const std::vector<T>& object) {
    // Check if the object is a one-dimensional vector
    if (object.empty()) {
        return false;
    }

    // Check if none of the elements are vectors
    for (const auto& element : object) {
        if (std::is_same<decltype(element), std::vector<T>>::value) {
            return false;
        }
    }

    // Check it has more than one element
    if (object.size() == 1) {
        return false;
    }

    return true;
}

template<typename T>
bool isMatrix(const std::vector<std::vector<T>>& object) {
    // Check if the object is a vector of vectors
    if (object.empty()) {
        return false;
    }

    // Get the number of columns in the first row
    size_t numColumns = object[0].size();

    // Check if all rows have the same number of columns
    for (const auto& row : object) {
        if (row.size() != numColumns) {
            return false;
        }
    }

    // Check the dimensions of the weight matrix
    size_t numInputs = object.empty() ? 0 : object[0].size(); // Number of input nodes
    if (numInputs == 1) {
        return false;
    }

    return true;
}

std::vector<std::vector<double>> matrixAddition(const std::vector<std::vector<double>>& matrix1, const std::vector<std::vector<double>>& matrix2) {
    size_t rows = matrix1.size();
    size_t cols = matrix1[0].size();

    std::vector<std::vector<double>> result(rows, std::vector<double>(cols));

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }

    return result;
}

std::vector<double> vectorAddition(const std::vector<double>& vector1, const std::vector<double>& vector2) {
    size_t size = vector1.size();
    std::vector<double> result(size);

    for (size_t i = 0; i < size; i++) {
        result[i] = vector1[i] + vector2[i];
    }

    return result;
}

void matrixScalarMultiplication(std::vector<std::vector<double>>& matrix, double scalar) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            matrix[i][j] *= scalar;
        }
    }
}

void vectorScalarMultiplication(std::vector<double>& vector, double scalar) {
    size_t size = vector.size();

    for (size_t i = 0; i < size; i++) {
        vector[i] *= scalar;
    }
}
