// Utils.h
#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <type_traits>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

// Function for testing if an object is a vector
template<typename T>
bool isVector(const std::vector<T>& object);

// Function for testing if an object is a matrix
template<typename T>
bool isMatrix(const std::vector<std::vector<T>>& object);

std::vector<std::vector<double>> matrixAddition(const std::vector<std::vector<double>>& matrix1, const std::vector<std::vector<double>>& matrix2);

std::vector<double> vectorAddition(const std::vector<double>& vector1, const std::vector<double>& vector2);

void matrixScalarMultiplication(std::vector<std::vector<double>>& matrix, double scalar);

void vectorScalarMultiplication(std::vector<double>& vector, double scalar);

#include "Utils.inl"

#endif // UTILS_H
