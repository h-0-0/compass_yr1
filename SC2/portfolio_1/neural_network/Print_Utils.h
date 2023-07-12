#ifndef PRINT_UTILS_H
#define PRINT_UTILS_H

#include <iostream>
#include <vector>

template <typename T>
void printVector(const std::vector<T>& vec);

template <typename T>
void printArray(const T* arr, size_t size);

template <typename T, size_t N>
void printArray(const T (&arr)[N]);

template <typename T>
void printMatrix(const std::vector<std::vector<T>>& matrix);

#include "Print_Utils.inl"

#endif // PRINT_UTILS_H
