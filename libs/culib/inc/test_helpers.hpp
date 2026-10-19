#pragma once

#include <cmath>
#include <exception>
#include <utility>
#include <vector>
#include <cmath>

#include <stdio.h>

#define ASSERT_TRUE(__func)                                               \
    {                                                                     \
        if (!__func) {                                                    \
            fprintf(stderr, "Error!: %d, File: %s, Line number: %d\n", 1, \
                    __FILE__, __LINE__);                                  \
            exit(1);                                                      \
        }                                                                 \
    }


template <typename ValueType>
auto compare_arrays(int length, const ValueType* arr1, const ValueType* arr2)
{
    for (int i = 0; i < length; ++i) {
        if (arr1[i] != arr2[i]) return std::pair<bool, int>{false, i};
    }
    return std::pair<bool, int>{true, -1};
}

template <typename ValueType>
auto compare_vectors(const std::vector<ValueType>& vec1,
                     const std::vector<ValueType>& vec2, double tolerance)
{
    ASSERT_TRUE(vec1.size() == vec2.size());
    // ASSERT_TRUE(vec1.size() == vec2.size());

    for (int r = 0; r < vec1.size(); r++) {
        if (std::abs(vec1[r] - vec2[r]) > tolerance) {
            return std::pair<bool, int>{false, r};
        }
    }
    return std::pair<bool, int>{true, -1};
}

template <typename ValueType>
auto compare_matrices(const std::vector<ValueType>& vec1,
    const std::vector<ValueType>& vec2, 
    const int matrix_size,//squre matrices only
    double tolerance)
{
    ASSERT_TRUE(vec1.size() == vec2.size());
    // ASSERT_TRUE(vec1.size() == vec2.size());

    for (int r = 0; r < matrix_size; r++) {
        for (int s = 0; s < matrix_size; s++) {
            if (std::abs(vec1[r*matrix_size+s] - vec2[r*matrix_size+s]) > tolerance) {
                printf("Expected:%g\n",(double)vec1[r*matrix_size+s]);
                printf("Obtained:%g\n",(double)vec2[r*matrix_size+s]);
                return std::pair<bool, int>{false, r*matrix_size+s};
            }
        }
    }
    return std::pair<bool, int>{true, -1};
}