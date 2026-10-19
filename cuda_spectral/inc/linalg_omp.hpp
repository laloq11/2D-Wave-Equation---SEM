#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "types.hpp"


namespace omp {


template <typename ValueType>
void dot(const ValueType* a, const ValueType* b, const size_type length,
         ValueType* res);


template <typename ValueType>
void axpy(const ValueType* alpha, const ValueType* x, const size_type length,
          ValueType* y);


template <typename ValueType>
void matvec(const size_type num_rows, const size_type num_cols,
            const ValueType* A, const ValueType* x, ValueType* b);


template <typename ValueType>
void matmul(const size_type num_rows, const size_type num_cols,
            const ValueType* A, const ValueType* B, ValueType* C);


}  // namespace omp
