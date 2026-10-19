#pragma once
#ifndef OPS_HPP
#define OPS_HPP
#include <iostream>
#include <string>
#include <vector>

#include <omp.h>

#include "types.hpp"

#include <stdio.h>

namespace omp_namespace {


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
// namespace omp_namespace {
//     template void axpy<double>(const double* x, const double* y, unsigned long size, double* result);
//     template void matvec<double>(unsigned long rows, unsigned long cols, const double* matrix, const double* vector, double* result);
//     template void dot<double>(const double* x, const double* y, unsigned long size, double* result);
//     template void matmul<double>(unsigned long rows, unsigned long cols, const double* matrix1, const double* matrix2, double* result);
// }
#endif // OPS_HPP
