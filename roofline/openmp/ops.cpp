#include <iostream>
#include <string>
#include <vector>

#include <omp.h>

#include "types.hpp"

#include <stdio.h>

#include "ops.hpp"

namespace omp_namespace {


template <typename ValueType>
void axpy(const ValueType* alpha, const ValueType* x, const size_type length,
          ValueType* y)
{
    // std::cout<<"just doing my jooooooooooooooooooooooooooooob"<<std::endl;
#pragma omp parallel for
    for (int i = 0; i < length; i++) {
        y[i] += alpha[0] * x[i];
    }
}


template <typename ValueType>
void dot(const ValueType* a, const ValueType* b, const size_type length,
         ValueType* res)
{
    ValueType temp = 0.0;
#pragma omp declare reduction(sum:ValueType : omp_out += omp_in)
#pragma omp parallel for reduction(sum : temp)
    for (size_type i = 0; i < length; i++) {
        temp += a[i] * b[i];
    }
    res[0] = temp;
}


template <typename ValueType>
void matvec(const size_type num_rows, const size_type num_cols,
            const ValueType* A, const ValueType* x, ValueType* b)
{
    // std::cout<<"just doing my jooooooooooooooooooooooooooooob"<<std::endl;

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        b[row] = 0.0;
        for (size_type col = 0; col < num_cols; ++col) {
            b[row] += A[row * num_cols + col] * x[col];
        }
    }
}


template <typename ValueType>
void matmul(const size_type num_rows, const size_type num_cols,
            const ValueType* A, const ValueType* B, ValueType* C)
{
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type i = 0; i < num_cols; ++i) {
            for (size_type col = 0; col < num_cols; ++col) {
                C[row * num_rows + col] +=
                    A[row * num_cols + i] * B[i * num_rows + col];
            }
        }
    }
}


// #define DECLARE_DOT(_vtype)                                              \
//     void dot(const _vtype* a, const _vtype* b, const std::size_t length, \
//              _vtype* res);

// INSTANTIATE_FOR_EACH_POD_TYPE(DECLARE_DOT);


// #define DECLARE_AXPY(_vtype)                                                  \
//     void axpy(const _vtype* alpha, const _vtype* x, const std::size_t length, \
//               _vtype* y);

// INSTANTIATE_FOR_EACH_POD_TYPE(DECLARE_AXPY);


// #define DECLARE_MATVEC(_vtype)                                      \
//     void matvec(const size_type num_rows, const size_type num_cols, \
//                 const _vtype* A, const _vtype* x, _vtype* b);

// INSTANTIATE_FOR_EACH_POD_TYPE(DECLARE_MATVEC);


// #define DECLARE_MATMUL(_vtype)                                      \
//     void matmul(const size_type num_rows, const size_type num_cols, \
//                 const _vtype* A, const _vtype* B, _vtype* C)

// INSTANTIATE_FOR_EACH_POD_TYPE(DECLARE_MATMUL);

#define DECLARE_DOT(_vtype)                                              \
    void dot(const _vtype* a, const _vtype* b, const std::size_t length, \
             _vtype* res);

INSTANTIATE_FOR_EACH_POD_TYPE(DECLARE_DOT);


#define DECLARE_AXPY(_vtype)                                                  \
    void axpy(const _vtype* alpha, const _vtype* x, const std::size_t length, \
              _vtype* y);

INSTANTIATE_FOR_EACH_POD_TYPE(DECLARE_AXPY);


#define DECLARE_MATVEC(_vtype)                                      \
    void matvec(const size_type num_rows, const size_type num_cols, \
                const _vtype* A, const _vtype* x, _vtype* b);

INSTANTIATE_FOR_EACH_POD_TYPE(DECLARE_MATVEC);


#define DECLARE_MATMUL(_vtype)                                      \
    void matmul(const size_type num_rows, const size_type num_cols, \
                const _vtype* A, const _vtype* B, _vtype* C)

INSTANTIATE_FOR_EACH_POD_TYPE(DECLARE_MATMUL);

}  // namespace omp
