#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <array>
#include <functional>

#include "types.hpp"
#include <stdio.h>

#include "linalg.hpp"
#include "alias.hpp"
#include "params.hpp"
#include "spectralcm.hpp"

#include <omp.h>


#include "alias.hpp"

#include "LagrangeBasisFunctions.hpp"
#include "GaussLobatto.hpp" 
#include "basisfunctions.hpp" 

#ifndef CUBASE_HPP
#define CUBASE_HPP

namespace cie {
namespace otFlyGPU {

// CPU reference implementations: used only to check the GPU results
// (tests and roofline verify). The GPU solver never calls them.
namespace reference {

template <typename ValueType>
void compute_fku_ref(const size_t it,const Params& params,
    const std::vector<ValueType>& u_tm1, std::vector<ValueType>& fku);

template <typename ValueType>
void compute_Mass_ref(std::vector<ValueType>& mu,
    const Params& params);

template <typename ValueType>
void alg_84_flat(const size_t D, const ValueType* B, 
    ValueType* Bdiff, 
    const ValueType* M, 
    const ValueType* v,size_t pD, ValueType* r);

template <typename ValueType>
void alg_84_mergeBd_flat(ValueType* Bd_diff, const ValueType* Bdiff,
        const size_t t, size_t row, size_t col, size_t dim);
template <typename ValueType>
void alg_82_matrices(const size_t D, const std::vector<Matrixt<ValueType>>& B,
    const std::vector<ValueType>& G, 
    std::vector<ValueType>& v);

template <typename ValueType>
void alg_82_flat(const size_t D, 
                    ValueType* B,
                    ValueType* G,
                    ValueType* v, 
                    size_t pD);

                    template <typename ValueType>
                    void alg_57_matrices(const size_t D, 
                        const std::vector<Matrixt<ValueType>>& R, 
                        std::vector<ValueType>& P);

template <typename ValueType>
void alg_57_flat(const size_t D,
                    ValueType* R,
                    ValueType* P,size_t pD, size_t p_size);

std::vector<size_t> createLocationMapcu(size_t polynomialDegree,
    size_t ndofdirection1,
    size_t ielement,
    size_t jelement);
std::vector<std::vector<size_t>> createLocationMapVector(Params params);

template <typename ValueType>
void get_elementforce(ValueType* Fe,const ValueType* globalF,
     const size_t* locationM, const size_t ndofelement);

template <typename ValueType>
std::vector<Matrixt<ValueType>> swapaxes_matrices(const std::vector<Matrixt<ValueType>>& A);

template <typename ValueType>
void swapaxes_matrices(std::vector<Matrixt<ValueType>>& A);

template <typename ValueType>
ValueType* swapaxes_flat(const ValueType* A,size_t dim, size_t row, size_t col);

} // namespace reference




template <typename ValueType>
std::vector<ValueType> kroncu(const std::vector<double>& a, 
                                const std::vector<double>& b, 
                                    ValueType factor);
template <typename ValueType>
void kron_flat(ValueType* G,const std::vector<double>& a, 
                const std::vector<double>& b, 
                ValueType factor,
                size_t m);   

std::vector<size_t> arangecu(size_t number,size_t addition);

// void createLocationMapcu2(size_t* lM,size_t polynomialDegree, 
//     size_t ndofdirection1, 
//     size_t ielement, 
//     size_t jelement);
// void createLocationMapVector2(size_t**lMVector,Params params);

template <typename ValueType>
void slicecu(ValueType* u_slice,
    const ValueType* u, 
    const std::vector<size_t>& indices, size_t ndofelement);

template <typename ValueType>
void genVV(Matrixt<ValueType>& Vec, size_t n, size_t m);

template <typename ValueType>
void genVV2(ValueType*** matrix, size_t n, size_t m);

template <typename ValueType>
std::vector<ValueType> matvec(Matrixt<ValueType>& Mat, 
    std::vector<ValueType>& vector);

template <typename ValueType>
void matvec2(ValueType* result, ValueType* Mat, ValueType* vector, size_t r, size_t c);

template <typename ValueType>
inline void Vectorwise_Mult_helpers_vector(const std::vector<ValueType>& R,
                                 std::vector<ValueType>& s) {
    for (size_t i = 0; i < R.size(); ++i) { s[i] *= R[i]; }
}

template <typename ValueType>
inline void Vectorwise_Mult_helpers_flat(const ValueType* R,
                                 ValueType* s,size_t ssize) {   
    for (size_t i = 0; i < ssize; ++i) { s[i] *= R[i]; }
}
template <typename ValueType>
inline void Vectorwise_sum_helpers_flat(const ValueType* R,
                                 ValueType* s,size_t ssize) {   
    for (size_t i = 0; i < ssize; ++i) { s[i] += R[i]; }
}
inline std::vector<Matrix> VectorMatrix_helpers(size_t sv, int Msize1, int Msize2) {
    /*
    sv: # of matrices
    Msize1: # of rows in matrices
    Msize2: # of columns in matrices
    */
    std::vector<Matrix> Mout;
    for (size_t m = 0; m < sv; ++m) {
        Matrix innerM(Msize1, Msize2);
        Mout.push_back(innerM);
    }
    return Mout;
}


template <typename ValueType>
Matrixt<ValueType> get_ShapesDiffcu(const std::vector<double>& coordinates,
                        const size_t diff);

template <typename ValueType>
void ShapeFunctionFactor(Matrixt<ValueType>& M, const ValueType f);

template <typename ValueType>
std::vector<Matrixt<ValueType>> concatenate_matrices(const Matrixt<ValueType>& A, 
                                    const Matrixt<ValueType>& C);
template <typename ValueType>
void concatenate_flat(ValueType* B, const Matrixt<ValueType>& A, 
                                    const Matrixt<ValueType>& C, 
                                    size_t rows, size_t dim);
void example1();

template <typename ValueType>
void hellocudabase(ValueType num);


// }
// namespace linalg{
template <typename ValueType>
ValueType** createMatrix(std::initializer_list<std::initializer_list<ValueType>> list) {
    size_t rows = list.size();
    size_t cols = list.begin()->size(); // Assume all rows have the same size

    ValueType** matrix = new ValueType*[rows];
    for (size_t i = 0; i < rows; ++i) {
        matrix[i] = new ValueType[cols];
    }

    size_t row_index = 0;
    for (const auto& row : list) {
        size_t col_index = 0;
        for (const auto& val : row) {
            matrix[row_index][col_index] = val;
            col_index++;
        }
        row_index++;
    }

    return matrix;
}
template float** createMatrix<float>(std::initializer_list<std::initializer_list<float>>);
template double** createMatrix<double>(std::initializer_list<std::initializer_list<double>>);


template <typename ValueType>
void deleteMatrix(ValueType** matrix, size_t rows) {
    if (matrix) {
        for (size_t i = 0; i < rows; ++i) {
            delete[] matrix[i];
        }
        delete[] matrix;
    }
}
template void deleteMatrix<float>(float**, size_t);
template void deleteMatrix<double>(double**, size_t);


template <typename ValueType>
void getPfraction(ValueType* Pfraction,const ValueType* P,
                            size_t j, size_t P_size,  size_t n_i) {

    // Ensure that the range is valid.
    if (j * n_i >= P_size || (j + 1) * n_i > P_size) {
        Pfraction = nullptr;
    }
    else {
        std::copy(P + j * n_i, P + (j + 1) * n_i, Pfraction);
    }
}
template void getPfraction<float>(float*, const float*, size_t,size_t,size_t);
template void getPfraction<double>(double*, const double*, size_t,size_t,size_t);

template <typename ValueType>
void getRfraction(ValueType* Rfraction, ValueType* R, 
                size_t dim,size_t i, size_t n_i) {
    for (size_t Ri = 0; Ri < n_i; Ri++) {
        for (size_t Rj = 0; Rj < n_i; Rj++) {
            Rfraction[Ri*n_i+Rj] = R[Ri*n_i*dim+Rj*dim+i];
        }
    }
}
template void getRfraction<float>(float*, float*,size_t,size_t,size_t);
template void getRfraction<double>(double*, double*,size_t,size_t,size_t);

template <typename ValueType>
void freeVV2(ValueType** matrix) {
    if (matrix) {
        delete[] matrix[0];
        delete[] matrix;
    }
}
template void freeVV2(float**);
template void freeVV2(double**);

template <typename ValueType>
void freeConcatenated(ValueType**** B, size_t rows) {
    if (*B) {
        for (size_t c = 0; c < rows; ++c) {
            freeVV2((*B)[c]); 
        }
        delete[] *B; 
        *B = nullptr;
    }
}
template void freeConcatenated(float****,size_t);
template void freeConcatenated(double****,size_t);

}
}
#endif // CUBASE_HPP

