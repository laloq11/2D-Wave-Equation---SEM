#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <array>
#include <functional>
#include <stdexcept>

#include "types.hpp"
#include <stdio.h>

#include "linalg.hpp"
#include "alias.hpp"
#include "params.hpp"
#include "spectralcm.hpp"

#include <omp.h>


#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>


// Largest polynomial degree the GPU code supports: MAX_pD sizes the shared and local arrays of every kernel.
#define MAX_pD 5

namespace cie {
namespace otFlyGPU {

// Throws for a polynomial degree the GPU arrays cannot hold. Call it on the host before any GPU work.
inline void check_polynomialDegree_gpu(size_t p) {
    if (p < 1 || p > MAX_pD)
        throw std::invalid_argument("GPU solver supports p = 1.." + std::to_string(MAX_pD)
                                    + ", got p = " + std::to_string(p));
}

template <typename ValueType>
struct spectral_thrust{
    thrust::device_vector<ValueType> gllCoords;
    thrust::device_vector<ValueType> d_B;
    thrust::device_vector<ValueType> d_Bdiff;
    thrust::device_vector<ValueType> d_G;
    thrust::device_vector<ValueType> d_M;
    ValueType* d_fku;
    ValueType* d_mu;
    // thrust::device_vector<ValueType> d_u_tm1;
    ValueType* d_u_tm1;
    ValueType* d_u_tm2;
};
template <typename ValueType>
struct spectral_pointers{
    ValueType* gllCoords;
    ValueType* d_B;
    ValueType* d_Bdiff;
    ValueType* d_G;
    ValueType* d_M;
    ValueType* d_fku;
    ValueType* d_mu;
    ValueType* d_u_tm1;
    ValueType* d_u_tm2;
    spectral_pointers(): gllCoords(nullptr), d_B(nullptr), 
    d_Bdiff(nullptr), d_G(nullptr),d_M(nullptr), 
    d_fku(nullptr), d_mu(nullptr), d_u_tm1(nullptr), 
    d_u_tm2(nullptr) {}

    spectral_pointers(spectral_thrust<ValueType>& sp_t){
        gllCoords = thrust::raw_pointer_cast(sp_t.gllCoords.data());
        d_B = thrust::raw_pointer_cast(sp_t.d_B.data());
        d_Bdiff = thrust::raw_pointer_cast(sp_t.d_Bdiff.data());
        d_G = thrust::raw_pointer_cast(sp_t.d_G.data());
        d_M = thrust::raw_pointer_cast(sp_t.d_M.data());
        // d_u_tm1 = thrust::raw_pointer_cast(sp_t.d_u_tm1.data());
        d_fku = sp_t.d_fku;
        d_mu = sp_t.d_mu;
        d_u_tm1 = sp_t.d_u_tm1;
        d_u_tm2 = sp_t.d_u_tm2;
    }
};
template <typename ValueType>
struct GPUpackage{
    size_t dim;
    size_t pD;
    size_t ndofelement;
    size_t ndofdirection1;
    size_t nelements_0;
    size_t nelements;
    size_t ndofglobal;
    ValueType dt;
    ValueType ft_dt;
};


//init of lambda functions
__host__ __device__ __forceinline__ float local_sqrt(float val){ return sqrtf(val); }
__host__ __device__ __forceinline__ double local_sqrt(double val){ return sqrt(val); }
__device__ __forceinline__ float local_exp(float in) { return expf(in);}
__device__ __forceinline__ double local_exp(double in) { return exp(in); }

template <typename ValueType>
struct lambdafx_kernel {
    // ValueType sigmaS;  
    ValueType sigmaSf;
    // reduce number of operations
    lambdafx_kernel(ValueType sigmaS_) 
        : sigmaSf(-1.0 / (2 * sigmaS_ * sigmaS_)) {}
    __device__ __forceinline__ ValueType operator()
                                    (ValueType x, ValueType y) const {
    return 10.0 * local_exp( sigmaSf*( x*x+ y*y) );
}
};

template <typename ValueType>
struct lambdaXY_kernel {
    ValueType ds;  
    __device__ __forceinline__ ValueType operator()(ValueType selement, ValueType s) const {
        return (selement + s / 2.0 + 0.5) * ds;
    }
};

template <typename ValueType>
struct lambdaft_kernel {
    ValueType sigmaTd;
    ValueType t0;  
    ValueType exp_factor;

    lambdaft_kernel(ValueType sigmaT_,ValueType t0_) 
    :sigmaTd(-1.0 / (2.0 * sigmaT_*sigmaT_)),t0(t0_) {
        const ValueType pi = 3.14159265358979323846;
        exp_factor = local_sqrt(2.0 * pi) * (sigmaT_*sigmaT_*sigmaT_) ;
    }

    __device__ __forceinline__ ValueType operator()(ValueType t) const {
        return -(t - t0) / ( exp_factor ) 
                    * local_exp( sigmaTd* (t - t0) *(t - t0) );
        }
};
// end of lambda functions


template <typename ValueType>
__host__ void solve_w_eq_2d_cu_singleThread( 
    const cie::spectralkernel::Params& params,
    bool w,std::string name);

template <typename ValueType>

__host__ void solve_w_eq_2d_cu_pDegreeThread( 
    const cie::spectralkernel::Params& params,
    bool w,std::string name);


template <typename ValueType>
__host__ void solve_w_eq_2d_decreaseGrid_singleThread(const cie::spectralkernel::Params& params,
    bool w,
    std::string name, 
    std::string storage_type);


template <typename ValueType>
__host__ void solve_w_eq_2d_decreaseGrid_pDegreeThread(const cie::spectralkernel::Params& params,
    bool w,
    std::string name, 
    std::string storage_type);

template <typename ValueType>
__host__ void compute_invM_fKu_cu_singleThread(std::vector<ValueType>& u_host,
    const cie::spectralkernel::Params& params,
    bool write,
    std::string filename,
    std::stringstream& debug_buffer);

template <typename ValueType>
__host__ void compute_invM_fKu_cu_pDegreeThread(std::vector<ValueType>& u_host,
    const cie::spectralkernel::Params& params,
    bool write,
    std::string filename,
    std::stringstream& debug_buffer);

template <typename ValueType>
__host__ void compute_fKu_cu_singleThread(std::vector<ValueType>& u_host,
        const cie::spectralkernel::Params& params, 
        spectral_pointers<ValueType>& sp,
        lambdaXY_kernel<ValueType> MapX,
        lambdaXY_kernel<ValueType> MapY,
        lambdafx_kernel<ValueType> fx,
        lambdaft_kernel<ValueType> ft_k,
        std::stringstream& debug_buffer,
        bool write,std::string filename
    );

template <typename ValueType>
__host__ void compute_fKu_cu_pDegreeThread(std::vector<ValueType>& u_host,
    const cie::spectralkernel::Params& params, 
        spectral_pointers<ValueType>& sp,
        lambdaXY_kernel<ValueType> MapX,
        lambdaXY_kernel<ValueType> MapY,
        lambdafx_kernel<ValueType> fx,
        lambdaft_kernel<ValueType> ft_k,
        std::stringstream& debug_buffer,
        bool write,std::string filename
    );


template <typename ValueType>
__host__ void  initialize_spectral_gpu(spectral_pointers<ValueType>& sp,
    const cie::spectralkernel::Params& params);


template <typename ValueType>
__global__ void init_values_kernel_gpu(ValueType* u_tm2,
    ValueType* u_tm1, ValueType* fku,size_t n);


template <typename ValueType>
__host__ void preinitialize_invM_fKu_gpu(const cie::spectralkernel::Params& params,
    spectral_pointers<ValueType>& sp,
    std::stringstream& debug_buffer);
template <typename ValueType>
__host__ void finish_simulation_gpu(std::vector<ValueType>& u_host,
    spectral_pointers<ValueType>& sp,
    std::stringstream& debug_buffer);

template <typename ValueType>
void terminate_spectral_gpu(spectral_pointers<ValueType>& sp);




template <typename ValueType>
__host__ void intermediate_u_gpu(ValueType* u_hostdata,
    const size_t ndofglobal, spectral_pointers<ValueType>& sp);


template <typename ValueType>
__global__ void u_rotate_gpu(ValueType* u_tm2,
ValueType* u_tm1, ValueType* fku, size_t n);

template <typename ValueType>
__global__ void solve_u_it_gpu(ValueType* u_tm2,
    ValueType* u_tm1, ValueType* fku,ValueType* mu, size_t n);


template <typename ValueType>
__host__ void test_fku_solverVersion(const size_t it,const cie::spectralkernel::Params& params,
    const std::vector<ValueType>& u_tm1, std::vector<ValueType>& fku, int version);


template <typename Funcft, typename Funcfx, typename Funcmap,typename ValueType>
__global__ void solve_fKu_singleThread(size_t it,size_t dim,size_t pD, size_t ndofelement, 
    size_t ndofdirection1,size_t nelements_0, size_t nelements, 
    size_t ndofglobal, ValueType dt, 
    Funcft ft, Funcfx fx, Funcmap mapX, Funcmap mapY, ValueType* gllCoords,
    ValueType* B, ValueType* Bdiff,ValueType* G,ValueType* M,
    ValueType* u_tm1, ValueType* fku);

template <typename Funcft, typename Funcfx, typename Funcmap,typename ValueType>
__global__ void solve_fKu_pDegreeThread(size_t it,size_t dim,size_t pD, size_t ndofelement, 
    size_t ndofdirection1,size_t nelements_0, size_t nelements, 
    size_t ndofglobal, ValueType dt, 
    Funcft ft, Funcfx fx, Funcmap mapX, Funcmap mapY, ValueType* gllCoords,
    ValueType* B, ValueType* Bdiff,ValueType* G,ValueType* M,
    ValueType* u_tm1, ValueType* fku);

template <typename Funcfx, typename FuncMap,typename ValueType>
__device__ void get_Force_xy_singleThread(Funcfx fx,FuncMap mapX, FuncMap mapY,
    ValueType* Fe_element,size_t ielement, size_t jelement, 
    const ValueType* gllCoords, size_t pD);

template <typename ValueType>
__global__ void get_elementforce_test(ValueType* Fe,const ValueType* globalF,
    const size_t* locationM, const size_t ndofelement);

template <typename ValueType>
__device__  void get_elementforcecuda(ValueType* Fe,const ValueType* globalF,
    const size_t* locationM, const size_t ndofelement);

template <typename ValueType>
void test_mass_singleThread(std::vector<ValueType>& mu,
        const cie::spectralkernel::Params& params, size_t num_blocks);

template <typename ValueType>
__global__ void solve_Mass_singleThread(size_t dim,size_t pD, size_t ndofelement,
     size_t ndofdirection1,size_t nelements_0, size_t nelements, 
     size_t ndofglobal, ValueType* B, ValueType* G, ValueType* mu);



template <typename ValueType>
__device__ void alg_84_kernel_singleThread(const size_t D, 
    const ValueType* B,
    const ValueType* Bdiff,
    const ValueType* M,
    ValueType* v, 
    size_t pD,ValueType* r);

template <typename ValueType>
__device__ void alg_84_kernel_pDegreeThread(const size_t D, 
    const ValueType* B,
    const ValueType* Bdiff,
    const ValueType* M,
    ValueType* v, 
    size_t pD,ValueType* r);


template <typename ValueType>
__global__ void test_alg_84_mergeBd(ValueType* Bd_diff, const ValueType* Bdiff,
        const size_t t, size_t row, size_t col, size_t dim, int version);

template <typename ValueType>
__device__ void alg_84_mergeBd_kernel_singleThread(ValueType* Bd_diff, const ValueType* Bdiff,
        const size_t t, size_t row, size_t col, size_t dim);

template <typename ValueType>
__device__ void alg_84_mergeBd_kernel_pDegreeThread(ValueType* Bd_diff, const ValueType* Bdiff,
        const size_t t, size_t row, size_t col, size_t dim);

template <typename ValueType>
__global__ void test_alg_82(const size_t D, 
                    const ValueType* B,
                    const ValueType* G,
                    ValueType* v, 
                    size_t pD,
                    int version);

template <typename ValueType>
__global__ void test_alg_84(const size_t D, 
    const ValueType* B,
    const ValueType* Bdiff,
    const ValueType* M,
    ValueType* v, 
    size_t pD,ValueType* r,int version);


template <typename ValueType>
__device__ void alg_82_kernel_singleThread(const size_t D, 
                    const ValueType* B,
                    const ValueType* G,
                    ValueType* v, 
                    size_t pD);
template <typename ValueType>
__device__ void alg_82_kernel_pDegreeThread(const size_t D, 
                    const ValueType* B,
                    const ValueType* G,
                    ValueType* v, 
                    size_t pD);
template <typename ValueType>
__global__ void testalg_57(const size_t D, 
    ValueType* R, 
    ValueType* P,
    size_t pD, 
    size_t p_size,
    int version);

template <typename ValueType>
__device__ void alg_57_kernel_singleThread(const size_t D, 
                    const ValueType* R, 
                    ValueType* P,size_t pD, size_t p_size);


template <typename ValueType>
__device__ void alg_57_kernel_pDegreeThread(const size_t D, 
                    const ValueType* R, 
                    ValueType* P,size_t pD, size_t p_size);


__global__ void testlocationMap(size_t* LocationMap,
    const size_t pD, 
    size_t ndofdirection1,
    size_t ielement, 
    size_t jelement);


__device__ void createLocationMapcuda(size_t* LocationMap, 
            const size_t pD, size_t ndofdirection1, 
            size_t ielement, size_t jelement);

__device__ void arangecu(size_t* outV,size_t number,size_t addition);

template <typename ValueType>
__device__ void matvec_kernel(ValueType* result, ValueType* Mat, ValueType* vector, size_t r, size_t c);

template <typename ValueType>
__device__ ValueType* swapaxes_copy_kernel_singleThread(const ValueType* B,size_t dim, size_t row, size_t col);

template <typename ValueType>
__device__ void swapaxes_copy_kernel_pDegreeThread(ValueType* A, const ValueType* B,
    size_t dim, size_t row, size_t col);

template <typename ValueType>
__device__ void swapaxes_inplace_kernel_singleThread(ValueType* B,size_t dim, size_t row, size_t col);

template <typename ValueType>
__device__ void swapaxes_inplace_kernel_pDegreeThread(ValueType* A,ValueType* B,size_t dim, size_t row, size_t col);

template <typename ValueType>
__device__ void vectorwise_mul(const ValueType* R,ValueType* s,size_t ssize);
template <typename ValueType>
__device__ void vectorwise_sum(const ValueType* R,ValueType* s,size_t ssize);

template <typename ValueType>
__device__ void getRfraction_kernel_singleThread(ValueType* Rfraction, const ValueType* R, 
                size_t dim,size_t i, size_t n_i) {
    for (size_t Ri = 0; Ri < n_i; Ri++) {
        for (size_t Rj = 0; Rj < n_i; Rj++) {
            Rfraction[Ri*n_i+Rj] = R[Ri*n_i*dim+Rj*dim+i];
        }
    }
}
template __device__ void getRfraction_kernel_singleThread<float>(float*,const  float*,size_t,size_t,size_t);
template __device__ void getRfraction_kernel_singleThread<double>(double*, const double*,size_t,size_t,size_t);



template <typename ValueType>
__forceinline__ __device__ void getRcopy_kernel(ValueType* Rcopy, const ValueType* R, 
                size_t dim,size_t row, size_t col) {
    for (size_t t = 0; t < row; t++) {
        for (size_t i = 0; i < col; i++) {
            for (size_t j = 0; j < dim; j++) {
                Rcopy[t*col*dim+i*dim +j] = R[t*col*dim+i*dim +j];
            }
        }
    }
}
template  __device__ void getRcopy_kernel<float>(float*, const float*,size_t,size_t,size_t);
template  __device__ void getRcopy_kernel<double>(double*, const double*,size_t,size_t,size_t);



template <typename ValueType>
__forceinline__ __device__ void getPfraction_kernel(ValueType* Pfraction,const ValueType* P,
                            size_t j, size_t P_size,  size_t n_i) {

    // Ensure that the range is valid.
    if (j * n_i >= P_size || (j + 1) * n_i > P_size) {
        Pfraction = nullptr;
    }
    else {
        // std::copy(P + j * n_i, P + (j + 1) * n_i, Pfraction);//this does not work in cuda
        //this could be parallelized by pD+1
        for (size_t idx = 0; idx < n_i; ++idx) {
            Pfraction[idx] = P[j * n_i + idx];
        }
    }
}
template  __device__ void getPfraction_kernel<float>(float*, const float*, size_t,size_t,size_t);
template  __device__ void getPfraction_kernel<double>(double*, const double*, size_t,size_t,size_t);



template <typename ValueType>
__device__ void getRfraction_kernel_pDegreeThread(ValueType* Rfraction, const ValueType* R, 
                size_t dim,size_t i, size_t n_i);

template <typename ValueType>
__device__ void swappy_singleThread(ValueType* P, ValueType* T, size_t p_size){
    //this could be parallelized by p_size warps
    for (size_t idx = 0; idx < p_size; ++idx) {
        ValueType temp = P[idx];
        P[idx] = T[idx];
        T[idx] = temp;
    }
}
template __device__ void swappy_singleThread<float>(float*, float*, size_t);
template __device__ void swappy_singleThread<double>(double*, double*, size_t);

template <typename ValueType>
__device__ void swappy_pDegreeThread(ValueType* P, ValueType* T, size_t p_size);

template <typename ValueType>
void slice_kernel(ValueType* u_slice,const ValueType* u, 
    const size_t* locM, size_t ndofelement);

template <typename ValueType>
void get_global_ForcesV(const Params& params,
    std::vector<ValueType>& globalF);

template <typename ValueType>
__global__ void HelloKernel(size_t dim,size_t pD, size_t ndofelement, size_t ndofdirection1,
    size_t nelements_0, size_t nelements, 
    size_t ndofglobal, size_t* locationMap, ValueType* B, ValueType* G, ValueType* mu) ;

template <typename ValueType>
__host__ void writeU_inter(std::vector<ValueType>& u, std::string filename);


template <typename ValueType>
__host__ void writeU_inter(thrust::host_vector<ValueType>& u, std::string filename);
// start of lambda functions

template <typename Func,typename ValueType>
__global__ void test_lambdaXY(Func f,ValueType* res,
    ValueType selement, ValueType s) {
    res[0] = f(selement,s); 
}

template <typename Func,typename ValueType>
__global__ void test_lambdafx(Func f,ValueType* res,
    ValueType x, ValueType y) {
    res[0] = f(x,y); 
}
template <typename Func,typename ValueType>
__global__ void test_lambdaft_kernel(Func f,ValueType* res,
    ValueType t) {
    res[0] = f(t); 
}
}//ontheFlyGPU
}//cie

