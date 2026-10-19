#include <cuda.h>
#include <cuda_runtime.h>
#include "cuontheFly.cuh"
#include "cubase.hpp"

#include "error_handling.hpp"
#include "gpu_helpers.hpp"
#include "test_helpers.hpp"
#include "timers.hpp"
#include "types.hpp"



#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
// #define MAX_PSIZE 72 // //Max 
#define MAX_PSIZE (MAX_pD + 1)*(MAX_pD + 1)*2  
// constexpr int MAX_PSIZE = 72; //(5+1)*(5+1)*2+2

namespace cie {
namespace otFlyGPU {
template struct spectral_thrust<float>;
template struct spectral_thrust<double>;

template struct spectral_pointers<float>;
template struct spectral_pointers<double>;


template struct GPUpackage<float>;
template struct GPUpackage<double>;

template <typename ValueType>
__host__ void solve_w_eq_2d_cu_singleThread( //solve_w_eq_2d_cu_singleThread
    const cie::spectralkernel::Params& params,
    bool w,std::string name) {

    std::stringstream debug_buffer;
    debug_buffer << "\t--\tInitializing parameters..." << std::endl;
    // cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid();
    debug_buffer << "\t--\tInitializing CUDA solver..." << std::endl;
    debug_buffer<<"Total DOF is: "<<params.ndofglobal()<<std::endl;
    std::vector<ValueType> u(params.ndofglobal());
    compute_invM_fKu_cu_singleThread<ValueType>(u,params, w,name,debug_buffer);
    debug_buffer << "\t--\tExiting program..." << std::endl;
    debug_buffer<<"Total DOF is: "<<params.ndofglobal()<<std::endl;
    std::cout<<debug_buffer.str();

}
template __host__ void solve_w_eq_2d_cu_singleThread<float>(
    const cie::spectralkernel::Params& ,bool,std::string);




template <typename ValueType>
__host__ void solve_w_eq_2d_cu_pDegreeThread( 
    const cie::spectralkernel::Params& params,
    bool w,std::string name) {
        /* done: create a compute_invM_fKu_cu_pDegreeThread*/

    std::stringstream debug_buffer;
    debug_buffer << "\t--\tInitializing parameters..." << std::endl;
    // cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid();
    debug_buffer << "\t--\tInitializing CUDA solver..." << std::endl;
    debug_buffer<<"Total DOF is: "<<params.ndofglobal()<<std::endl;
    std::vector<ValueType> u(params.ndofglobal());
    compute_invM_fKu_cu_pDegreeThread<ValueType>(u,params, w,name,debug_buffer);
    debug_buffer << "\t--\tExiting program..." << std::endl;
    debug_buffer<<"Total DOF is: "<<params.ndofglobal()<<std::endl;
    std::cout<<debug_buffer.str();

}
template __host__ void solve_w_eq_2d_cu_pDegreeThread<float>(
    const cie::spectralkernel::Params& ,bool,std::string);



template <typename ValueType>
__host__ void solve_w_eq_2d_decreaseGrid_singleThread(const cie::spectralkernel::Params& params,
    bool w,
    std::string name, 
    std::string storage_type) {
    /*
        Still to define correctly how storage_type will work. 
    */

    std::stringstream debug_buffer;

    debug_buffer << "\t--\tInitializing singleThread solver..." << std::endl;
    std::vector<ValueType> u(params.ndofglobal());
    compute_invM_fKu_cu_singleThread<ValueType>(u,params, w,name, debug_buffer);
    debug_buffer << "\t--\tExiting program..." << std::endl;
    std::cout<<debug_buffer.str();

}
template __host__ void solve_w_eq_2d_decreaseGrid_singleThread<float>(const cie::spectralkernel::Params&,
    bool, std::string, std::string);


template <typename ValueType>
__host__ void solve_w_eq_2d_decreaseGrid_pDegreeThread(const cie::spectralkernel::Params& params,
    bool w,
    std::string name, 
    std::string storage_type) {
    /*
    Still to define correctly how storage_type will work. 
    */

    std::stringstream debug_buffer;
    debug_buffer << "\t--\tInitializing pDegreeThread solver..." << std::endl;
    std::vector<ValueType> u(params.ndofglobal());
    compute_invM_fKu_cu_pDegreeThread<ValueType>(u,params, w,name, debug_buffer);
    debug_buffer << "\t--\tExiting program..." << std::endl;
    std::cout<<debug_buffer.str();

}
template __host__ void solve_w_eq_2d_decreaseGrid_pDegreeThread<float>(const cie::spectralkernel::Params&,
    bool, std::string, std::string);

template <typename ValueType>
__host__ void compute_invM_fKu_cu_singleThread(std::vector<ValueType>& u_host, //compute_invM_fKu_cu_singleThread
    const cie::spectralkernel::Params& params,
    bool write,
    std::string filename,
    std::stringstream& debug_buffer) {
        // u[it]
        // params containing all case data
    size_t polynomialDegree = params.polynomialDegree();
    size_t ndofglobal = params.ndofglobal();
    size_t nelements = params.nelements(0)*params.nelements(1);
    size_t dim =2;

    int threadsPerBlock2 = 256;
    int blocksPerGrid2 = (ndofglobal + threadsPerBlock2 - 1)
                            / threadsPerBlock2;


    debug_buffer<<"Starting CUDA solver..."<<std::endl;

    spectral_pointers<ValueType> sp;

    initialize_spectral_gpu<ValueType>(sp,params);

    ValueType dx= params.dx();
    ValueType dy= params.dy();
    ValueType sS= params.sigmaS();
    ValueType sT= params.sigmaT();
    ValueType t0= params.t0();
    lambdaXY_kernel<ValueType> MapX{dx}; 
    lambdaXY_kernel<ValueType> MapY{dy}; 
    lambdafx_kernel<ValueType> fx{sS}; 
    lambdaft_kernel<ValueType> ft_k{sT,t0}; 
    
    debug_buffer << "\t--\tInitializing solver..." << std::endl;

    debug_buffer<<"\t\t\t***  Mass matrix ***"<<std::endl;

    solve_Mass_singleThread<ValueType><<<nelements,1>>>(dim,
        polynomialDegree,
        params.ndofelement(),
        params.ndofdirection()[1],
        params.nelements(0),
        nelements,
        ndofglobal,
        sp.d_B,
        sp.d_G,
        sp.d_mu
    );
    ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
    ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
    debug_buffer<<"\t\t\t***  Mass matrix *** --- DONE"<<std::endl;

    debug_buffer<<"\t\t\t***   iteration variables solver ***"<<std::endl;
    init_values_kernel_gpu<<<blocksPerGrid2, threadsPerBlock2>>>(sp.d_u_tm2,
        sp.d_u_tm1,sp.d_fku,ndofglobal);
    ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());

    ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());    
    debug_buffer<<"\t\t\t***   iteration variables solver ***--- DONE"<<std::endl;

    debug_buffer << "\t--\tStarting solver... Running "<< params.nsteps()<< std::endl;
    for (size_t it = 2; it < (params.nsteps()+1); ++it) {

        // ValueType ft_dt = params.ft(it * params.dt());

        solve_fKu_singleThread<lambdaft_kernel<ValueType>,
        lambdafx_kernel<ValueType>,
        lambdaXY_kernel<ValueType>,
        ValueType><<<nelements,1>>>(it,dim,
        polynomialDegree,
        params.ndofelement(),
        params.ndofdirection()[1],
        params.nelements(0),
        nelements,
        ndofglobal, params.dt(),
        ft_k,
        fx,
        MapX,
        MapY,
        sp.gllCoords,
        sp.d_B,sp.d_Bdiff,
        sp.d_G,
        sp.d_M,
        sp.d_u_tm1,
        sp.d_fku
        );
        ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());

        ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());

        
        solve_u_it_gpu<<<blocksPerGrid2, threadsPerBlock2>>>(sp.d_u_tm2,
            sp.d_u_tm1,sp.d_fku,sp.d_mu,ndofglobal);
        ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());

        ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());

        u_rotate_gpu<<<blocksPerGrid2, threadsPerBlock2>>>(sp.d_u_tm2,
                sp.d_u_tm1,sp.d_fku,ndofglobal);
        ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());

        ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
        
        if (it % params.samplingFrecuency() == 0) {
                intermediate_u_gpu(u_host.data(),ndofglobal,sp);
                
                debug_buffer<< "Iteration:\t" << it
                << "\t with u_(" << ndofglobal / 2
                << ")= " << u_host[ndofglobal / 2] << std::endl;
                
                if (write){
                debug_buffer<<"\t\t Initiating write..."<<filename<<std::endl;
                writeU_inter<ValueType>(u_host,filename);
            }
        }
    }

    terminate_spectral_gpu(sp);
}
template __host__ void compute_invM_fKu_cu_singleThread<float>(
    std::vector<float>& ,const cie::spectralkernel::Params&,
    bool,
    std::string, 
    std::stringstream&);

template <typename ValueType>
__host__ void compute_invM_fKu_cu_pDegreeThread(std::vector<ValueType>& u_host, //compute_invM_fKu_cu_singleThread
    const cie::spectralkernel::Params& params,
    bool write,
    std::string filename,
    std::stringstream& debug_buffer) {
        // u[it]
        // params containing all case data
    size_t polynomialDegree = params.polynomialDegree();
    size_t P_size = (polynomialDegree+1) * (polynomialDegree+1);
    size_t ndofglobal = params.ndofglobal();
    size_t nelements = params.nelements(0)*params.nelements(1);
    size_t dim =2;

    int threadsPerBlock2 = 256;
    int blocksPerGrid2 = (ndofglobal + threadsPerBlock2 - 1)
                            / threadsPerBlock2;


    debug_buffer<<"Starting CUDA solver..."<<std::endl;

    spectral_pointers<ValueType> sp;

    initialize_spectral_gpu<ValueType>(sp,params);

    ValueType dx= params.dx();
    ValueType dy= params.dy();
    ValueType sS= params.sigmaS();
    ValueType sT= params.sigmaT();
    ValueType t0= params.t0();
    lambdaXY_kernel<ValueType> MapX{dx}; 
    lambdaXY_kernel<ValueType> MapY{dy}; 
    lambdafx_kernel<ValueType> fx{sS}; 
    lambdaft_kernel<ValueType> ft_k{sT,t0}; 
    
    debug_buffer << "\t--\tInitializing solver..." << std::endl;

    debug_buffer<<"\t\t\t***  Mass matrix ***"<<std::endl;

    solve_Mass_singleThread<ValueType><<<nelements,1>>>(dim,
        polynomialDegree,
        params.ndofelement(),
        params.ndofdirection()[1],
        params.nelements(0),
        nelements,
        ndofglobal,
        sp.d_B,
        sp.d_G,
        sp.d_mu
    );
    ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
    ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
    debug_buffer<<"\t\t\t***  Mass matrix *** --- DONE"<<std::endl;

    debug_buffer<<"\t\t\t***   iteration variables solver ***"<<std::endl;
    init_values_kernel_gpu<<<blocksPerGrid2, threadsPerBlock2>>>(sp.d_u_tm2,
        sp.d_u_tm1,sp.d_fku,ndofglobal);
    ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());

    ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());    
    debug_buffer<<"\t\t\t***   iteration variables solver ***--- DONE"<<std::endl;

    debug_buffer << "\t--\tStarting solver... Running "<< params.nsteps()<< std::endl;
    for (size_t it = 2; it < (params.nsteps()+1); ++it) {

        // ValueType ft_dt = params.ft(it * params.dt());

        solve_fKu_pDegreeThread<lambdaft_kernel<ValueType>,
        lambdafx_kernel<ValueType>,
        lambdaXY_kernel<ValueType>,
        ValueType><<<nelements,P_size>>>(it,dim,
        polynomialDegree,
        params.ndofelement(),
        params.ndofdirection()[1],
        params.nelements(0),
        nelements,
        ndofglobal, params.dt(),
        ft_k,
        fx,
        MapX,
        MapY,
        sp.gllCoords,
        sp.d_B,sp.d_Bdiff,
        sp.d_G,
        sp.d_M,
        sp.d_u_tm1,
        sp.d_fku
        );
        ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());

        ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());

        
        solve_u_it_gpu<<<blocksPerGrid2, threadsPerBlock2>>>(sp.d_u_tm2,
            sp.d_u_tm1,sp.d_fku,sp.d_mu,ndofglobal);
        ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());

        ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());

        u_rotate_gpu<<<blocksPerGrid2, threadsPerBlock2>>>(sp.d_u_tm2,
                sp.d_u_tm1,sp.d_fku,ndofglobal);
        ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());

        ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
        
        if (it % params.samplingFrecuency() == 0) {
                intermediate_u_gpu(u_host.data(),ndofglobal,sp);
                
                debug_buffer<< "Iteration:\t" << it
                << "\t with u_(" << ndofglobal / 2
                << ")= " << u_host[ndofglobal / 2] << std::endl;
                
                if (write){
                debug_buffer<<"\t\t Initiating write..."<<filename<<std::endl;
                writeU_inter<ValueType>(u_host,filename);
            }
        }
    }

    terminate_spectral_gpu(sp);
}
template __host__ void compute_invM_fKu_cu_pDegreeThread<float>(
    std::vector<float>& ,const cie::spectralkernel::Params&,
    bool,
    std::string, 
    std::stringstream&);



template <typename ValueType>
__host__ void preinitialize_invM_fKu_gpu(const cie::spectralkernel::Params& params,
    spectral_pointers<ValueType>& sp,
    std::stringstream& debug_buffer) {
        /*
        Is this function exclusively for singleThread or is gpu in general?
        If this is gpu general
            preinitialize_invM_fKu_gpu --> preinitialize_invM_fKu_GGPU
        */
        // u[it]
        // params containing all case data
    size_t polynomialDegree = params.polynomialDegree();
    size_t ndofglobal = params.ndofglobal();
    size_t nelements = params.nelements(0)*params.nelements(1);
    size_t dim =2;

    int threadsPerBlock2 = 256;
    int blocksPerGrid2 = (ndofglobal + threadsPerBlock2 - 1)
                            / threadsPerBlock2;


    debug_buffer<<"Starting CUDA solver..."<<std::endl;

    initialize_spectral_gpu<ValueType>(sp,params);

    ValueType dx= params.dx();
    ValueType dy= params.dy();
    ValueType sS= params.sigmaS();
    ValueType sT= params.sigmaT();
    ValueType t0= params.t0();
    lambdaXY_kernel<ValueType> MapX{dx}; 
    lambdaXY_kernel<ValueType> MapY{dy}; 
    lambdafx_kernel<ValueType> fx{sS}; 
    lambdaft_kernel<ValueType> ft_k{sT,t0}; 
    
    debug_buffer << "\t--\tInitializing solver..." << std::endl;

    debug_buffer<<"\t\t\t***  Mass matrix ***"<<std::endl;

    solve_Mass_singleThread<ValueType><<<nelements,1>>>(dim,
        polynomialDegree,
        params.ndofelement(),
        params.ndofdirection()[1],
        params.nelements(0),
        nelements,
        ndofglobal,
        sp.d_B,
        sp.d_G,
        sp.d_mu
    );
    ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
    ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
    debug_buffer<<"\t\t\t***  Mass matrix *** --- DONE"<<std::endl;

    debug_buffer<<"\t\t\t***   iteration variables solver ***"<<std::endl;
    init_values_kernel_gpu<<<blocksPerGrid2, threadsPerBlock2>>>(sp.d_u_tm2,
        sp.d_u_tm1,sp.d_fku,ndofglobal);
    ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
    ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());

    debug_buffer<<"\t\t\t***   iteration variables solver ***--- DONE"<<std::endl;

    debug_buffer << "\t--\tStarting solver... Running "<< params.nsteps()<< std::endl;
}
template __host__ void preinitialize_invM_fKu_gpu<float>(
                const cie::spectralkernel::Params&,
                spectral_pointers<float>& ,std::stringstream&);


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

    ){
    size_t polynomialDegree = params.polynomialDegree();
    size_t ndofglobal = params.ndofglobal();
    size_t nelements = params.nelements(0)*params.nelements(1);
    size_t dim =2;
    int threadsPerBlock2 = 256;
    int blocksPerGrid2 = (ndofglobal + threadsPerBlock2 - 1)
                            / threadsPerBlock2;

    for (size_t it = 2; it < (params.nsteps()+1); ++it) {

        solve_fKu_singleThread<lambdaft_kernel<ValueType>,
        lambdafx_kernel<ValueType>,
        lambdaXY_kernel<ValueType>,
        ValueType> <<<nelements,1>>>(it,
        dim,
        polynomialDegree,
        params.ndofelement(),
        params.ndofdirection()[1],
        params.nelements(0),
        nelements,
        ndofglobal, params.dt(),
        ft_k,
        fx,
        MapX,
        MapY,
        sp.gllCoords,
        sp.d_B,sp.d_Bdiff,
        sp.d_G,
        sp.d_M,
        sp.d_u_tm1,
        sp.d_fku
        );
        ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
        ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());

        solve_u_it_gpu<<<blocksPerGrid2, threadsPerBlock2>>>(sp.d_u_tm2,
            sp.d_u_tm1,
            sp.d_fku,
            sp.d_mu,
            ndofglobal);
        ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
        ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());

        u_rotate_gpu<<<blocksPerGrid2, threadsPerBlock2>>>(sp.d_u_tm2,
            sp.d_u_tm1,
            sp.d_fku,
            ndofglobal);
        ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
        ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
        if (it % params.samplingFrecuency() == 0) {
            //     intermediate_u_gpu(u_host.data(),ndofglobal,sp);

            //     debug_buffer<< "Iteration:\t" << it
            //     << "\t with u_(" << ndofglobal / 2
            //     << ")= " << u_host[ndofglobal / 2] << std::endl;
            //     if (write){
            //         debug_buffer<<"\t\t Initiating write..."<<filename<<std::endl;
            //         writeU_inter<ValueType>(u_host,filename);
            // }
        }
    }
}

template __host__ void compute_fKu_cu_singleThread<float>(
    std::vector<float>& ,const cie::spectralkernel::Params&,
    spectral_pointers<float>& ,
    lambdaXY_kernel<float>,
    lambdaXY_kernel<float>,
    lambdafx_kernel<float>,
    lambdaft_kernel<float>,
    std::stringstream&,
    bool,std::string);





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
    ){
    size_t polynomialDegree = params.polynomialDegree();
    size_t P_size = (polynomialDegree+1) * (polynomialDegree+1);
    size_t ndofglobal = params.ndofglobal();
    size_t nelements = params.nelements(0)*params.nelements(1);
    size_t dim =2;
    int threadsPerBlock2 = 256;
    int blocksPerGrid2 = (ndofglobal + threadsPerBlock2 - 1)
                            / threadsPerBlock2;

    for (size_t it = 2; it < (params.nsteps()+1); ++it) {

        solve_fKu_pDegreeThread<lambdaft_kernel<ValueType>,
        lambdafx_kernel<ValueType>,
        lambdaXY_kernel<ValueType>,
        ValueType> <<<nelements,P_size>>>(it,
        dim,
        polynomialDegree,
        params.ndofelement(),
        params.ndofdirection()[1],
        params.nelements(0),
        nelements,
        ndofglobal, params.dt(),
        ft_k,
        fx,
        MapX,
        MapY,
        sp.gllCoords,
        sp.d_B,sp.d_Bdiff,
        sp.d_G,
        sp.d_M,
        sp.d_u_tm1,
        sp.d_fku
        );
        ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
        ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());

        solve_u_it_gpu<<<blocksPerGrid2, threadsPerBlock2>>>(sp.d_u_tm2,
            sp.d_u_tm1,
            sp.d_fku,
            sp.d_mu,
            ndofglobal);
        ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
        ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());

        u_rotate_gpu<<<blocksPerGrid2, threadsPerBlock2>>>(sp.d_u_tm2,
            sp.d_u_tm1,
            sp.d_fku,
            ndofglobal);
        ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
        ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
        if (it % params.samplingFrecuency() == 0) {
            //     intermediate_u_gpu(u_host.data(),ndofglobal,sp);

            //     debug_buffer<< "Iteration:\t" << it
            //     << "\t with u_(" << ndofglobal / 2
            //     << ")= " << u_host[ndofglobal / 2] << std::endl;
            //     if (write){
            //         debug_buffer<<"\t\t Initiating write..."<<filename<<std::endl;
            //         writeU_inter<ValueType>(u_host,filename);
            // }
        }
    }
}

template __host__ void compute_fKu_cu_pDegreeThread<float>(
    std::vector<float>& ,const cie::spectralkernel::Params&,
    spectral_pointers<float>& ,
    lambdaXY_kernel<float>,
    lambdaXY_kernel<float>,
    lambdafx_kernel<float>,
    lambdaft_kernel<float>,
    std::stringstream&,
    bool,std::string);


template <typename ValueType>
__host__ void finish_simulation_gpu(std::vector<ValueType>& u_host,
    spectral_pointers<ValueType>& sp,
    std::stringstream& debug_buffer){
    debug_buffer << "\t--\t Finishing program..." << std::endl;

    //commented as we might not need this... 
    //due to it  % params.samplingFrecuency() {last time step} is going 
    //going to be written back to host... 
    debug_buffer << "\t--\t Copying data back from device" << std::endl;
    intermediate_u_gpu(u_host.data(),u_host.size(),sp);

    debug_buffer << "\t--Freeing up space on device..." << std::endl;
    terminate_spectral_gpu(sp);

    debug_buffer << "\t--\tExiting program..." << std::endl;
}
template __host__ void finish_simulation_gpu<float>(std::vector<float>&,
    spectral_pointers<float>&,
    std::stringstream&);


template <typename ValueType>
__global__ void u_rotate_gpu(ValueType* u_tm2,
ValueType* u_tm1, ValueType* fku, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double b_val = u_tm1[i];
        u_tm2[i] = b_val;
        u_tm1[i] = fku[i]; //this is always going to be the last iteration val
        fku[i] = 0.0;
    }
}
template __global__ void u_rotate_gpu<float>(float*,
    float*,float*,size_t);


template <typename ValueType>
__global__ void solve_u_it_gpu(ValueType* u_tm2,
    ValueType* u_tm1, ValueType* fku,ValueType* mu, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        fku[i] = 2.0 * u_tm1[i] - u_tm2[i] +fku[i]/mu[i];
    }
}
template __global__ void solve_u_it_gpu<float>(float*,
    float*,float*,float*,size_t);


template <typename ValueType>
__host__ void initialize_spectral_gpu(spectral_pointers<ValueType>& sp,
    const cie::spectralkernel::Params& params) {

    check_polynomialDegree_gpu(params.polynomialDegree());

	ValueType rho = params.rho();
	ValueType dx =  params.dx();
	ValueType dy =  params.dy();
    ValueType c = params.c();
	ValueType facG = (rho * (dx/2.0) * (dy/2.0));
	ValueType facM = (rho * (dx/2.0) * (dy/2.0))* c * c;
	ValueType facS = (2.0/dx);
	ValueType facR = (2.0/dy);
    // B
    
    //memory alloc
    size_t polynomialDegree = params.polynomialDegree();
    //dont allocate
    adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1); 

        // end dont allocate
    size_t nelements0 = params.nelements(0);
    size_t ndofdirection1 = params.ndofdirection()[1];
    size_t nelements = params.nelements(1)*params.nelements(0);
    size_t ndofelement = params.ndofelement();
    size_t dim = 2;
    size_t ndofglobal = params.ndofglobal();

	// ValueType* fku = new ValueType[ndofglobal];
    ValueType* gllCoords = new ValueType[(polynomialDegree + 1)];
    for (size_t i=0;i<polynomialDegree+1;i++) {
        gllCoords[i] =params.gllcoords()[i]; 
        // printf(gllCoords[i]);
    };
	Matrixt<ValueType> S = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), 0);
	Matrixt<ValueType> R = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), 0);


	Matrixt<ValueType> Sdiff = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), 1);
	Matrixt<ValueType> Rdiff = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), 1);
	cie::otFlyGPU::ShapeFunctionFactor(Sdiff,facS);
	cie::otFlyGPU::ShapeFunctionFactor(Rdiff,facR);


	ValueType* B = new ValueType[(polynomialDegree + 1)*(polynomialDegree + 1)*2];
	cie::otFlyGPU::concatenate_flat(B, S, R,polynomialDegree+1,2); //pD+1,dim 
	ValueType* Bdiff = new ValueType[(polynomialDegree + 1)*(polynomialDegree + 1)*2];
	cie::otFlyGPU::concatenate_flat(Bdiff, Sdiff, Rdiff,polynomialDegree+1,2); //pD+1,dim 

	ValueType* G = new ValueType[(polynomialDegree+1)*(polynomialDegree+1)];
	kron_flat(G,gauss.getWeights(), gauss.getWeights(), facG , (polynomialDegree+1));
	ValueType* M = new ValueType[(polynomialDegree+1)*(polynomialDegree+1)];
	kron_flat(M,gauss.getWeights(), gauss.getWeights(), facM , (polynomialDegree+1));

    // std::vector<ValueType> global_Forces(params.ndofglobal(), 0.0); 
    // get_global_ForcesV<ValueType>(params,global_Forces);

    size_t P_size = (polynomialDegree + 1)*(polynomialDegree + 1);

    ASSERT_NO_CUDA_ERRORS(cudaMalloc(&sp.d_G, 
        sizeof(ValueType) * P_size ) );
    ASSERT_NO_CUDA_ERRORS(cudaMalloc(&sp.d_M, 
        sizeof(ValueType) * P_size ) );

    ASSERT_NO_CUDA_ERRORS(cudaMalloc(&sp.d_B,
        sizeof(ValueType) * (P_size*2) ) );

    ASSERT_NO_CUDA_ERRORS(cudaMalloc(&sp.d_Bdiff,
        sizeof(ValueType) * (P_size*2) ) );

    ASSERT_NO_CUDA_ERRORS(cudaMalloc(&sp.gllCoords,
        sizeof(ValueType) * (polynomialDegree+1)));

    ASSERT_NO_CUDA_ERRORS(cudaMalloc(&sp.d_fku, 
        sizeof(ValueType) * ndofglobal ) );

    ASSERT_NO_CUDA_ERRORS(cudaMalloc(&sp.d_mu, 
        sizeof(ValueType) * ndofglobal ) );

    ASSERT_NO_CUDA_ERRORS(cudaMalloc(&sp.d_u_tm1, 
        sizeof(ValueType) * ndofglobal ) );

    ASSERT_NO_CUDA_ERRORS(cudaMalloc(&sp.d_u_tm2, 
        sizeof(ValueType) * ndofglobal ) );



   //copy data from Host to Device
    ASSERT_NO_CUDA_ERRORS(cudaMemcpy(sp.d_B, B,
        sizeof(ValueType) * (P_size*2),
        cudaMemcpyHostToDevice));

    ASSERT_NO_CUDA_ERRORS(cudaMemcpy(sp.d_Bdiff, Bdiff,
        sizeof(ValueType) * (P_size*2),
        cudaMemcpyHostToDevice));

    ASSERT_NO_CUDA_ERRORS(cudaMemcpy(sp.d_G, G,
        sizeof(ValueType) * P_size,
        cudaMemcpyHostToDevice));

    ASSERT_NO_CUDA_ERRORS(cudaMemcpy(sp.d_M, M,
        sizeof(ValueType) * P_size,
        cudaMemcpyHostToDevice));

    ASSERT_NO_CUDA_ERRORS(cudaMemcpy(sp.gllCoords, gllCoords,
        sizeof(ValueType) * (polynomialDegree+1),
        cudaMemcpyHostToDevice));

    // host copies are no longer needed; the device copies are freed by terminate_spectral_gpu
    delete[] B;
    delete[] Bdiff;
    delete[] G;
    delete[] M;
    delete[] gllCoords;
}
template __host__ void initialize_spectral_gpu(spectral_pointers<float>&,const cie::spectralkernel::Params& params);


template <typename ValueType>
__global__ void init_values_kernel_gpu(ValueType* u_tm2,
    ValueType* u_tm1, ValueType* fku,size_t ndofglobal){

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ndofglobal) {
        u_tm2[i]=0.0;
        u_tm1[i]=0.0;
        fku[i] = 0.0;
    }
}
template __global__ void init_values_kernel_gpu<float>(float*,float*,float*,size_t);


template <typename ValueType>
void terminate_spectral_gpu(spectral_pointers<ValueType>& sp){
    // std::cout<<"\t\t -- Freeing space on device."<<std::endl;
    cudaFree(sp.d_B);
    cudaFree(sp.d_Bdiff);
    cudaFree(sp.d_G);
    cudaFree(sp.d_M);
    cudaFree(sp.d_fku);
    cudaFree(sp.d_mu);
    cudaFree(sp.gllCoords);
    cudaFree(sp.d_u_tm1);
    cudaFree(sp.d_u_tm2);
}
template void terminate_spectral_gpu<float>(spectral_pointers<float>&);
template void terminate_spectral_gpu<double>(spectral_pointers<double>&);


template <typename ValueType>
__host__ void intermediate_u_gpu(ValueType* u_hostdata,
    const size_t ndofglobal, spectral_pointers<ValueType>& sp){

    // copy back data.
    ASSERT_NO_CUDA_ERRORS(cudaMemcpy(u_hostdata, sp.d_u_tm1,
        sizeof(ValueType) * ndofglobal,
        cudaMemcpyDeviceToHost));

    ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
    ASSERT_NO_CUDA_ERRORS( cudaDeviceSynchronize() );
}

template __host__ void intermediate_u_gpu<float>(float*,
    const size_t,spectral_pointers<float>&);
template __host__ void intermediate_u_gpu<double>(double*,
    const size_t,spectral_pointers<double>&);




template <typename ValueType>
__host__ void test_fku_solverVersion(const size_t it,const cie::spectralkernel::Params& params,
    const std::vector<ValueType>& u_tm1, std::vector<ValueType>& fku, int version) {

    check_polynomialDegree_gpu(params.polynomialDegree());

    // std::fill(fku.begin(),fku.end(),0.0);
    // size_t it = 2;

    // ValueType ft_dt = params.ft(it * params.dt());

	ValueType rho = 1;//params.rho();
	ValueType dx =  params.dx();
	ValueType dy =  params.dy();
    ValueType c = params.c();
	ValueType facG = (rho * (dx/2.0) * (dy/2.0));
	ValueType facM = (rho * (dx/2.0) * (dy/2.0))* c * c;
	ValueType facS = (2.0/dx);
	ValueType facR = (2.0/dy);
    // B
    
    //memory alloc
    size_t polynomialDegree = params.polynomialDegree();
    //dont allocate
    adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1); 

        // end dont allocate
    size_t nelements0 = params.nelements(0);
    size_t ndofdirection1 = params.ndofdirection()[1];
    size_t nelements = params.nelements(1)*params.nelements(0);
    size_t P_size = (polynomialDegree + 1)*(polynomialDegree + 1);
    size_t ndofelement = params.ndofelement();
    size_t dim = 2;
    size_t ndofglobal = params.ndofglobal();

	// ValueType* fku = new ValueType[ndofglobal];

	Matrixt<ValueType> S = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), 0);
	Matrixt<ValueType> R = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), 0);


	Matrixt<ValueType> Sdiff = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), 1);
	Matrixt<ValueType> Rdiff = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), 1);
	cie::otFlyGPU::ShapeFunctionFactor(Sdiff,facS);
	cie::otFlyGPU::ShapeFunctionFactor(Rdiff,facR);


	ValueType* B = new ValueType[(polynomialDegree + 1)*(polynomialDegree + 1)*2];
	cie::otFlyGPU::concatenate_flat(B, S, R,polynomialDegree+1,2); //pD+1,dim 
	ValueType* Bdiff = new ValueType[(polynomialDegree + 1)*(polynomialDegree + 1)*2];
	cie::otFlyGPU::concatenate_flat(Bdiff, Sdiff, Rdiff,polynomialDegree+1,2); //pD+1,dim 

	ValueType* G = new ValueType[(polynomialDegree+1)*(polynomialDegree+1)];
	kron_flat(G,gauss.getWeights(), gauss.getWeights(), facG , (polynomialDegree+1));
	ValueType* M = new ValueType[(polynomialDegree+1)*(polynomialDegree+1)];
	kron_flat(M,gauss.getWeights(), gauss.getWeights(), facM , (polynomialDegree+1));


    ValueType* d_B;
    ValueType* d_Bdiff;
    ValueType* d_G;
    ValueType* d_fku; 

    ValueType* d_u_tm1;
    ValueType* d_M;
    ValueType* d_gllCoords;
    ValueType* gllCoords = new ValueType[(polynomialDegree + 1)];
    for (size_t i=0;i<polynomialDegree+1;i++) gllCoords[i] = params.gllcoords()[i]; 


    ValueType sS= params.sigmaS();
    ValueType sT= params.sigmaT();
    ValueType t0= params.t0();
    lambdaXY_kernel<ValueType> MapX{dx}; 
    lambdaXY_kernel<ValueType> MapY{dy}; 
    lambdafx_kernel<ValueType> fx{sS}; 
    lambdaft_kernel<ValueType> ft_k{sT,t0}; 

    ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_B,
        sizeof(ValueType) * ((polynomialDegree+1)*(polynomialDegree+1)*2)));
    ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_Bdiff,
        sizeof(ValueType) * ((polynomialDegree+1)*(polynomialDegree+1)*2)));

    ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_G, sizeof(ValueType) * P_size ) );
   ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_fku, sizeof(ValueType) * ndofglobal ) );
   ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_u_tm1, sizeof(ValueType) * ndofglobal ) );
   
   ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_M, sizeof(ValueType) * P_size ) );
   ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_gllCoords,
    sizeof(ValueType) * (polynomialDegree+1)));


   //copy data from Host to Device
      //copy data from Host to Device
    ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_fku, fku.data(),
        sizeof(ValueType) * (ndofglobal),
        cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_B, B,
        sizeof(ValueType) * (P_size*2),
        cudaMemcpyHostToDevice));
        ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_Bdiff, Bdiff,
        sizeof(ValueType) * (P_size*2),
        cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_G, G,
        sizeof(ValueType) * P_size,
        cudaMemcpyHostToDevice));

    ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_gllCoords, gllCoords,
        sizeof(ValueType) * (polynomialDegree+1),
        cudaMemcpyHostToDevice));
   //new, different from Mass below:
    ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_M, M,
        sizeof(ValueType) * P_size,
        cudaMemcpyHostToDevice));

    ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_u_tm1, u_tm1.data(),
        sizeof(ValueType) * ndofglobal,
        cudaMemcpyHostToDevice));

        
    ValueType dt = params.dt();

    if (version ==1){
        solve_fKu_singleThread<lambdaft_kernel<ValueType>,
            lambdafx_kernel<ValueType>,
            lambdaXY_kernel<ValueType>,
            ValueType><<<nelements,1>>>(it,dim,
            polynomialDegree,
            params.ndofelement(),
            params.ndofdirection()[1],
            params.nelements(0),
            nelements,
            ndofglobal, 
            dt, 
            ft_k,
            fx,
            MapX,
            MapY,
            d_gllCoords,
            d_B,d_Bdiff,
            d_G,d_M,
            d_u_tm1,
            d_fku);
    }    
    else if (version ==2){
        solve_fKu_pDegreeThread<lambdaft_kernel<ValueType>,
            lambdafx_kernel<ValueType>,
            lambdaXY_kernel<ValueType>,
            ValueType><<<nelements,P_size>>>(it,dim,
            polynomialDegree,
            params.ndofelement(),
            params.ndofdirection()[1],
            params.nelements(0),
            nelements,
            ndofglobal, 
            dt, 
            ft_k,
            fx,
            MapX,
            MapY,
            d_gllCoords,
            d_B,d_Bdiff,
            d_G,d_M,
            d_u_tm1,
            d_fku);
    }

    ASSERT_NO_CUDA_ERRORS( cudaDeviceSynchronize() );

    //copy back data.
    ASSERT_NO_CUDA_ERRORS(cudaMemcpy(fku.data(), d_fku,
    sizeof(ValueType) * ndofglobal,
    cudaMemcpyDeviceToHost));

    // this function will delete now the data. 
    //in a normal computation we might don't want to do this as
    //sending back data will be expensive. We would probably only free up
    //the mu but remain the rest intact. 
    cudaFree(d_B);
    cudaFree(d_Bdiff);
    cudaFree(d_M);
    cudaFree(d_G);
    cudaFree(d_fku);
    cudaFree(d_gllCoords);
    cudaFree(d_u_tm1);

    delete[] B;
    delete[] Bdiff;
    delete[] G;
    delete[] M;
    delete[] gllCoords;
}
template __host__ void test_fku_solverVersion<float>(const size_t,const cie::spectralkernel::Params&,
    const std::vector<float>&, std::vector<float>&,int);




template <typename Funcft, typename Funcfx, typename Funcmap,typename ValueType>
__global__ void solve_fKu_singleThread(size_t it,size_t dim,size_t pD, size_t ndofelement, 
    size_t ndofdirection1,size_t nelements_0, size_t nelements, 
    size_t ndofglobal, ValueType dt, 
    Funcft ft, Funcfx fx, Funcmap mapX, Funcmap mapY, ValueType* gllCoords,
    ValueType* B, ValueType* Bdiff,ValueType* G,ValueType* M,
    ValueType* u_tm1, ValueType* fku){
        /* Potentially this is only running in a single thread. 
            uses: original kernels. 
            This is called by the name cuda1.
        
        */
        // ValueType* g_Forces,

    // ValueType ft_dt = params.ft(it * params.dt());
    ValueType ft_dt = ft(it*dt);
    // size_t ndofelement = (pD + 1)*(pD + 1);
    // static_assert(MAX_PSIZE >= (pD+1)*(pD+1), "MAX_PSIZE too small");
    
    //instead of blockDim we might need nelements
    size_t blockid = blockIdx.x;// * blockDim.x + threadIdx.x;
    size_t ielement = blockid % nelements_0;
    size_t jelement = blockid / nelements_0;
    // if (blockid ==1 && it > 690) printf("f(%lu)=%f",it,ft_dt);
    
    if (ielement+jelement*nelements_0 <nelements){
        ValueType Fe_element[MAX_PSIZE];// = new ValueType[ndofelement];
        ValueType Fi_element[MAX_PSIZE]; //= new ValueType[ndofelement];
        //initialize Fi_element
        for (size_t i=0;i<ndofelement;i++) 
                Fi_element[i]=static_cast<ValueType>(0);

        ValueType u_m[MAX_PSIZE];// = new ValueType[ndofelement];
        size_t locationMap[MAX_PSIZE];// = new size_t[ndofelement];

        createLocationMapcuda(locationMap,pD,ndofdirection1,
            ielement,jelement);
        slice_kernel(u_m, u_tm1, locationMap,ndofelement);

        alg_84_kernel_singleThread(dim, B,Bdiff, M, u_m,pD,Fi_element);

        get_Force_xy_singleThread(fx,mapX,mapY,
            Fe_element,ielement,jelement,gllCoords,pD);

        alg_82_kernel_singleThread(dim,B,G,Fe_element,pD);
        for (size_t i = 0; i < ndofelement; ++i) {
            ValueType val = dt*dt * (Fe_element[i] * ft_dt - Fi_element[i]);
            assert(locationMap[i] < ndofglobal);
            atomicAdd(&fku[locationMap[i]], val);
        }
    }
}

template <typename Funcft, typename Funcfx, typename Funcmap,typename ValueType>
__global__ void solve_fKu_pDegreeThread(size_t it,size_t dim,size_t pD, size_t ndofelement, 
    size_t ndofdirection1,size_t nelements_0, size_t nelements, 
    size_t ndofglobal, ValueType dt, 
    Funcft ft, Funcfx fx, Funcmap mapX, Funcmap mapY, ValueType* gllCoords,
    ValueType* B, ValueType* Bdiff,ValueType* G,ValueType* M,
    ValueType* u_tm1, ValueType* fku){
        /* This is the latest cuda implementatoin. potentially uses the number of (pD+1)^2
            as the size of the SM. This means that (pD+1)^2 threads are inside the 
        */
    
    int tid = threadIdx.x;

    __shared__ ValueType ft_dt;
    if (tid ==0) ft_dt = ft(it*dt);

    size_t blockid = blockIdx.x;
    size_t ielement = blockid % nelements_0;
    size_t jelement = blockid / nelements_0;
    
    if (ielement+jelement*nelements_0 < nelements){
        __shared__ ValueType Fe_element[MAX_PSIZE];// = new ValueType[ndofelement];
        __shared__ ValueType Fi_element[MAX_PSIZE]; //= new ValueType[ndofelement];
        __shared__ ValueType u_m[MAX_PSIZE];// = new ValueType[ndofelement];
        __shared__ size_t locationMap[MAX_PSIZE];// = new size_t[ndofelement];
        //initialize Fi_element
        Fi_element[tid]=0.0;
        __syncthreads();

        if (tid == 0 ) createLocationMapcuda(locationMap,pD,ndofdirection1,
            ielement,jelement);
        if (tid == 0) slice_kernel(u_m, u_tm1, locationMap,ndofelement);
        __syncthreads();
        // if (tid == 0 && blockid ==0) printf("Location map and slice done correctly");
        alg_84_kernel_pDegreeThread(dim, B,Bdiff, M, u_m,pD,Fi_element);

        // if (tid == 0) get_Force_xy_singleThread(fx,mapX,mapY,
        //     Fe_element,ielement,jelement,gllCoords,pD);
        get_Force_xy_pDegreeThread(fx,mapX,mapY,
            Fe_element,ielement,jelement,gllCoords,pD);
        __syncthreads();
        // if (tid == 0 && blockid ==0)  printf("up to force xy done correctly");

        // if (tid == 0) 
        alg_82_kernel_pDegreeThread(dim,B,G,Fe_element,pD);
        __syncthreads();
        // if (tid == 0 && blockid ==0)  printf("Starting now the global assembly");

        // for (size_t i = 0; i < ndofelement; ++i) {
        if (tid<=ndofelement) {
            ValueType val = dt*dt * (Fe_element[tid] * ft_dt - Fi_element[tid]);
            assert(locationMap[tid] < ndofglobal);
            atomicAdd(&fku[locationMap[tid]], val);
        }
        // }
    }
}


template <typename Funcfx, typename FuncMap,typename ValueType>
__device__ void get_Force_xy_singleThread(Funcfx fx,FuncMap mapX, FuncMap mapY,
    ValueType* Fe_element,size_t ielement, size_t jelement, 
    const ValueType* gllCoords, size_t pD){

        for (size_t igauss = 0; igauss < pD+1; ++igauss) {
            ValueType r = gllCoords[igauss];
            for (size_t jgauss = 0; jgauss < pD+1; ++jgauss) {
                ValueType s = gllCoords[jgauss];
                ValueType x = mapX(ielement, r); 
                ValueType y = mapY(jelement, s);
                // Add contributions
                Fe_element[igauss * (pD+1) + jgauss] = fx(x, y);
        }
    }
}

template <typename Funcfx, typename FuncMap,typename ValueType>
__device__ void get_Force_xy_pDegreeThread(Funcfx fx,FuncMap mapX, FuncMap mapY,
    ValueType* Fe_element,size_t ielement, size_t jelement, 
    const ValueType* gllCoords, size_t pD){

    int tid = threadIdx.x;

    size_t igauss = tid / (pD+1);
    size_t jgauss = tid % (pD+1);

    ValueType r = gllCoords[igauss];
    ValueType s = gllCoords[jgauss];
    ValueType x = mapX(ielement, r); 
    ValueType y = mapY(jelement, s);

    Fe_element[igauss * (pD+1) + jgauss] = fx(x, y);
}


template <typename ValueType>
void test_mass_singleThread(std::vector<ValueType>& mu,
    const cie::spectralkernel::Params& params, size_t num_blocks) {
        check_polynomialDegree_gpu(params.polynomialDegree());
        /*
        Only one way to implement the mass. 
        Future work will require a variable int ver to test different 
        versions.
        This function is for testing mainly. 
        for device call use solve_Mass_singleThread function
        */
    std::fill(mu.begin(),mu.end(),0.0);

    ValueType fac =  params.rho() * ( params.dx()/ 2.0 ) * ( params.dy() / 2.0);
    ValueType dif = 0.0;
    
    size_t polynomialDegree = params.polynomialDegree();
    size_t nelements = params.nelements(1)*params.nelements(0);
    size_t P_size = (polynomialDegree + 1)*(polynomialDegree + 1);
    size_t ndofelement = params.ndofelement();
    size_t dim = 2;
    size_t ndofglobal = params.ndofglobal();

    adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1); 
    Matrixt<ValueType> S = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), dif);
    Matrixt<ValueType> R = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), dif);


    ValueType* B =new ValueType[P_size*2];
    cie::otFlyGPU::concatenate_flat(B, S, R,polynomialDegree + 1,dim); //pD+1,dim         

    ValueType* G = new ValueType[P_size];
    kron_flat(G,gauss.getWeights(), gauss.getWeights(), fac , (polynomialDegree+1));
    // ValueType f = 1.0;


    ValueType* d_B;
    ValueType* d_G;
    ValueType* d_mu;//no need to copy host2device
    //alocate

    ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_B,
            sizeof(ValueType) * ((polynomialDegree+1)*(polynomialDegree+1)*2)));
    // ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_locationMap, sizeof(ValueType) * ndofelement ) );
    ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_G, sizeof(ValueType) * P_size ) );
    ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_mu, sizeof(ValueType) * ndofglobal ) );

    //copy data from Host to Device
    ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_B, B,
    sizeof(ValueType) * (P_size*2),
    cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_G, G,
    sizeof(ValueType) * P_size,
    cudaMemcpyHostToDevice));

    //execute kernel
    // HelloKernel<<<num_blocks,1>>>(dim,polynomialDegree,params.ndofelement(),
    // params.ndofdirection()[1],params.nelements(0),nelements,
    // params.ndofglobal(),d_locationMap,d_B,d_G,d_mu);
    // ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize() );
    
    // std::cout<<nelements<<"<-elements,blocks->"<<num_blocks<<std::endl;


    solve_Mass_singleThread<ValueType><<<nelements,1>>>(dim,
        polynomialDegree,
        params.ndofelement(),
        params.ndofdirection()[1],
        params.nelements(0),
        nelements,
        params.ndofglobal(),
        d_B,
        d_G,
        d_mu
    );
    ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize() );
    //copy back data.
    ASSERT_NO_CUDA_ERRORS(cudaMemcpy(mu.data(), d_mu,
    sizeof(ValueType) * ndofglobal,
    cudaMemcpyDeviceToHost));

    // this function will delete now the data. 
    //in a normal computation we might don't want to do this as
    //sending back data will be expensive. We would probably only free up
    //the mu but remain the rest intact. 
    cudaFree(d_B);
    cudaFree(d_G);
    cudaFree(d_mu);
    delete[] B;
    delete[] G;

}

template void test_mass_singleThread<float>(std::vector<float>&, 
    const cie::spectralkernel::Params&,size_t);
// template void test_mass_singleThread<double>(std::vector<double>&,
// const cie::spectralkernel::Params&,size_t);

template <typename ValueType>
__global__ void solve_Mass_singleThread(size_t dim,size_t pD, size_t ndofelement,
    size_t ndofdirection1,size_t nelements_0, size_t nelements, 
    size_t ndofglobal, ValueType* B, ValueType* G, ValueType* mu){
        /* 
        TODO: Check if cu_singleThread should apply here. 
        */
        // size_t ndofelement = (pD + 1)*(pD + 1);
        //instead of blockDim we might need nelements
        size_t blockid = blockIdx.x;// * blockDim.x + threadIdx.x;
        size_t ielement = blockid % nelements_0;
        size_t jelement = blockid / nelements_0;
        
        if (ielement+jelement*nelements_0 <nelements){
            size_t* locationMap = new size_t[ndofelement];

            createLocationMapcuda(locationMap,pD,ndofdirection1,
                ielement,jelement);
            ValueType* M_el= new ValueType[ndofelement];
            ValueType v = 1.0;
            for (size_t m = 0;m<ndofelement;m++) M_el[m]=v;      
            alg_82_kernel_singleThread(dim,B,G,M_el,pD);
            for (size_t i = 0; i < ndofelement; ++i) {
                atomicAdd(&mu[locationMap[i]], M_el[i]);
            }
            __syncthreads();

            delete[] locationMap;
            delete[] M_el;
    }
}
        

template <typename ValueType>
__global__ void test_alg_84(const size_t D, 
    const ValueType* B,
    const ValueType* Bdiff,
    const ValueType* M,
    ValueType* v, 
    size_t pD,ValueType* r,int version) {
    if (version ==1){
        alg_84_kernel_singleThread<ValueType>(D,B,Bdiff,M,v,pD,r);
    }
    else if (version ==2){
        alg_84_kernel_pDegreeThread<ValueType>(D,B,Bdiff,M,v,pD,r);
    }

    }
template __global__ void test_alg_84<float>(const size_t,
     const float*, const float*,const float*, float*,size_t, float*,int);



template <typename ValueType>
__device__ void alg_84_kernel_singleThread(const size_t D, 
    const ValueType* B,
    const ValueType* Bdiff,
    const ValueType* M,
    ValueType* v, 
    size_t pD,ValueType* r) {

    size_t p_size=(pD+1)*(pD+1);
    // ValueType* w1 = new ValueType[vsize];
    // ValueType* Bd_diff= new ValueType[vsize*D];
    // printf("CUDA alg 84");
    ValueType w1[MAX_PSIZE];
    ValueType Bd_diff[MAX_PSIZE];

    for (size_t di = 0; di < D; ++di) {
        for(size_t ri=0;ri<p_size;ri++) w1[ri]=v[ri];

        for(size_t i = 0; i < D; ++i) {
            if (i == di) {
                alg_84_mergeBd_kernel_singleThread(Bd_diff, Bdiff,i,pD+1,pD+1,D);
                // std::cout<<di<<"<-di, i = "<<i<<std::endl;
            }
            else {
                alg_84_mergeBd_kernel_singleThread(Bd_diff, B, i,pD+1,pD+1,D);
                // std::cout<<"i = "<<i<<std::endl;
            }
        }
        alg_57_kernel_singleThread<ValueType>(D, Bd_diff, w1,pD,p_size);

        vectorwise_mul(M, w1,p_size);

        // ValueType* Bd_diff2 = swapaxes_copy_kernel_singleThread(Bd_diff,D,pD+1,pD+1);
        swapaxes_inplace_kernel_singleThread(Bd_diff,D,pD+1,pD+1);
        alg_57_kernel_singleThread<ValueType>(D, Bd_diff, w1,pD,p_size);

        vectorwise_sum(w1,r,p_size);

        // delete[] Bd_diff2;
    }
    // delete[] w1;
    // delete[] Bd_diff;

}
template __device__ void alg_84_kernel_singleThread<float>(const size_t, 
                const float*, const float*, const float*, float*,  size_t ,float*);
                
template <typename ValueType>
__device__ void alg_84_kernel_pDegreeThread(const size_t D, 
            const ValueType* B,
            const ValueType* Bdiff,
            const ValueType* M,
            ValueType* v, 
            size_t pD,ValueType* r) {
    int tid = threadIdx.x;
    size_t p_size=(pD+1)*(pD+1);
    __shared__ ValueType w1[MAX_PSIZE];
    __shared__ ValueType Bd_diff[MAX_PSIZE];
    __shared__ ValueType swapcache[MAX_PSIZE*2];

    for (size_t di = 0; di < D; ++di) {
        if (tid < p_size) w1[tid] = v[tid];

        for(size_t i = 0; i < D; ++i) {
            if (i == di) {
                alg_84_mergeBd_kernel_pDegreeThread(Bd_diff, Bdiff,i,pD+1,pD+1,D);
            }
            else {
                alg_84_mergeBd_kernel_pDegreeThread(Bd_diff, B, i,pD+1,pD+1,D);
            }
        }
        __syncthreads();

        alg_57_kernel_pDegreeThread<ValueType>(D, Bd_diff, w1,pD,p_size);

        if (tid <= p_size) w1[tid] *=M[tid];
        __syncthreads();

        swapaxes_inplace_kernel_pDegreeThread(swapcache,Bd_diff,D,pD+1,pD+1);

        alg_57_kernel_pDegreeThread<ValueType>(D, Bd_diff, w1,pD,p_size);

        if (tid <= p_size) r[tid] +=w1[tid];
        __syncthreads();
    }
}
template __device__ void alg_84_kernel_pDegreeThread<float>(const size_t, 
                const float*, const float*, const float*, float*,  size_t ,float*);

template <typename ValueType>
__global__ void test_alg_84_mergeBd(ValueType* Bd_diff, const ValueType* Bdiff,
        const size_t t, size_t row, size_t col, size_t dim, int version) {

    if (version ==1){
        alg_84_mergeBd_kernel_singleThread(Bd_diff,Bdiff,t,row,col,dim);
    }
    else if (version ==2){
        alg_84_mergeBd_kernel_pDegreeThread(Bd_diff,Bdiff,t,row,col,dim);
    }
}
template __global__ void test_alg_84_mergeBd<float>(float*,const float*,
        const size_t,size_t,size_t,size_t,int);

template <typename ValueType>
__device__ void alg_84_mergeBd_kernel_singleThread(ValueType* Bd_diff, const ValueType* Bdiff,
        const size_t j, size_t row, size_t col, size_t dim) {
    for (size_t t = 0; t < row; ++t) {
        for (size_t i = 0; i < col; ++i) {
            Bd_diff[t*col*dim + i*dim + j] = Bdiff[t*col*dim + i*dim + j];
        }
    }
}
template __device__ void alg_84_mergeBd_kernel_singleThread<float>(float*,const float*,
        const size_t,size_t,size_t,size_t);

template <typename ValueType>
__device__ void alg_84_mergeBd_kernel_pDegreeThread(ValueType* Bd_diff, const ValueType* Bdiff,
        const size_t j, size_t row, size_t col, size_t dim) {

    int tid = threadIdx.x;
    size_t tt = tid / col;
    size_t ii = tid % col;

    size_t dst_idx = tt * col * dim + ii * dim + j;
    
    Bd_diff[dst_idx] = Bdiff[dst_idx];
}
template __device__ void alg_84_mergeBd_kernel_pDegreeThread<float>(float*,const float*,
        const size_t,size_t,size_t,size_t);

template <typename ValueType>
__global__ void test_alg_82(const size_t D, 
                    const ValueType* B,
                    const ValueType* G,
                    ValueType* v, 
                    size_t pD, 
                    int version) {
    if (version ==1){
        alg_82_kernel_singleThread(D,B,G,v,pD);
    }
    else if (version ==2){
        alg_82_kernel_pDegreeThread(D,B,G,v,pD);
    }
}
template __global__ void test_alg_82<float>(const size_t, const float*, const float*, float*,size_t,int);
template __global__ void test_alg_82<double>(const size_t, const double*, const double*, double*,size_t,int);

template <typename ValueType>
__device__ void alg_82_kernel_singleThread(const size_t D, 
                    const ValueType* B,
                    const ValueType* G,
                    ValueType* v, 
                    size_t pD) {
    size_t vsize=(pD+1)*(pD+1);
    alg_57_kernel_singleThread(D, B, v,pD,vsize);
    vectorwise_mul(G,v,vsize);
    ValueType* B2 = swapaxes_copy_kernel_singleThread(B,D,pD+1,pD+1);
    alg_57_kernel_singleThread(D, B2,v,pD,vsize);
    delete[] B2;
}
template __device__ void alg_82_kernel_singleThread<float>(const size_t, const float*, const float*, float*,size_t);
template __device__ void alg_82_kernel_singleThread<double>(const size_t, const double*, const double*, double*,size_t);

template <typename ValueType>
__device__ void alg_82_kernel_pDegreeThread(const size_t D, 
                    const ValueType* B,
                    const ValueType* G,
                    ValueType* v, 
                    size_t pD) {
    int tid = threadIdx.x;

    size_t p_size=(pD+1)*(pD+1);
    __shared__ ValueType B2[MAX_PSIZE*2];//(P.size(), 0.0);

    alg_57_kernel_pDegreeThread(D, B, v,pD,p_size);

    // vectorwise_mul(G,v,p_size);
    if (tid <= p_size) v[tid] *=G[tid];
    __syncthreads();

    swapaxes_copy_kernel_pDegreeThread(B2,B,D,pD+1,pD+1);

    alg_57_kernel_pDegreeThread(D, B2,v,pD,p_size);
    // delete[] B2;
}
template __device__ void alg_82_kernel_pDegreeThread<float>(const size_t, const float*, const float*, float*,size_t);
template __device__ void alg_82_kernel_pDegreeThread<double>(const size_t, const double*, const double*, double*,size_t);



template <typename ValueType>
__global__ void testalg_57(const size_t D, 
                                ValueType* R, 
                                ValueType* P,
                                size_t pD, 
                                size_t p_size,
                                int version){
    int bsize = blockDim.x;
    if (version ==1){
        alg_57_kernel_singleThread(D,R,P,pD,p_size);
    }
    else if (version ==2 && bsize >= p_size){
        // if (threadIdx.x == 0) printf("\t\t -->57.2\n");
        alg_57_kernel_pDegreeThread(D,R,P,pD,p_size);
    }


} 
template __global__ void testalg_57<float>(const size_t, float*, float*,size_t,size_t,int);
template __global__ void testalg_57<double>(const size_t, double*, double*,size_t,size_t,int);

template <typename ValueType>
__device__ void alg_57_kernel_singleThread(const size_t D, 
                    const ValueType* R, 
                    ValueType* P,size_t pD, size_t p_size) {
    size_t s = pow(pD+1, D);
    ValueType* T= new ValueType[p_size];//(P.size(), 0.0);
    ValueType* Pfraction = new ValueType[pD+1];
    ValueType* Rfraction= new ValueType[(pD+1)*(pD+1)];
    //nullptr;//new ValueType*[pD+1];
    ValueType* val_T= new ValueType[pD+1];

    // genVV2(&Rfraction,pD+1,pD+1);//initialize matrix
    // std::fill(T, T + p_size, 0.0);
    for (size_t t=0;t<p_size;t++) T[t]=0.0;//initialize T

    size_t stride = (s / (pD+1));
    for (size_t i = 0; i < D; ++i){
        for (size_t j = 0; j < stride; ++j){ 
            getPfraction_kernel(Pfraction,P,j,p_size,pD+1);
            getRfraction_kernel_singleThread(Rfraction,R,D,i,pD+1);
            size_t value_idx = 0;
            matvec_kernel(val_T,Rfraction,Pfraction, pD+1,pD+1);//Matrix*vector
            for (size_t inner = j; inner < s; inner += stride) {
                T[inner] = val_T[value_idx++];
            }
        }//# Fehler in Davide's Diss -> linke Seite beginnend bei j
        swappy_singleThread(P, T,p_size);
    }
    delete[] Pfraction;
    delete[] Rfraction; 
    delete[] T;
    delete[] val_T;
}
template __device__ void alg_57_kernel_singleThread<float>(const size_t,const  float*, float*,size_t,size_t);
template __device__ void alg_57_kernel_singleThread<double>(const size_t,const  double*, double*,size_t,size_t);



template <typename ValueType>
__device__ void alg_57_kernel_pDegreeThread(const size_t D,
                    const ValueType* R,
                    ValueType* P, size_t pD, size_t p_size) {
    int tid = threadIdx.x;

    size_t s = pow(pD+1, D);
    __shared__ ValueType T[MAX_PSIZE];
    __shared__ ValueType Rfraction[(MAX_pD+1)*(MAX_pD+1)];

    for (size_t k = tid; k < p_size; k += blockDim.x) T[k] = 0.0;
    __syncthreads();

    size_t stride = s / (pD+1);

    for (size_t i = 0; i < D; ++i){
        getRfraction_kernel_pDegreeThread(Rfraction,R,D,i,pD+1);   // all threads: tid < (pD+1)^2 each fill one entry
        __syncthreads();

        for (size_t j = tid; j < stride; j += blockDim.x){
            for (size_t row = 0; row <= pD; ++row){
                ValueType sum = 0.0;
                for (size_t col = 0; col <= pD; ++col)
                    sum += Rfraction[row*(pD+1) + col] * P[j*(pD+1) + col];
                T[j + row*stride] = sum;
            }
        }
        __syncthreads();
        swappy_pDegreeThread(P, T, p_size);
        __syncthreads();
    }
                    }
template __device__ void alg_57_kernel_pDegreeThread<float>(const size_t,const  float*, float*,size_t,size_t);
template __device__ void alg_57_kernel_pDegreeThread<double>(const size_t,const  double*, double*,size_t,size_t);




template <typename ValueType>
__device__ void matvec_kernel(ValueType* result, 
                    ValueType* Mat, ValueType* vector, 
                    size_t r, size_t c){
    for (size_t iRow = 0; iRow < r; ++iRow){
        result[iRow] = 0.0;
        for (size_t jCol = 0; jCol < c; ++jCol){
            result[iRow] += Mat[iRow*c+jCol] * vector[jCol];
        }
    }
}
template __device__ void matvec_kernel<float>(float*,float*, float*,size_t,size_t);
template __device__ void matvec_kernel<double>(double*,double*, double*,size_t,size_t);




template <typename ValueType>
__device__ void vectorwise_mul(const ValueType* R,ValueType* s,size_t p_size) {   
    for (size_t i = 0; i < p_size; ++i) s[i] *= R[i];
}
template __device__ void vectorwise_mul<float>(const float*,float*, size_t);
template __device__ void vectorwise_mul<double>(const double*,double*, size_t);

template <typename ValueType>
__device__ void vectorwise_sum(const ValueType* R,ValueType* s,size_t p_size) {   
    for (size_t i = 0; i < p_size; ++i) s[i] += R[i];
}
template __device__ void vectorwise_sum<float>(const float*,float*, size_t);
template __device__ void vectorwise_sum<double>(const double*,double*, size_t);

template <typename ValueType>
__device__ ValueType* swapaxes_copy_kernel_singleThread(const ValueType* B,size_t dim, size_t row, size_t col) {
    //  trying to reproduce np.swapaxes(A,0,1);
    // this is only if t and i is the same number!!!
    //     meaning this only works for algorithm57
    // runtime_check(A.size() == A[0].size(),
        // "Dimension problem and number of Matrices not equal to number of rows!");
    ValueType* A = new ValueType[col*row*dim];
    // std::copy(B, B + col * row * dim, A);
    getRcopy_kernel(A,B,dim,row,col);

    for (size_t t = 0; t < row; ++t) {
        for (size_t i = 0; i < col; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                A[t*col*dim+i*dim+j] = B[i*col*dim+t*dim+j];
            }
        }
    }
    return A;
}
template __device__ float* swapaxes_copy_kernel_singleThread<float>(const float*,size_t,size_t,size_t);
template __device__ double* swapaxes_copy_kernel_singleThread<double>(const double*,size_t,size_t,size_t);

template <typename ValueType>
__device__ void swapaxes_copy_kernel_pDegreeThread(ValueType* A, const ValueType* B,
    size_t dim, size_t row, size_t col) {

    // ValueType* A = new ValueType[col*row*dim];
    // __shared__ ValueType A[MAX_PSIZE*2];
    int tid = threadIdx.x;
    size_t total = row * col;// * dim; //use with dim when using p_size * 2 threads

    if (tid >= total) return;

    size_t t = tid / (col);
    size_t i = (tid) % col;

    size_t src_idx = i * col * dim + t * dim ;
    size_t dst_idx = t * col * dim + i * dim ;
    for (size_t j = 0;j<dim; j++){
        A[dst_idx+j] = B[src_idx+j];
    }
    __syncthreads();

}
template __device__ void swapaxes_copy_kernel_pDegreeThread<float>(float*,const float*,size_t,size_t,size_t);
template __device__ void swapaxes_copy_kernel_pDegreeThread<double>(double*,const double*,size_t,size_t,size_t);



template <typename ValueType>
__device__ void swapaxes_inplace_kernel_singleThread(ValueType* B,size_t dim, size_t row, size_t col) {
    //  trying to reproduce np.swapaxes(A,0,1);
    // this is only if t and i is the same number!!!
    //     meaning this only works for algorithm57
    ValueType* A = new ValueType[col*row*dim];
    getRcopy_kernel(A,B,dim,row,col);

    for (size_t t = 0; t < row; ++t) {
        for (size_t i = 0; i < col; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                B[t*col*dim+i*dim+j] = A[i*col*dim+t*dim+j];
            }
        }
    }
    delete[] A;
}
template __device__ void swapaxes_inplace_kernel_singleThread<float>(float*,size_t,size_t,size_t);
template __device__ void swapaxes_inplace_kernel_singleThread<double>(double*,size_t,size_t,size_t);

template <typename ValueType>
__device__ void swapaxes_inplace_kernel_pDegreeThread(ValueType* A, ValueType* B,size_t dim, size_t row, size_t col) {

    // ValueType* A = new ValueType[col*row*dim];
    // __shared__ ValueType A[MAX_PSIZE*2];
    int tid = threadIdx.x;
    size_t total = row * col;// * dim; //use with dim when using p_size * 2 threads

    if (tid >= total) return;

    size_t t = tid / (col);
    size_t i = (tid) % col;

    for (size_t j = 0; j < dim; j++) {
        int idx = t * col * dim + i * dim + j;
        A[idx] = B[idx];
    }
    __syncthreads();
    //              Transposing...
    // size_t j = tid % dim; // this could even go up to p_size * 2 threads...
    // Do the axis-transposed copy
    size_t src_idx = i * col * dim + t * dim ;
    size_t dst_idx = t * col * dim + i * dim ;
    for (size_t j = 0;j<dim; j++){
        B[dst_idx+j] = A[src_idx+j];
    }
    __syncthreads();

}
template __device__ void swapaxes_inplace_kernel_pDegreeThread<float>(float*,float*,size_t,size_t,size_t);
template __device__ void swapaxes_inplace_kernel_pDegreeThread<double>(double*,double*,size_t,size_t,size_t);


__global__ void testlocationMap(size_t* LocationMap,
            const size_t pD, 
            size_t ndofdirection1,
            size_t ielement, 
            size_t jelement){
    createLocationMapcuda(LocationMap,pD,ndofdirection1,ielement,jelement);
} 

__device__ void createLocationMapcuda(size_t* LocationMap, 
        const size_t pD, size_t ndofdirection1, 
        size_t ielement, size_t jelement){

    // size_t* dofIndicesI = (size_t*)malloc((pD+1) * sizeof(size_t));
    // size_t* dofIndicesJ = (size_t*)malloc((pD+1) * sizeof(size_t));
    size_t dofIndicesI[MAX_pD + 1];
    size_t dofIndicesJ[MAX_pD + 1];
    // extern __shared__ size_t sharedMem[];
    // size_t* dofIndicesI = sharedMem;  
    // size_t* dofIndicesJ = sharedMem + (polynomialDegree + 1);

    arangecu(dofIndicesI, pD + 1, pD * ielement);
    arangecu(dofIndicesJ, pD + 1, pD * jelement);

    for (size_t i = 0; i < pD+1; ++i) {
        for (size_t j = 0; j < pD+1; ++j) {
            LocationMap[i*(pD+1) + j] = dofIndicesI[i]*ndofdirection1 + dofIndicesJ[j];
        }
    }

}

__device__ void arangecu(size_t* outV,size_t number,size_t addition) {
    for (size_t i = 0; i < number; ++i) {
        outV[i]=i+addition;
    }
}

template <typename ValueType>
__device__ void getRfraction_kernel_pDegreeThread(ValueType* Rfraction, const ValueType* R, 
                size_t dim,size_t i, size_t n_i) {
    int tid = threadIdx.x;
    int Ri = tid / n_i;
    int Rj = tid % n_i;
    if (tid < n_i * n_i) {
        Rfraction[Ri*n_i+Rj] = R[Ri*n_i*dim+Rj*dim+i];
    }
}
template __device__ void getRfraction_kernel_pDegreeThread<float>(float*,const  float*,size_t,size_t,size_t);
template __device__ void getRfraction_kernel_pDegreeThread<double>(double*, const double*,size_t,size_t,size_t);

template <typename ValueType>
__device__ void swappy_pDegreeThread(ValueType* P, ValueType* T, size_t p_size){
    int tid = threadIdx.x;
    if (tid < p_size){
        ValueType temp = P[tid];
        P[tid] = T[tid];
        T[tid] = temp;
    }
    __syncthreads();
}
template __device__ void swappy_pDegreeThread<float>(float*, float*, size_t);
template __device__ void swappy_pDegreeThread<double>(double*, double*, size_t);


template <typename ValueType>
__device__ void slice_kernel(ValueType* u_slice,const ValueType* u, 
    const size_t* locM, size_t ndofelement){
    for (size_t si =0;si<ndofelement;si++) {
        u_slice[si] = u[locM[si]];
    }
}
template __device__ void slice_kernel<double>(double*,const double*, const size_t*, size_t);
template __device__ void slice_kernel<float>(float*, const float*, const size_t*, size_t);
    

template <typename ValueType>
__global__ void HelloKernel(size_t dim,size_t pD, size_t ndofelement, size_t ndofdirection1,
    size_t nelements_0, size_t nelements, 
    size_t ndofglobal, size_t* locationMap, ValueType* B, ValueType* G, ValueType* mu) {

    size_t blockid = blockIdx.x;// * blockDim.x + threadIdx.x;
    size_t ielement = blockid % nelements_0;
    size_t jelement = blockid / nelements_0;
    size_t elementid = jelement * nelements_0 +ielement;
    printf("GPU thread %d, block %d doing elementid = %lu, iel=%lu jel=%lu\n",
    threadIdx.x, blockIdx.x, elementid,ielement,jelement);
}
template __global__ void HelloKernel<float>(size_t, size_t, size_t,
    size_t, size_t,size_t,size_t,size_t*,
    float*,float*,float*);

template <typename ValueType>
__host__ void writeU_inter(std::vector<ValueType>& u, std::string filename) {
    // std::cout << "Starting writing on file..." ;

    std::ofstream file(filename, std::ios::app); 
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    file << std::setprecision(8);   // same as the CPU writer

    for (size_t i = 0; i < u.size(); ++i) {
        file << u[i];
        if (i != u.size() - 1) {
            file << ",";
        }
    }
    file << "\n"; 
    // std::cout << "Exiting writing on file..." << std::endl;
}
template __host__ void writeU_inter<float>(std::vector<float>&, std::string);

template <typename ValueType>
__host__ void writeU_inter(thrust::host_vector<ValueType>& u, std::string filename) {
    // std::cout << "Starting writing on file..." ;

    std::ofstream file(filename, std::ios::app); 
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    file << std::setprecision(8);   // same as the CPU writer

    for (size_t i = 0; i < u.size(); ++i) {
        file << u[i];
        if (i != u.size() - 1) {
            file << ",";
        }
    }
    file << "\n"; 
    // std::cout << "Exiting writing on file..." << std::endl;
}
template __host__ void writeU_inter<float>(thrust::host_vector<float>&, std::string);

template struct lambdaXY_kernel<float>;
template struct lambdaXY_kernel<double>;

template struct lambdaft_kernel<float>;
template struct lambdaft_kernel<double>;

template struct lambdafx_kernel<float>;
template struct lambdafx_kernel<double>;

}

}