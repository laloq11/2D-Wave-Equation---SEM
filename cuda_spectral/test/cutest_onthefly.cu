#include "catch.hpp"

#include "alias.hpp"

// #include "ontheFly.hpp" 
#include "LagrangeBasisFunctions.hpp"
#include "GaussLobatto.hpp" 
// #include "basisfunctions.hpp" 
#include <fstream>
#include <cmath>
#include <algorithm>

#include "cubase.hpp"
#include "cuontheFly.cuh"
#include <cuda_runtime.h>

__global__ void addKernel(int *a, int *b, int *c) {
    int idx = threadIdx.x;
    c[idx] = a[idx] + b[idx];
}



namespace cie {
namespace otFlyGPU {
using ValueType = float;

TEST_CASE("CUDAcreateLocationMapcu", "[cuda]"){
    printf("------------------------------------------\n");
    printf("CUDA create location map test\n");
	ValueType eps = 1e-12;
    size_t nx = 10;
    size_t ny = 10;
    size_t pD = 2;
    cie::spectralkernel::Params params = 
    cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(nx,ny,pD);
	// ValueType rho = params.rho();
	// ValueType dx = params.dx();
	// ValueType dy = params.dy();
	// ValueType fac = (rho * (dx/2.0) * (dy/2.0));
    size_t ndofdirection1 = params.ndofdirection()[1];

    std::vector<size_t> toChallenge((pD+1)*(pD+1),0.0);

    size_t* d_toChallenge;

    size_t a_size = ((pD+1)*(pD+1));

    REQUIRE(cudaMalloc(&d_toChallenge, sizeof(size_t) * ((pD+1)*(pD+1)))
        == cudaSuccess);

    for (size_t i = 0;i<10;i++){
        for (size_t j = 0;j<10;j++){
            std::vector<size_t> expected = reference::createLocationMapcu(pD,
            ndofdirection1,i,j);

            testlocationMap<<<1,1>>>(d_toChallenge,pD,ndofdirection1,i,j);
            REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

            REQUIRE(cudaMemcpy(toChallenge.data(), d_toChallenge,
            sizeof(size_t) * a_size,
            cudaMemcpyDeviceToHost)== cudaSuccess);

            for (size_t k =0;k<(pD+1)*(pD+1);k++){
                CHECK(toChallenge[k] == Approx(expected[k]).margin(eps));
            }
        }
    }
    cudaFree(d_toChallenge);

}

TEST_CASE("CUDAalg_57", "[cuda]"){
	std::cout<<"CUDA entering test for alg57"<<std::endl;
	// initialization
	ValueType eps = 1e-12;

	ValueType rho = 1.0;
	ValueType dx = 1.0;
	ValueType dy = 1.0;
	ValueType fac = (rho * (dx/2.0) * (dy/2.0));

	//G
	size_t polynomialDegree = 2;
	adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1);
	size_t pD1=polynomialDegree+1;

	ValueType* G = new ValueType[(polynomialDegree+1)*(polynomialDegree+1)];
	kron_flat(G,gauss.getWeights(), gauss.getWeights(), fac , (polynomialDegree+1));
	// B
	Matrixt<ValueType> R1({ {0.22007998, 0.26355171},
		{0.77065457, 0.91751922},
		{0.05943222, 0.93462959} });

	Matrixt<ValueType> R2({ {0.16564331, 0.59244317},
		{0.70164211, 0.32450572},
		{0.29389116, 0.22698156} });

	Matrixt<ValueType> R3({ {0.1430465 , 0.41063075},
		{0.94203222, 0.11368537},
		{0.1389887 , 0.65897952} });
	std::vector<Matrixt<ValueType>> Rs = {R1,R2,R3};
	ValueType* R = new ValueType[(polynomialDegree + 1)*(polynomialDegree + 1)*2];
	for (size_t v = 0; v < pD1; ++v) {
		for (size_t c = 0; c < pD1; ++c) {
			for (size_t r = 0; r < 2; ++r) {
				R[v*pD1*2+c*2+r] = Rs[v][c][r];
			}
		}
	}

	size_t P_size= (polynomialDegree + 1)*(polynomialDegree + 1);
    ValueType* P = new ValueType[P_size];
    for (size_t i=0;i<P_size;i++) P[i]=1;

	std::vector<ValueType> result = {2.2218384, 2.4567019, 2.58976008,
								1.20131775, 1.32830525,
								1.40024799, 1.24265777, 
								1.37401518, 1.44843364};
	// alg_57_flat<ValueType>(2, R, P, polynomialDegree,P_size);

    std::vector<ValueType> P2(P_size,1.0);
    std::vector<ValueType> P3(P_size,1.0);
    ValueType* d_R;
    ValueType* d_P;
    ValueType* d_P3;
        //alocate
    REQUIRE(cudaMalloc(&d_R, sizeof(ValueType) * (P_size*2))
    == cudaSuccess);
    REQUIRE(cudaMalloc(&d_P, sizeof(ValueType) * P_size ) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_P3, sizeof(ValueType) * P_size ) == cudaSuccess);
        //copy data
    REQUIRE(cudaMemcpy(d_R, R,
    sizeof(ValueType) * (P_size*2),
    cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE(cudaMemcpy(d_P, P2.data(),
    sizeof(ValueType) * P_size,
    cudaMemcpyHostToDevice)== cudaSuccess);
    REQUIRE(cudaMemcpy(d_P3, P3.data(),
    sizeof(ValueType) * P_size,
    cudaMemcpyHostToDevice)== cudaSuccess);
        //execute kernel
    
    testalg_57<ValueType><<<1,1>>>(2,d_R,d_P,polynomialDegree,P_size,1);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    testalg_57<ValueType><<<1,P_size>>>(2,d_R,d_P3,polynomialDegree,P_size,2);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        //copy back data.
    REQUIRE(cudaMemcpy(P2.data(), d_P,
    sizeof(ValueType) * P_size,
    cudaMemcpyDeviceToHost)== cudaSuccess);
    REQUIRE(cudaMemcpy(P3.data(), d_P3,
    sizeof(ValueType) * P_size,
    cudaMemcpyDeviceToHost)== cudaSuccess);


	for (size_t i = 0; i < P_size; ++i) {
		CHECK(P2[i] == Approx(result[i]).margin(eps));
	}
	for (size_t i = 0; i < P_size; ++i) {
		CHECK(P3[i] == Approx(result[i]).margin(eps));
	}
    delete[] R;
	delete[] G;
    cudaFree(d_R);
    cudaFree(d_P);
    cudaFree(d_P3);

}

TEST_CASE("CUDAalg_82", "[cuda]"){
	std::cout<<"CUDA entering test for alg82"<<std::endl;

	// initialization
	ValueType eps = 1e-12;
	ValueType rho = 1.0;
	ValueType dx = 1.0;
	ValueType dy = 1.0;
	ValueType fac = (rho * (dx/2.0) * (dy/2.0));
	size_t dif = 0.0;
	//G
	size_t polynomialDegree = 2;
	adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1);
	// std::vector<ValueType> G = kroncu(gauss.getWeights(), gauss.getWeights(),fac);
	// ValueType* G = kron_flat(gauss.getWeights(), gauss.getWeights(), fac );
	ValueType* G = new ValueType[(polynomialDegree+1)*(polynomialDegree+1)];
	kron_flat(G,gauss.getWeights(), gauss.getWeights(), fac , (polynomialDegree+1));
	// B
	Matrixt<ValueType> S = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), dif);
	Matrixt<ValueType> R = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), dif);

	// std::vector<Matrixt<ValueType>> B = cie::otFlyGPU::concatenate_matrices(S, R);
	ValueType* B =new ValueType[(polynomialDegree + 1)*(polynomialDegree + 1)*2];
	cie::otFlyGPU::concatenate_flat(B, S, R,polynomialDegree + 1,2); //pD+1,dim 
	// std::vector<ValueType> P = { 1., 1., 1., 1., 1., 1., 1., 1., 1. };
	size_t P_size = (polynomialDegree + 1)*(polynomialDegree + 1);
	ValueType f = 1.0;
	ValueType* P = new ValueType[P_size];
	std::fill(P,P+P_size,f);

	ValueType* P2 = new ValueType[P_size];
	std::fill(P2,P2+P_size,f);
	size_t dim = 2;
	// alg_82_flat<ValueType>(dim,B,G,P,polynomialDegree);


    ValueType* d_B;
    ValueType* d_P;
    ValueType* d_P2;
    ValueType* d_G;
        //alocate
    REQUIRE(cudaMalloc(&d_B, sizeof(ValueType) * ((polynomialDegree+1)*(polynomialDegree+1)*2))
    == cudaSuccess);
    REQUIRE(cudaMalloc(&d_P, sizeof(ValueType) * P_size ) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_P2, sizeof(ValueType) * P_size ) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_G, sizeof(ValueType) * P_size ) == cudaSuccess);
        //copy data
    REQUIRE(cudaMemcpy(d_B, B,
    sizeof(ValueType) * ((polynomialDegree+1)*(polynomialDegree+1)*2),
    cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE(cudaMemcpy(d_P, P,
    sizeof(ValueType) * P_size,
    cudaMemcpyHostToDevice)== cudaSuccess);
    REQUIRE(cudaMemcpy(d_P2, P2,
    sizeof(ValueType) * P_size,
    cudaMemcpyHostToDevice)== cudaSuccess);
    REQUIRE(cudaMemcpy(d_G, G,
    sizeof(ValueType) * P_size,
    cudaMemcpyHostToDevice)== cudaSuccess);

        //execute kernel
    test_alg_82<ValueType><<<1,1>>>(2,d_B,d_G,d_P,polynomialDegree,1); // alg_82_kernel
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

        //copy back data.
    REQUIRE(cudaMemcpy(P, d_P,
    sizeof(ValueType) * P_size,
    cudaMemcpyDeviceToHost)== cudaSuccess);

            //execute kernel
    test_alg_82<ValueType><<<1,P_size>>>(2,d_B,d_G,d_P2,polynomialDegree,2); // alg_82_kernel2
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

        //copy back data.
    REQUIRE(cudaMemcpy(P2, d_P2,
    sizeof(ValueType) * P_size,
    cudaMemcpyDeviceToHost)== cudaSuccess);

	std::vector<ValueType> result82 = { 0.02777778, 0.11111111, 0.02777778, 0.11111111, 0.44444444,
		0.11111111, 0.02777778, 0.11111111, 0.02777778 };

	for (size_t i = 0; i < (polynomialDegree + 1)*(polynomialDegree + 1); ++i) {
		CHECK(P[i] == Approx(result82[i]).margin(eps));
	}
	for (size_t i = 0; i < (polynomialDegree + 1)*(polynomialDegree + 1); ++i) {
		CHECK(P2[i] == Approx(result82[i]).margin(eps));
	}
    cudaFree(d_B);
	cudaFree(d_G);
	cudaFree(d_P);
	cudaFree(d_P2);

}
TEST_CASE("CUDA_Mass", "[cuda]"){
	std::cout<<"CUDA entering final MASS CUDA test"<<std::endl;
	ValueType eps = 1e-12;

	size_t nx = 10;
    size_t ny = 10;
    size_t PolDegree = 2;
    // cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(nx,ny,PolDegree);
    
    cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(nx,ny,PolDegree);

	std::vector<ValueType> mu_ref(params.ndofglobal(),0.0);

	std::vector<ValueType> mu_cuda(params.ndofglobal(),0.0);
	
	size_t nelements = params.nelements(1)*params.nelements(0);

    test_mass_singleThread(mu_cuda, params,nelements);
    reference::compute_Mass_ref(mu_ref, params);
	for (size_t i = 0; i < params.ndofglobal(); ++i) {
		CHECK(mu_cuda[i] == Approx(mu_ref[i]).margin(eps));
	}

}
    // ValueType* r = new ValueType[vsize];
    // for (size_t ri=0;ri<vsize;ri++) r[ri]=0;


TEST_CASE("Algorithm84_CUDA") {
	std::cout<<"Entering alg84 CUDA"<<std::endl;
	// initialization
	// ValueType eps = 1e-12;
	ValueType eps = 1e-12;

	ValueType rho = 1.0;
	ValueType dx =  1.0;
	ValueType dy =  1.0;
	ValueType c = 1.0;
	ValueType fac_m = (rho * (dx/2.0) * (dy/2.0) * c * c);
	ValueType facS = (2.0/dx);
	ValueType facR = (2.0/dy);
	size_t dim =2;

	//G
	size_t polynomialDegree = 2;
	adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1);

	size_t P_size = (polynomialDegree+1)*(polynomialDegree+1);

	ValueType* M = new ValueType[P_size];
	kron_flat(M,gauss.getWeights(), gauss.getWeights(), fac_m , (polynomialDegree+1));

	// std::vector<double> M = kron(gauss.getWeights(), gauss.getWeights(),
	// 	(rho * (dx / 2.0) * (dy / 2.0)));
	// B

	Matrixt<ValueType> S = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), 0);
	Matrixt<ValueType> R = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), 0);


	Matrixt<ValueType> Sdiff = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), 1);
	Matrixt<ValueType> Rdiff = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), 1);
	cie::otFlyGPU::ShapeFunctionFactor(Sdiff,facS);
	cie::otFlyGPU::ShapeFunctionFactor(Rdiff,facR);


	ValueType* B = new ValueType[P_size*dim];
	cie::otFlyGPU::concatenate_flat(B, S, R,polynomialDegree+1,2); //pD+1,dim 
	ValueType* Bdiff = new ValueType[P_size*dim];
	cie::otFlyGPU::concatenate_flat(Bdiff, Sdiff, Rdiff,polynomialDegree+1,dim); //pD+1,dim 

	std::vector<ValueType> v = { 0.04234617, 0.9941601 , 0.39384291, 0.78736183, 0.86274709,
		0.25537733, 0.18095811, 0.12164062, 0.25661357 };
	ValueType* r = new ValueType[P_size];
    for (size_t ri=0;ri<P_size;ri++) r[ri]=0;
	ValueType* r2 = new ValueType[P_size];
    for (size_t ri=0;ri<P_size;ri++) r2[ri]=0;
	//GPU initialization step
    ValueType* d_B;
    ValueType* d_Bdiff;
    ValueType* d_M;
    ValueType* d_v;
    ValueType* d_r;
    ValueType* d_r2;
	//alocatememory
	
    REQUIRE(cudaMalloc(&d_B, sizeof(ValueType) * (P_size*2))
    == cudaSuccess);
    REQUIRE(cudaMalloc(&d_Bdiff, sizeof(ValueType) * (P_size*2))
    == cudaSuccess);
    REQUIRE(cudaMalloc(&d_v, sizeof(ValueType) * P_size ) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_r, sizeof(ValueType) * P_size ) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_r2, sizeof(ValueType) * P_size ) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_M, sizeof(ValueType) * P_size ) == cudaSuccess);
	
	//copy data
    REQUIRE(cudaMemcpy(d_B, B,
    sizeof(ValueType) * (P_size*2),
    cudaMemcpyHostToDevice) == cudaSuccess);

	REQUIRE(cudaMemcpy(d_Bdiff, Bdiff,
		sizeof(ValueType) * (P_size*2),
		cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE(cudaMemcpy(d_v, v.data(),
    sizeof(ValueType) * P_size,
    cudaMemcpyHostToDevice)== cudaSuccess);
    REQUIRE(cudaMemcpy(d_r, r,
    sizeof(ValueType) * P_size,
    cudaMemcpyHostToDevice)== cudaSuccess);
    REQUIRE(cudaMemcpy(d_r2, r2,
    sizeof(ValueType) * P_size,
    cudaMemcpyHostToDevice)== cudaSuccess);
    REQUIRE(cudaMemcpy(d_M, M,
    sizeof(ValueType) * P_size,
    cudaMemcpyHostToDevice)== cudaSuccess);


    test_alg_84<ValueType><<<1,P_size>>>(dim,d_B,d_Bdiff,d_M,
		d_v,polynomialDegree,d_r2,2); //alg_84_kernel_pDegreeThread
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    test_alg_84<ValueType><<<1,1>>>(dim,d_B,d_Bdiff,d_M,
		d_v,polynomialDegree,d_r,1); // alg_84_kernel
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);



	//copy back data.
    REQUIRE(cudaMemcpy(r, d_r,
    sizeof(ValueType) * P_size,
    cudaMemcpyDeviceToHost)== cudaSuccess);

    REQUIRE(cudaMemcpy(r2, d_r2,
    sizeof(ValueType) * P_size,
    cudaMemcpyDeviceToHost)== cudaSuccess);

	std::vector<double> result84 = { -0.72691823,  0.72956597, -0.23241883,  0.34839383,  2.29768615,
		-1.02363937, -0.24664702, -1.2099807 ,  0.06395819 };

	for (size_t i = 0; i < P_size; ++i) {
		CHECK(r[i] == Approx(result84[i]).margin(eps));
	}
	for (size_t i = 0; i < P_size; ++i) {
		CHECK(r2[i] == Approx(result84[i]).margin(eps));
    }
	cudaFree(d_B);
	cudaFree(d_Bdiff);
	cudaFree(d_M);
	cudaFree(d_v);
	cudaFree(d_r);
	cudaFree(d_r2);
}
TEST_CASE("globalforces CUDA check") {
// 	std::cout<<"global forces for CUDA!!"<<std::endl;

// 	size_t nx = 5;
//     size_t ny = 5;
//     size_t polynomialDegree = 2;
// 	ValueType epsilon = 1e-6;

//     // cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(nx,ny,PolDegree);

//     cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(nx,ny,polynomialDegree);
//     size_t ndofelement = params.ndofelement();
//     size_t ndofdirection1 = params.ndofdirection()[1];

//     size_t ndofglobal = params.ndofglobal();
//     size_t P_size = (polynomialDegree + 1)*(polynomialDegree + 1);


// 	std::vector<ValueType> global_Forces(params.ndofglobal(), 0.0); 
//     get_global_ForcesV<ValueType>(params,global_Forces);

//     ValueType* d_Forces;
//     ValueType* de_Fe_el;
//     size_t* d_lM;
// //    ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_Forces, sizeof(ValueType)  ) );
// 	REQUIRE(cudaMalloc(&d_Forces, sizeof(ValueType) * ndofglobal )
// 	== cudaSuccess);
// 	REQUIRE(cudaMalloc(&de_Fe_el, sizeof(ValueType) * P_size)
// 	== cudaSuccess);
// 	REQUIRE(cudaMalloc(&d_lM, sizeof(size_t) * P_size)
// 	== cudaSuccess);

//     // ValueType* Fe_element = new ValueType[P_size];
// 	std::vector<ValueType> Fe_element(P_size); 

//     REQUIRE(cudaMemcpy(d_Forces, global_Forces.data(),
//         sizeof(ValueType) * ndofglobal,
//         cudaMemcpyHostToDevice)== cudaSuccess);

// 	for (size_t iel =0;iel<params.nelements(0);iel++){
// 		for (size_t jel =0;jel<params.nelements(1);jel++){
//             std::vector<size_t> locationMap = createLocationMapcu(polynomialDegree,
// 				ndofdirection1,
// 				iel, 
// 				jel);

// 			REQUIRE(cudaMemcpy(d_lM, locationMap.data(),
// 				sizeof(size_t) * P_size,
// 				cudaMemcpyHostToDevice)== cudaSuccess);

// 			get_elementforce_test<<<1,1>>> (de_Fe_el,d_Forces,
// 				 d_lM,ndofelement);

// 			REQUIRE( cudaDeviceSynchronize() == cudaSuccess);

// 			REQUIRE( cudaMemcpy(Fe_element.data(), de_Fe_el,
// 					sizeof(ValueType) * P_size,
// 					cudaMemcpyDeviceToHost)== cudaSuccess);

// 			for (size_t igauss =0;igauss<polynomialDegree+1;igauss++){
// 				double r = params.gllcoords()[igauss];
// 				for (size_t jgauss =0;jgauss<polynomialDegree+1;jgauss++){
// 					double s = params.gllcoords()[jgauss];
//                     size_t ii = igauss * (params.polynomialDegree()+1) + jgauss;
//                     double x = params.mapX(iel, r); 
//                     double y = params.mapY(jel, s);

// 					ValueType force_ref = params.fx(x, y);

// 					CHECK(force_ref == Approx(Fe_element[ii]).margin(epsilon));
// 				}
// 			}
// 		}
// 	}
// 	cudaFree(de_Fe_el);
// 	cudaFree(d_Forces);
// 	cudaFree(d_lM);
}


TEST_CASE("fKuCUDATEST"){
	std::cout<<"fKu test for CUDA."<<std::endl;

	ValueType eps = 1e-6;

	size_t nx = 5;
    size_t ny = 5;
    size_t PolDegree = 2;
    // cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(nx,ny,PolDegree);

    
    cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(nx,ny,PolDegree);

	// std::vector<ValueType> mu_ref(params.ndofglobal(),0.0);
	// std::vector<ValueType> u_ref(params.ndofglobal(),1.0);
	size_t it = 10;

	std::vector<ValueType> u_ref = {1.31354E-07,1.90615E-09,
		-3.17975E-10,-1.92196E-12,4.6561E-13,1.58387E-15,
		-3.88901E-16,-7.57143E-19,1.87365E-19,2.0661E-22,
		-1.02767E-22,1.90615E-09,2.94127E-11,-5.04772E-12,
		-2.68456E-14,6.53901E-15,1.84045E-17,-4.53498E-18,
		-7.10054E-21,1.76103E-21,1.51925E-24,-7.56655E-25,
		-3.17975E-10,-5.04772E-12,8.75856E-13,4.50633E-15,
		-1.09963E-15,-2.88146E-18,7.11167E-19,9.88753E-22,
		-2.45592E-22,-1.72194E-25,8.58823E-26,-1.92196E-12,
		-2.68456E-14,4.50633E-15,1.75986E-17,-4.32529E-18,
		-7.98344E-21,1.97983E-21,1.72883E-24,-4.30883E-25,
		-1.43195E-28,7.15976E-29,4.6561E-13,6.53901E-15,
		-1.09963E-15,-4.32529E-18,1.06293E-18,1.97449E-21,
		-4.89633E-22,-4.3E-25,1.07169E-25,3.57988E-29,
		-1.78994E-29,1.58387E-15,1.84045E-17,-2.88146E-18,
		-7.98344E-21,1.97449E-21,2.30295E-24,-5.73529E-25,
		-2.38659E-28,5.96647E-29,0,0,-3.88901E-16,-4.53498E-18,
		7.11167E-19,1.97983E-21,-4.89633E-22,-5.73529E-25,1.4283E-25,
		5.96647E-29,-1.49162E-29,0,0,-7.57143E-19,-7.10054E-21,
		9.88753E-22,1.72883E-24,-4.3E-25,-2.38659E-28,5.96647E-29,
		0,0,0,0,1.87365E-19,1.76103E-21,-2.45592E-22,-4.30883E-25,
		1.07169E-25,5.96647E-29,-1.49162E-29,0,0,0,0,2.0661E-22,
		1.51925E-24,-1.72194E-25,-1.43195E-28,3.57988E-29,0,0,0,0,0,
		0,-1.02767E-22,-7.56655E-25,8.58823E-26,7.15976E-29,-1.78994E-29
		,0,0,0,0,0,0
	};

	std::vector<ValueType> fKu_ref={1.13614E-10,9.70356E-12,
		-8.58399E-13,-1.96511E-14,2.31746E-15,2.92214E-17,
		-3.53704E-18,-2.4613E-20,3.02103E-21,1.19125E-23,
		-1.47424E-24,9.70356E-12,9.73875E-13,-9.05095E-14,
		-1.75392E-15,2.09513E-16,2.05871E-18,-2.51034E-19,
		-1.34681E-21,1.66014E-22,5.01922E-25,-6.22693E-26,
		-8.58399E-13,-9.05095E-14,8.4918E-15,1.63431E-16,
		-1.9558E-17,-1.84608E-19,2.2549E-20,1.12713E-22,
		-1.39146E-23,-3.75262E-26,4.66191E-27,-1.96511E-14,
		-1.75392E-15,1.63431E-16,2.287E-18,-2.77271E-19,
		-1.83574E-21,2.25939E-22,7.57965E-25,-9.40306E-26,
		-1.54214E-28,1.92217E-29,2.31746E-15,2.09513E-16,
		-1.9558E-17,-2.77271E-19,3.36053E-20,2.24766E-22,
		-2.76602E-23,-9.35542E-26,1.16054E-26,1.91637E-29,
		-2.38859E-30,2.92214E-17,2.05871E-18,-1.84608E-19,
		-1.83574E-21,2.24766E-22,1.00829E-24,-1.24845E-25,
		-2.56629E-28,3.19655E-29,0,0,-3.53704E-18,-2.51034E-19,
		2.2549E-20,2.25939E-22,-2.76602E-23,-1.24845E-25,
		1.54575E-26,3.19373E-29,-3.97803E-30,0,0,-2.4613E-20,
		-1.34681E-21,1.12713E-22,7.57965E-25,-9.35542E-26,
		-2.56629E-28,3.19373E-29,0,0,0,0,3.02103E-21,1.66014E-22,
		-1.39146E-23,-9.40306E-26,1.16054E-26,3.19655E-29,
		-3.97803E-30,0,0,0,0,1.19125E-23,5.01922E-25,-3.75262E-26,
		-1.54214E-28,1.91637E-29,0,0,0,0,0,0,-1.47424E-24,-6.22693E-26,
		4.66191E-27,1.92217E-29,-2.38859E-30,0,0,0,0,0,0
	};



	std::vector<ValueType> fKu_cuda(params.ndofglobal(),0.0);

    test_fku_solverVersion(it, params, u_ref, fKu_cuda,1);

	// the values are ~1e-10, so the margin must scale with them (F5)
	ValueType scale = 0.0;
	for (auto v : fKu_ref) scale = std::max(scale, std::abs(v));
	for (size_t i = 0; i < params.ndofglobal(); ++i) {
		CHECK(fKu_cuda[i] == Approx(fKu_ref[i]).epsilon(1e-4).margin(1e-4 * scale));
	}
}

TEST_CASE("fKuCUDATEST2"){
	std::cout<<"fKu test for CUDA parallelized."<<std::endl;

	ValueType eps = 1e-6;

	size_t nx = 5;
    size_t ny = 5;
    size_t PolDegree = 2;
    // cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(nx,ny,PolDegree);

    
    cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(nx,ny,PolDegree);

	// std::vector<ValueType> mu_ref(params.ndofglobal(),0.0);
	// std::vector<ValueType> u_ref(params.ndofglobal(),1.0);
	size_t it = 10;

	std::vector<ValueType> u_ref = {1.31354E-07,1.90615E-09,
		-3.17975E-10,-1.92196E-12,4.6561E-13,1.58387E-15,
		-3.88901E-16,-7.57143E-19,1.87365E-19,2.0661E-22,
		-1.02767E-22,1.90615E-09,2.94127E-11,-5.04772E-12,
		-2.68456E-14,6.53901E-15,1.84045E-17,-4.53498E-18,
		-7.10054E-21,1.76103E-21,1.51925E-24,-7.56655E-25,
		-3.17975E-10,-5.04772E-12,8.75856E-13,4.50633E-15,
		-1.09963E-15,-2.88146E-18,7.11167E-19,9.88753E-22,
		-2.45592E-22,-1.72194E-25,8.58823E-26,-1.92196E-12,
		-2.68456E-14,4.50633E-15,1.75986E-17,-4.32529E-18,
		-7.98344E-21,1.97983E-21,1.72883E-24,-4.30883E-25,
		-1.43195E-28,7.15976E-29,4.6561E-13,6.53901E-15,
		-1.09963E-15,-4.32529E-18,1.06293E-18,1.97449E-21,
		-4.89633E-22,-4.3E-25,1.07169E-25,3.57988E-29,
		-1.78994E-29,1.58387E-15,1.84045E-17,-2.88146E-18,
		-7.98344E-21,1.97449E-21,2.30295E-24,-5.73529E-25,
		-2.38659E-28,5.96647E-29,0,0,-3.88901E-16,-4.53498E-18,
		7.11167E-19,1.97983E-21,-4.89633E-22,-5.73529E-25,1.4283E-25,
		5.96647E-29,-1.49162E-29,0,0,-7.57143E-19,-7.10054E-21,
		9.88753E-22,1.72883E-24,-4.3E-25,-2.38659E-28,5.96647E-29,
		0,0,0,0,1.87365E-19,1.76103E-21,-2.45592E-22,-4.30883E-25,
		1.07169E-25,5.96647E-29,-1.49162E-29,0,0,0,0,2.0661E-22,
		1.51925E-24,-1.72194E-25,-1.43195E-28,3.57988E-29,0,0,0,0,0,
		0,-1.02767E-22,-7.56655E-25,8.58823E-26,7.15976E-29,-1.78994E-29
		,0,0,0,0,0,0
	};

	std::vector<ValueType> fKu_ref={1.13614E-10,9.70356E-12,
		-8.58399E-13,-1.96511E-14,2.31746E-15,2.92214E-17,
		-3.53704E-18,-2.4613E-20,3.02103E-21,1.19125E-23,
		-1.47424E-24,9.70356E-12,9.73875E-13,-9.05095E-14,
		-1.75392E-15,2.09513E-16,2.05871E-18,-2.51034E-19,
		-1.34681E-21,1.66014E-22,5.01922E-25,-6.22693E-26,
		-8.58399E-13,-9.05095E-14,8.4918E-15,1.63431E-16,
		-1.9558E-17,-1.84608E-19,2.2549E-20,1.12713E-22,
		-1.39146E-23,-3.75262E-26,4.66191E-27,-1.96511E-14,
		-1.75392E-15,1.63431E-16,2.287E-18,-2.77271E-19,
		-1.83574E-21,2.25939E-22,7.57965E-25,-9.40306E-26,
		-1.54214E-28,1.92217E-29,2.31746E-15,2.09513E-16,
		-1.9558E-17,-2.77271E-19,3.36053E-20,2.24766E-22,
		-2.76602E-23,-9.35542E-26,1.16054E-26,1.91637E-29,
		-2.38859E-30,2.92214E-17,2.05871E-18,-1.84608E-19,
		-1.83574E-21,2.24766E-22,1.00829E-24,-1.24845E-25,
		-2.56629E-28,3.19655E-29,0,0,-3.53704E-18,-2.51034E-19,
		2.2549E-20,2.25939E-22,-2.76602E-23,-1.24845E-25,
		1.54575E-26,3.19373E-29,-3.97803E-30,0,0,-2.4613E-20,
		-1.34681E-21,1.12713E-22,7.57965E-25,-9.35542E-26,
		-2.56629E-28,3.19373E-29,0,0,0,0,3.02103E-21,1.66014E-22,
		-1.39146E-23,-9.40306E-26,1.16054E-26,3.19655E-29,
		-3.97803E-30,0,0,0,0,1.19125E-23,5.01922E-25,-3.75262E-26,
		-1.54214E-28,1.91637E-29,0,0,0,0,0,0,-1.47424E-24,-6.22693E-26,
		4.66191E-27,1.92217E-29,-2.38859E-30,0,0,0,0,0,0
	};



	std::vector<ValueType> fKu_cuda(params.ndofglobal(),0.0);

    test_fku_solverVersion(it, params, u_ref, fKu_cuda,2);

	// the values are ~1e-10, so the margin must scale with them (F5)
	ValueType scale = 0.0;
	for (auto v : fKu_ref) scale = std::max(scale, std::abs(v));
	for (size_t i = 0; i < params.ndofglobal(); ++i) {
		CHECK(fKu_cuda[i] == Approx(fKu_ref[i]).epsilon(1e-4).margin(1e-4 * scale));
	}
}

// Both GPU versions against the CPU reference for every p the GPU accepts.
// p = 5 needs two warps per block in the pDegreeThread version, which is where the race of F1 showed.
TEST_CASE("fKu GPU versions vs CPU reference, p = 2..5", "[cuda]"){
	size_t ne = 4;
	size_t it = 10;
	for (size_t p = 2; p <= 5; ++p) {
		cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(ne, ne, p);
		size_t ndof = params.ndofglobal();

		std::vector<ValueType> u(ndof);
		for (size_t i = 0; i < ndof; ++i) u[i] = std::sin(0.37 * i) + 0.5;

		std::vector<ValueType> ref(ndof, 0.0);
		reference::compute_fku_ref(it, params, u, ref);

		for (int version = 1; version <= 2; ++version) {
			std::vector<ValueType> gpu(ndof, 0.0);
			test_fku_solverVersion(it, params, u, gpu, version);

			double diff = 0.0, norm = 0.0;
			for (size_t i = 0; i < ndof; ++i) {
				diff += double(gpu[i] - ref[i]) * double(gpu[i] - ref[i]);
				norm += double(ref[i]) * double(ref[i]);
			}
			INFO("p = " << p << ", version = " << version << ", rel. L2 = " << std::sqrt(diff / norm));
			REQUIRE(norm > 0.0);
			CHECK(std::sqrt(diff / norm) < 1e-5);
		}
	}
}



TEST_CASE("CUDA Kernel Test", "[cuda]") {
printf("basic test to see CUDA functionality\n");

const int N = 1;
int h_a[N] = {1}, h_b[N] = {2}, h_c[N];
int *d_a, *d_b, *d_c;

REQUIRE(cudaMalloc(&d_a, N * sizeof(int)) == cudaSuccess);
REQUIRE(cudaMalloc(&d_b, N * sizeof(int)) == cudaSuccess);
REQUIRE(cudaMalloc(&d_c, N * sizeof(int)) == cudaSuccess);

REQUIRE(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
REQUIRE(cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);

addKernel<<<1, N>>>(d_a, d_b, d_c);
REQUIRE(cudaDeviceSynchronize() == cudaSuccess);  // Ensure execution is complete

REQUIRE(cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);

REQUIRE(h_c[0] == 3);  // Assertion

// Cleanup
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
}
// From test_parameters.cpp found under spectralcm folder
TEST_CASE("MappingTestCUDA")
{
    printf("Testing MappingTestCUDA aka lambdaXY in ");
    printf("params but in cuda lambdaXY_kernel.\n");
    double eps = 1e-12;
    //initializing
    std::array<size_t, 2> nelements = { 5, 5 };
    std::array<double, 2> lengths = { 2.0, 2.0 }; //x, y dimension
    std::array<double, 2>center = { 1.0, 1.0 };

    double nsteps = 1000;
    size_t polynomialDegree = 2;

    double duration = 12;
    double radius = 0.4;
    double frequency = 1.0;
    double sigmaS = 0.06;
    double E = 1.0;
    double rho = 1.0;
    double domain = 1.0;
    double alpha = 1.0;
    double c = 1.0;

    Params params(nelements, lengths, center, E, rho, domain, alpha, c,
        duration, radius, frequency, sigmaS,
        nsteps, polynomialDegree);

    float ielement = 2;
    float jelement = 2;
    float s = params.gllcoords(1);
    float r = params.gllcoords(1);
    std::vector<float> result = { 1.0,1.0 };

    // CHECK(params.mapX(ielement, r) == Approx(result[0]).margin(eps));
    // CHECK(params.mapY(jelement, s) == Approx(result[1]).margin(eps));

    
    ValueType* kernel_rx;
    ValueType* kernel_ry;
    std::vector<ValueType> kernel_resx(1,0.0);
    std::vector<ValueType> kernel_resy(1,0.0);

    cie::otFlyGPU::lambdaXY_kernel<float> x_k{params.dx()}; 
    cie::otFlyGPU::lambdaXY_kernel<float> y_k{params.dy()}; 


    REQUIRE(cudaMalloc(&kernel_rx, sizeof(ValueType)) == cudaSuccess);
    REQUIRE(cudaMalloc(&kernel_ry, sizeof(ValueType)) == cudaSuccess);

    cie::otFlyGPU::test_lambdaXY<<<1, 1>>>(x_k,kernel_rx,ielement,r);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    cie::otFlyGPU::test_lambdaXY<<<1, 1>>>(y_k,kernel_ry,jelement,s);


    REQUIRE(
        cudaMemcpy(kernel_resx.data(),kernel_rx, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);
    REQUIRE(
        cudaMemcpy(kernel_resy.data(),kernel_ry, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);



    CHECK(kernel_resx[0] == Approx(result[0]).margin(eps));
    CHECK(kernel_resy[0] == Approx(result[1]).margin(eps));


    ielement = 4;
    jelement = 4;
    result = { 1.8,1.8 };

    cie::otFlyGPU::test_lambdaXY<<<1, 1>>>(x_k,kernel_rx,ielement,r);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    cie::otFlyGPU::test_lambdaXY<<<1, 1>>>(y_k,kernel_ry,jelement,s);

    REQUIRE(
        cudaMemcpy(kernel_resx.data(),kernel_rx, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);
    REQUIRE(
        cudaMemcpy(kernel_resy.data(),kernel_ry, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);
    CHECK(kernel_resx[0] == Approx(result[0]).margin(eps));
    CHECK(kernel_resy[0] == Approx(result[1]).margin(eps));

    ielement = 0;
    jelement = 4;
    s = params.gllcoords(2);
    r = params.gllcoords(1);
    result = { 0.2,2.0 };
    cie::otFlyGPU::test_lambdaXY<<<1, 1>>>(x_k,kernel_rx,ielement,r);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    cie::otFlyGPU::test_lambdaXY<<<1, 1>>>(y_k,kernel_ry,jelement,s);

    REQUIRE(
        cudaMemcpy(kernel_resx.data(),kernel_rx, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);
    REQUIRE(
        cudaMemcpy(kernel_resy.data(),kernel_ry, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);
    CHECK(kernel_resx[0] == Approx(result[0]).margin(eps));
    CHECK(kernel_resy[0] == Approx(result[1]).margin(eps));


    REQUIRE(cudaFree(kernel_ry)==cudaSuccess);
    REQUIRE(cudaFree(kernel_rx)==cudaSuccess);

}//MappingTest

TEST_CASE("spatialfunctiontestCUDA")
{
    double eps = 1e-8;
    //initializing
    std::array<size_t, 2> nelements = { 5, 5 };
    std::array<double, 2> lengths = { 2.0, 2.0 }; //x, y dimension
    std::array<double, 2>center = { 1.0, 1.0 };

    double nsteps = 1000;
    size_t polynomialDegree = 2;

    double duration = 12;
    double radius = 0.4;
    double frequency = 1.0;
    double sigmaS = 0.06;
    double E = 1.0;
    double rho = 1.0;
    double domain = 1.0;
    double alpha = 1.0;
    double c = 1.0;
    Params params(nelements, lengths, center, E, rho, domain, alpha, c,
        duration, radius, frequency, sigmaS,
        nsteps, polynomialDegree);

    ValueType ielement = 0;
    ValueType jelement = 0;
    ValueType s = params.gllcoords(0);
    ValueType r = params.gllcoords(0);
    ValueType result_x = 10;// { 1.0, 1.0 };

    ValueType* lambda_x;
    ValueType* lambda_y;
    ValueType* res_fx;
    std::vector<ValueType> lambda_resx(1,0.0);
    std::vector<ValueType> lambda_resy(1,0.0);
    std::vector<ValueType> lambda_resfx(1,0.0);

    cie::otFlyGPU::lambdaXY_kernel<ValueType> x_k{params.dx()}; 
    cie::otFlyGPU::lambdaXY_kernel<ValueType> y_k{params.dy()}; 


    REQUIRE(cudaMalloc(&lambda_x, sizeof(ValueType)) == cudaSuccess);
    REQUIRE(cudaMalloc(&lambda_y, sizeof(ValueType)) == cudaSuccess);

    cie::otFlyGPU::test_lambdaXY<<<1, 1>>>(x_k,lambda_x,ielement,r);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    cie::otFlyGPU::test_lambdaXY<<<1, 1>>>(y_k,lambda_y,jelement,s);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    REQUIRE(
        cudaMemcpy(lambda_resx.data(),lambda_x, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);
    REQUIRE(
        cudaMemcpy(lambda_resy.data(),lambda_y, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);


    cie::otFlyGPU::lambdafx_kernel<ValueType> fx{sigmaS}; 
    REQUIRE(cudaMalloc(&res_fx, sizeof(ValueType)) == cudaSuccess);

    cie::otFlyGPU::test_lambdafx<<<1, 1>>>(fx,res_fx,
        lambda_resx[0],lambda_resy[0]);

    REQUIRE(
        cudaMemcpy(lambda_resfx.data(),res_fx, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);

    CHECK(lambda_resfx[0] == Approx(result_x).margin(eps));



    //another test
    result_x = 0.038659201394728004;
    s = params.gllcoords(1);
    cie::otFlyGPU::test_lambdaXY<<<1, 1>>>(y_k,lambda_y,jelement,s);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(
        cudaMemcpy(lambda_resy.data(),lambda_y, sizeof(ValueType), 
                        cudaMemcpyDeviceToHost)== cudaSuccess);

    cie::otFlyGPU::test_lambdafx<<<1, 1>>>(fx,res_fx,
        lambda_resx[0],lambda_resy[0]);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    REQUIRE(
        cudaMemcpy(lambda_resfx.data(),res_fx, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);

    CHECK(lambda_resfx[0] == Approx(result_x).margin(eps));



    size_t P_size= (polynomialDegree+1)*(polynomialDegree+1);

    cie::otFlyGPU::test_lambdafx<<<1, P_size>>>(fx,res_fx,
    lambda_resx[0],lambda_resy[0]);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    REQUIRE(
        cudaMemcpy(lambda_resfx.data(),res_fx, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);

    CHECK(lambda_resfx[0] == Approx(result_x).margin(eps));

    //another test
    result_x = 2.23363143620315e-09;
    ielement = 1;
    jelement = 0;
    s = params.gllcoords(0);
    r = params.gllcoords(0);

    // x = params.mapX(ielement, r);
    // y = params.mapY(jelement, s);
    cie::otFlyGPU::test_lambdaXY<<<1, 1>>>(x_k,lambda_x,ielement,r);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    cie::otFlyGPU::test_lambdaXY<<<1, 1>>>(y_k,lambda_y,jelement,s);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    REQUIRE(
        cudaMemcpy(lambda_resx.data(),lambda_x, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);
    REQUIRE(
        cudaMemcpy(lambda_resy.data(),lambda_y, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);

    cie::otFlyGPU::test_lambdafx<<<1, 1>>>(fx,res_fx,
        lambda_resx[0],lambda_resy[0]);

    REQUIRE(
        cudaMemcpy(lambda_resfx.data(),res_fx, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);

    CHECK(lambda_resfx[0] == Approx(result_x).margin(eps));

    //another test
    result_x = 1.473562036913071e-308;
    ielement = 4;
    jelement = 4;
    s = params.gllcoords(0);
    r = params.gllcoords(0);
    // x = params.mapX(ielement, r);
    // y = params.mapY(jelement, s);
    cie::otFlyGPU::test_lambdaXY<<<1, 1>>>(x_k,lambda_x,ielement,r);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    cie::otFlyGPU::test_lambdaXY<<<1, 1>>>(y_k,lambda_y,jelement,s);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    REQUIRE(
        cudaMemcpy(lambda_resx.data(),lambda_x, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);
    REQUIRE(
        cudaMemcpy(lambda_resy.data(),lambda_y, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);

    result_x = 2.23363143620315e-09;
    cie::otFlyGPU::test_lambdafx<<<1, 1>>>(fx,res_fx,
        lambda_resx[0],lambda_resy[0]);

    REQUIRE(
        cudaMemcpy(lambda_resfx.data(),res_fx, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);

    CHECK(lambda_resfx[0] == Approx(result_x).margin(eps));


}//spatialfunctiontest
TEST_CASE("timefunctiontestCUDA")
{
    double eps = 1e-12;
    //initializing
    std::array<size_t, 2> nelements = { 5, 5 };
    std::array<double, 2> lengths = { 2.0, 2.0 }; //x, y dimension
    std::array<double, 2>center = { 1.0, 1.0 };

    double nsteps = 1000;
    size_t polynomialDegree = 2;

    double duration = 12;
    double radius = 0.4;
    double frequency = 1.0;
    double sigmaS = 0.06;
    double E = 1.0;
    double rho = 1.0;
    double domain = 1.0;
    double alpha = 1.0;
    double c = 1.0;

    Params params(nelements, lengths, center,E,rho,domain,alpha,c,
        duration, radius, frequency, sigmaS,
        nsteps, polynomialDegree);
        ValueType pi = 3.14159265358979323846;

    ValueType t0 = 1.0 / frequency;
    ValueType sigmaT = 1.0 / (2.0 * pi * frequency);
    ValueType* lambda_t;
    std::vector<ValueType> lambda_rest(1,0.0);
    cie::otFlyGPU::lambdaft_kernel<ValueType> ft_k{sigmaT,t0}; 

    REQUIRE(cudaMalloc(&lambda_t, sizeof(ValueType)) == cudaSuccess);


    ValueType t = .05;
    ValueType result_t = 1.723356746322617e-06;
    cie::otFlyGPU::test_lambdaft_kernel<<<1, 1>>>(ft_k,lambda_t,t);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(
        cudaMemcpy(lambda_rest.data(),lambda_t, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);
    CHECK(lambda_rest[0] == Approx(result_t).margin(eps));

    t = .005;
    result_t = 3.2074158468037215e-07;
    cie::otFlyGPU::test_lambdaft_kernel<<<1, 1>>>(ft_k,lambda_t,t);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(
        cudaMemcpy(lambda_rest.data(),lambda_t, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);
    CHECK(lambda_rest[0] == Approx(result_t).margin(eps));

    t = .25;
    result_t = 0.0011179100567117002;
    cie::otFlyGPU::test_lambdaft_kernel<<<1, 1>>>(ft_k,lambda_t,t);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(
        cudaMemcpy(lambda_rest.data(),lambda_t, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);
    CHECK(lambda_rest[0] == Approx(result_t).margin(eps));

    t = .75;
    result_t = 7.2044418165232935;
    cie::otFlyGPU::test_lambdaft_kernel<<<1, 1>>>(ft_k,lambda_t,t);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(
        cudaMemcpy(lambda_rest.data(),lambda_t, sizeof(ValueType), 
                            cudaMemcpyDeviceToHost)== cudaSuccess);
    CHECK(lambda_rest[0] == Approx(result_t).margin(eps));
}//timefunctiontest

}
}