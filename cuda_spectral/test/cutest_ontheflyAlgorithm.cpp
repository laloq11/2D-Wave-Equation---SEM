#include "catch.hpp"

#include "alias.hpp"

// #include "ontheFly.hpp" 
#include "LagrangeBasisFunctions.hpp"
#include "GaussLobatto.hpp" 
// #include "basisfunctions.hpp" 
#include <fstream>

#include "cubase.hpp"
#include "cuontheFly.cuh"


namespace cie {
namespace otFlyGPU {
	using ValueType = float;

TEST_CASE("kroneckertest"){
	std::cout<<"KRONECKER TESTING using ValueType styles"<<std::endl;

	ValueType eps = 1e-12;

	ValueType rho = 1.0;
	ValueType dx =  1.0;
	ValueType dy =  1.0;
	ValueType fac = (rho * (dx/2.0) * (dy/2.0));
	std::vector<std::vector<ValueType>> result_newaxis = { {1.,0.,	0.},
															{0.,	1.,	0.},
															{0.,0.,1.} };
	std::vector<ValueType> result_kron = { 0.02777778, 0.11111111, 0.02777778, 
										0.11111111, 0.44444444,0.11111111, 
										0.02777778, 0.11111111, 0.02777778 };
	size_t polynomialDegree = 2;
	adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1);


	ValueType* G = new ValueType[(polynomialDegree+1)*(polynomialDegree+1)];
	kron_flat(G,gauss.getWeights(), gauss.getWeights(), fac , (polynomialDegree+1));

	size_t G_size = (polynomialDegree + 1)*(polynomialDegree + 1);
	for (size_t t = 0; t < G_size; ++t) {
		CHECK(G[t] == Approx(result_kron[t]).margin(eps));
	}
	delete[] G;

}//kroneckertest

TEST_CASE("Algorithm57CUDATest") {
	std::cout<<"entering new testing for alg57"<<std::endl;
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
	reference::alg_57_flat<ValueType>(2, R, P, polynomialDegree,P_size);

	for (size_t i = 0; i < P_size; ++i) {
		CHECK(P[i] == Approx(result[i]).margin(eps));
	}

    delete[] R;
	delete[] G;

}

TEST_CASE("concatenateCUDATest") {
	std::cout<<"entering concatenate CUDA Test"<<std::endl;

	double eps = 1e-12;
	size_t dif = 0.0;

	// double rho = 1.0;
	// double dx = 1.0;
	// double dy = 1.0;

	Matrixt<ValueType> M1 ({ {1., 1.},
		{0., 0.},
		{0., 0.} });
	Matrixt<ValueType> M2 ({ {0., 0.},
	{1., 1.},
	{0., 0.} });
	Matrixt<ValueType> M3({ {0., 0.},
	{0., 0.},
	{1., 1.} });
	std::vector<Matrixt<ValueType>> result_concatenate = { M1, M2, M3 };

	size_t polynomialDegree = 2;
	adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1);
	Matrixt<ValueType> S = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), dif);
	Matrixt<ValueType> R = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), dif);
	ValueType* B = new ValueType[(polynomialDegree + 1)*(polynomialDegree + 1)*2];
	cie::otFlyGPU::concatenate_flat(B, S, R,polynomialDegree+1,2); //pD+1,dim 
	// std::cout<<"--------------------------------------------"<<std::endl;
	// std::cout<<"# matrices in vector: "<<B.size()<<std::endl;
	// std::cout<<"B rows :"<<B[0].size1()<<std::endl;
	// std::cout<<"B[0] cols :"<<B[0].size2()<<std::endl;
	// std::cout<<"--------------------------------------------"<<std::endl;
	size_t rows = result_concatenate[0].size();
	size_t cols = result_concatenate[0][0].size();
	for (size_t v = 0; v < result_concatenate.size(); ++v) {
		for (size_t c = 0; c < rows; ++c) {
			for (size_t r = 0; r < cols; ++r) {
				CHECK(B[v*cols*rows+c*cols +r] == Approx(result_concatenate[v][c][r]).margin(eps));
			}
		}
	}
	delete[] B;
	// freeConcatenated(&B, polynomialDegree + 1); // Deallocate

}//concatenateTest
TEST_CASE("CONCATENATETEST2") {
	Matrixt<ValueType> M1({ {-3., -3.},{ 4.,  4. },{ -1., -1. } });
	Matrixt<ValueType> M2({ {-1., -1.},{ 0.,  0.},{ 1.,  1.} });
	Matrixt<ValueType> M3({ { 1.,  1.},{-4., -4.},{ 3.,  3.} });
	double eps = 1e-12;

	// ValueType rho = 1.0;
	ValueType dx =  1.0;
	ValueType dy =  1.0;
	ValueType facS = (2.0/dx);
	ValueType facR = (2.0/dy);

	std::vector<Matrixt<ValueType>> result_concatenate = { M1, M2, M3 };

	size_t polynomialDegree = 2;
	adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1);
	Matrixt<ValueType> S = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), 1);
	Matrixt<ValueType> R = cie::otFlyGPU::get_ShapesDiffcu<ValueType>(gauss.getCoordinates(), 1);
	cie::otFlyGPU::ShapeFunctionFactor(S,facS);
	cie::otFlyGPU::ShapeFunctionFactor(R,facR);
	ValueType* B =new ValueType[(polynomialDegree + 1)*(polynomialDegree + 1)*2];
	cie::otFlyGPU::concatenate_flat(B, S, R,polynomialDegree + 1,2); //pD+1,dim 
	// std::vector<Matrixt<ValueType>> B = cie::otFlyGPU::concatenate_matrices(S, R);
	size_t rows = result_concatenate[0].size();
	size_t cols = result_concatenate[0][0].size();
	for (size_t v = 0; v < result_concatenate.size(); ++v) {
		for (size_t c = 0; c < rows; ++c) {
			for (size_t r = 0; r < cols; ++r) {
				CHECK(B[v*cols*rows+c*cols +r] == Approx(result_concatenate[v][c][r]).margin(eps));
			}
		}
	}
	delete[] B;

}//concatenateTest2
TEST_CASE("swapaxesTest") {
	std::cout<<"entering SWAP CUDA Test"<<std::endl;

	double eps = 1e-12;

	Matrix R1({ {1,2}  ,{3,4}  ,{5,6} });
	Matrix R2({ {7,8}  ,{9,10} ,{11,12} });
	Matrix R3({ {13,14},{15,16},{17,18} });

	std::vector<Matrix> R = { R1,R2,R3 };
	ValueType* R2swap = new ValueType[3*3*2];
	for (size_t v = 0; v < 3; ++v) {
		for (size_t c = 0; c < 3; ++c) {
			for (size_t r = 0; r < 2; ++r) {
				R2swap[v*3*2+c*2+r] = R[v](c,r);
			}
		}
	}
	ValueType* Rswap= reference::swapaxes_flat(R2swap,2,3,3);


	Matrix R1n({ {1,2}  ,{7,8}  ,{13,14} });
	Matrix R2n({ {3,4}  ,{9,10} ,{15,16} });
	Matrix R3n({ {5,6},{11,12},{17,18} });
	std::vector<Matrix> Rresult = { R1n,R2n,R3n };

	for (size_t t = 0; t < Rresult.size(); ++t) {
		for (size_t i = 0; i < Rresult[0].size1(); ++i) {
			for (size_t j = 0; j < Rresult[0].size2(); ++j) {
				CHECK(Rswap[t*3*2+i*2 +j] == Approx(Rresult[t](i,j)).margin(eps));

			}
		}
	}
}
TEST_CASE("Algorithm82CUDATest") {

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
	size_t dim = 2;
	reference::alg_82_flat<ValueType>(dim,B,G,P,polynomialDegree);

	std::vector<ValueType> result82 = { 0.02777778, 0.11111111, 0.02777778, 0.11111111, 0.44444444,
		0.11111111, 0.02777778, 0.11111111, 0.02777778 };

	for (size_t i = 0; i < (polynomialDegree + 1)*(polynomialDegree + 1); ++i) {
		CHECK(P[i] == Approx(result82[i]).margin(eps));
	}

}

TEST_CASE("MassCUDATEST"){
	ValueType eps = 1e-12;

	size_t nx = 10;
    size_t ny = 10;
    size_t PolDegree = 2;
    // cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(nx,ny,PolDegree);

    
    cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(nx,ny,PolDegree);

	std::vector<ValueType> mu_ref(params.ndofglobal(),0.0);

    reference::compute_Mass_ref(mu_ref, params);

	std::vector<ValueType> mu_cuda(params.ndofglobal(),0.0);

    reference::compute_Mass_ref(mu_cuda, params);



	for (size_t i = 0; i < params.ndofglobal(); ++i) {
		CHECK(mu_cuda[i] == Approx(mu_ref[i]).margin(eps));
	}
}

TEST_CASE("Algorithm84_valuetype") {
	std::cout<<"entering new testing for alg84 value type"<<std::endl;
	// initialization
	ValueType eps = 1e-12;

	ValueType rho = 1.0;
	ValueType dx =  1.0;
	ValueType dy =  1.0;
	ValueType c = 1.0;
	ValueType fac = (rho * (dx/2.0) * (dy/2.0) * c * c);
	ValueType facS = (2.0/dx);
	ValueType facR = (2.0/dy);
	size_t dim =2;
	//G
	size_t polynomialDegree = 2;

	size_t P_size = (polynomialDegree+1)*(polynomialDegree+1);

	adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1);

	ValueType* M = new ValueType[(polynomialDegree+1)*(polynomialDegree+1)];
	kron_flat(M,gauss.getWeights(), gauss.getWeights(), fac , (polynomialDegree+1));

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
	cie::otFlyGPU::concatenate_flat(B, S, R,polynomialDegree+1,dim); //pD+1,dim 
	ValueType* Bdiff = new ValueType[P_size*dim];
	cie::otFlyGPU::concatenate_flat(Bdiff, Sdiff, Rdiff,polynomialDegree+1,dim); //pD+1,dim 

	std::vector<ValueType> v = { 0.04234617, 0.9941601 , 0.39384291, 0.78736183, 0.86274709,
		0.25537733, 0.18095811, 0.12164062, 0.25661357 };

	ValueType* r = new ValueType[P_size];
    for (size_t ri=0;ri<P_size;ri++) r[ri]=0;
	reference::alg_84_flat(2, B,Bdiff, M, v.data(),polynomialDegree,r);

	std::vector<double> result84 = { -0.72691823,  0.72956597, -0.23241883,  0.34839383,  2.29768615,
		-1.02363937, -0.24664702, -1.2099807 ,  0.06395819 };

	for (int i = 0; i < P_size; ++i) {
		CHECK(r[i] == Approx(result84[i]).margin(eps));
	}
}

TEST_CASE("globalforces check") {
	std::cout<<"global forces for  vector type."<<std::endl;

	size_t nx = 5;
    size_t ny = 5;
    size_t PolDegree = 2;
	ValueType epsilon = 1e-6;

    // cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(nx,ny,PolDegree);
	size_t vsize = (PolDegree+1)*(PolDegree+1);

    cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(nx,ny,PolDegree);
    size_t ndofelement = params.ndofelement();
    size_t ndofdirection1 = params.ndofdirection()[1];



	std::vector<ValueType> global_Forces(params.ndofglobal(), 0.0); 
    get_global_ForcesV<ValueType>(params,global_Forces);

    // ValueType* Fe_element = new ValueType[vsize];
	std::vector<ValueType> Fe_element(vsize); 

	for (size_t iel =0;iel<params.nelements(0);iel++){
		for (size_t jel =0;jel<params.nelements(1);jel++){
            std::vector<size_t> locationMap = reference::createLocationMapcu(PolDegree,
				ndofdirection1,
				iel, 
				jel);
			reference::get_elementforce<ValueType>(Fe_element.data(),global_Forces.data(),
							locationMap.data(),ndofelement);

			for (size_t igauss =0;igauss<PolDegree+1;igauss++){
				double r = params.gllcoords()[igauss];
				for (size_t jgauss =0;jgauss<PolDegree+1;jgauss++){
					double s = params.gllcoords()[jgauss];
                    size_t ii = igauss * (params.polynomialDegree()+1) + jgauss;
                    double x = params.mapX(iel, r); 
                    double y = params.mapY(jel, s);

					ValueType force_ref = params.fx(x, y);

					// CHECK(r[i] == Approx(result84[i]).margin(eps));

					CHECK(force_ref == Approx(Fe_element[ii]).margin(epsilon));
				}
			}
		}
	}
}

TEST_CASE("fku_cpu_vectortype"){
	std::cout<<"fKu test for vector type."<<std::endl;

	ValueType eps = 1e-6;

	size_t nx = 5;
    size_t ny = 5;
    size_t PolDegree = 2;
    // cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(nx,ny,PolDegree);

    
    cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(nx,ny,PolDegree);

	// std::vector<ValueType> mu_ref(params.ndofglobal(),0.0);
	// std::vector<ValueType> u_ref(params.ndofglobal(),1.0);
	size_t it = 10;
	
	// std::vector<ValueType> u_ref = {
	// 	1.31354022e-07,  1.90615488e-09, -3.17975347e-10, -1.92196003e-12,
	// 	4.65610089e-13,  1.58386672e-15, -3.88901125e-16, -7.57142697e-19,
	// 	1.87365378e-19,  2.06609785e-22, -1.02767412e-22,  1.90615488e-09,
	// 	2.94126703e-11, -5.04772078e-12, -2.68455721e-14,  6.53901153e-15,
	// 	1.84044612e-17, -4.53498069e-18, -7.10053611e-21,  1.76103041e-21,
	// 	1.51924977e-24, -7.56655347e-25, -3.17975347e-10, -5.04772078e-12,
	// 	8.75855988e-13,  4.50632897e-15, -1.09963156e-15, -2.88145732e-18,
	// 	7.11166975e-19,  9.88753112e-22, -2.45591830e-22, -1.72194261e-25,
	// 	8.58823378e-26, -1.92196003e-12, -2.68455721e-14,  4.50632897e-15,
	// 	1.75985968e-17, -4.32528601e-18, -7.98343780e-21,  1.97982524e-21,
	// 	1.72883020e-24, -4.30882995e-25, 0.0,  7.15976425e-29,
	// 	4.65610089e-13,  6.53901153e-15, -1.09963156e-15, -4.32528601e-18,
	// 	1.06292906e-18,  1.97448630e-21, -4.89633444e-22, -4.29999957e-25,
	// 	1.07168850e-25, 0.0, 0.0,  1.58386672e-15,
	// 	1.84044612e-17, -2.88145732e-18, -7.98343780e-21,  1.97448630e-21,
	// 	2.30294784e-24, -5.73529365e-25, 0.0,  5.96647026e-29,
	// 	0.0,  0.0, -3.88901125e-16, -4.53498069e-18,
	// 	7.11166975e-19,  1.97982524e-21, -4.89633444e-22, -5.73529365e-25,
	// 	1.42830443e-25,  5.96647028e-29, 0.0,  0.0,
	// 	0.0, -7.57142697e-19, -7.10053611e-21,  9.88753112e-22,
	// 	1.72883020e-24, -4.29999957e-25, 0.0,  5.96647028e-29,
	// 	0.0,  2.75045068e-38, 0.0,  0.0,
	// 	1.87365378e-19,  1.76103041e-21, -2.45591830e-22, -4.30882995e-25,
	// 	1.07168850e-25,  5.96647026e-29, 0.0,  2.75045068e-38,
	// 	0.0,  0.0, 0.0,  2.06609785e-22,
	// 	1.51924977e-24, -1.72194261e-25, 0.0,  3.57988217e-29,
	// 	0.0,  2.20036070e-38, 0.0,  0.0,
	// 	0.0, -1.02767412e-22, -7.56655347e-25,
	// 	8.58823378e-26,  7.15976425e-29, 0.0,  4.40072138e-38,
	// 	0.0,  0.0, 0.0,  0.0,
	// 	0.0
	// };
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
	//    std::vector<ValueType> fKu_ref={
	// 	1.13613516e-10,  9.70355940e-12, -8.58398582e-13, -1.96511099e-14,
	// 	2.31745974e-15,  2.92214387e-17, -3.53704120e-18, -2.46130158e-20,
	// 	3.02102541e-21,  1.19124749e-23, -1.47423916e-24,  9.70355940e-12,
	// 	9.73874539e-13, -9.05095328e-14, -1.75392081e-15,  2.09513330e-16,
	// 	2.05871206e-18, -2.51034389e-19, -1.34680869e-21,  1.66013625e-22,
	// 	5.01922394e-25, -6.22693205e-26, -8.58398582e-13, -9.05095328e-14,
	// 	8.49179991e-15,  1.63431256e-16, -1.95579696e-17, -1.84607564e-19,
	// 	2.25490078e-20,  1.12713140e-22, -1.39146005e-23, -3.75261595e-26,
	// 	4.66190884e-27, -1.96511099e-14, -1.75392081e-15,  1.63431256e-16,
	// 	2.28700492e-18, -2.77271177e-19, -1.83573895e-21,  2.25938751e-22,
	// 	7.57965351e-25, -9.40305611e-26, 0.0,  0.0,
	// 	2.31745974e-15,  2.09513330e-16, -1.95579696e-17, -2.77271177e-19,
	// 	3.36052656e-20,  2.24765691e-22, -2.76601614e-23, -9.35542252e-26,
	// 	1.16054125e-26,  0.0,  0.0,  2.92214387e-17,
	// 	2.05871206e-18, -1.84607564e-19, -1.83573895e-21,  2.24765691e-22,
	// 	1.00828512e-24, -1.24845104e-25, 0.0,  0.0,
	// 	0.0,  0.0, -3.53704120e-18, -2.51034389e-19,
	// 	2.25490078e-20,  2.25938751e-22, -2.76601614e-23, -1.24845104e-25,
	// 	1.54574744e-26,  0.0,  0.0,  0.0,
	// 	0.0, -2.46130158e-20, -1.34680869e-21,  1.12713140e-22,
	// 	7.57965351e-25, -9.35542252e-26, 0.0,  0.0,
	// 	0.0,  1.26740771e-41, 0.0,  3.02102541e-21,
	// 	1.66013625e-22, -1.39146005e-23, -9.40305611e-26,  1.16054125e-26,
	// 	0.0,  0.0,  0.0,  4.77317639e-34,
	// 	0.0,  1.19124749e-23,  5.01922394e-25, -3.75261595e-26,
	// 	0.0,  0.0,  0.0,  1.26740771e-41, 0.0,
	// 	0.0,  0.0, -1.47423916e-24, -6.22693205e-26,
	// 	4.66190884e-27,  0.0,  0.0,  0.0,
	// 	0.0,  0.0,  0.0,  0.0,
	// 	0.0
	//    };

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
	// std::vector<ValueType> mu_cuda(params.ndofglobal(),0.0);
	std::vector<ValueType> fKu_2test(params.ndofglobal(),0.0);

	// std::vector<ValueType> u_cu(params.ndofglobal(),1.0);
    reference::compute_fku_ref(it, params, u_ref, fKu_2test);

	for (size_t i = 0; i < params.ndofglobal(); ++i) {
		CHECK(fKu_2test[i] == Approx(fKu_ref[i]).margin(eps));
	}
}



TEST_CASE("Write_test") {
	// double eps = 1e-12;

	// int nsteps = 100;
	// int ndofglobal = 40;
	// ontheFly test(nsteps, ndofglobal);
	// Matrix vector_Result = test.hardcodedwriteonu();

	// //std::string name = "output4_lalo.csv";
	// std::string name = "C:\\Repo\\Cpp_builds\\spectral_cm\\results\\output6_lalo.csv";

	// cie::otFly::writeU_(vector_Result, name);

}



// TEST_CASE("Write simulation") {
// 	double eps = 1e-12;
// 	//initializing
// 	std::array<int, 2> nelements = { 5, 5 };
// 	std::array<double, 2> lengths = { 2.0, 2.0 }; //x, y dimension
// 	std::array<double, 2>center = { 1.0, 1.0 };

// 	int nsteps = 1000;
// 	size_t polynomialDegree = 2;

// 	double duration = 12;
// 	double radius = 0.4;
// 	double frequency = 1.0;
// 	double sigmaS = 0.06;
// 	double E = 1.0;
// 	double rho = 1.0;
// 	double domain = 1.0;
// 	double alpha = 1.0;
// 	double c = 1.0;

// 	cie::spectralkernel::Params params(nelements, lengths, center, E, rho, domain, alpha, c,
// 		duration, radius, frequency, sigmaS,
// 		nsteps, polynomialDegree);

// 	cie::otFly::ontheFly solver(params.nsteps(), params.ndofglobal());

// 	Matrix solution1 =solver.otF_Solve(params,true);
// 	std::string name = "C:\\Repo\\Cpp_builds\\spectral_cm\\results\\test2.csv";

// 	cie::otFly::writeU_(solution1, name);

// }


}//otFly
}//cie

//old tests using the main functions of algorithms
//these were re-done so that they could fit reference functions...

// TEST_CASE("Algorithm57Test") {
// 	// initialization
// 	double eps = 1e-12;

// 	double rho = 1.0;
// 	double dx = 1.0;
// 	double dy = 1.0;

// 	//G
// 	size_t polynomialDegree = 2;
// 	adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1);
// 	std::vector<double> G = kron(gauss.getWeights(), gauss.getWeights(),
// 		(rho * (dx / 2.0) * (dy / 2.0)));
// 	// B

// 	Matrix R1({ {0.22007998, 0.26355171},
// 		{0.77065457, 0.91751922},
// 		{0.05943222, 0.93462959} });

// 	Matrix R2({ {0.16564331, 0.59244317},
// 		{0.70164211, 0.32450572},
// 		{0.29389116, 0.22698156} });

// 	Matrix R3({ {0.1430465 , 0.41063075},
// 		{0.94203222, 0.11368537},
// 		{0.1389887 , 0.65897952} });

// 	std::vector<Matrix> R = { R1,R2,R3 };// B = cie::otFly::concatenate(S, R);

// 	std::vector<double> P = { 1., 1., 1., 1., 1., 1., 1., 1., 1. };

// 	std::vector<double> result = {2.2218384, 2.4567019, 2.58976008,
// 								1.20131775, 1.32830525,
// 								1.40024799, 1.24265777, 
// 								1.37401518, 1.44843364};
// 	std::vector<double> alg57 = alg_57(2, R, P);
// 	for (int i = 0; i < alg57.size(); ++i) {
// 		CHECK(alg57[i] == Approx(result[i]).margin(eps));
// 	}
// }





// TEST_CASE("Algorithm84Test") {
// 	// initialization
// 	double eps = 1e-12;

// 	double rho = 1.0;
// 	double dx = 1.0;
// 	double dy = 1.0;

// 	//G
// 	size_t polynomialDegree = 2;
// 	adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1);
// 	std::vector<double> M = kron(gauss.getWeights(), gauss.getWeights(),
// 		(rho * (dx / 2.0) * (dy / 2.0)));
// 	// B
// 	Matrix S = cie::spectralkernel::get_ShapesDiff(gauss.getCoordinates(), 0);
// 	Matrix R = cie::spectralkernel::get_ShapesDiff(gauss.getCoordinates(), 0);
// 	Matrix Sdiff = cie::spectralkernel::get_ShapesDiff(gauss.getCoordinates(), 1) * (2.0/dx);//* 2 / dx
// 	Matrix Rdiff = cie::spectralkernel::get_ShapesDiff(gauss.getCoordinates(), 1) * (2.0 / dy);// * 2 / dy
// 	//Matrix Sdiff = Sdiff_deta * (2.0 / dx);
// 	//Matrix Rdiff = Rdiff_deta * (2.0 / dy);

// 	std::vector<Matrix> B = cie::otFly::concatenate(S, R);
// 	std::vector<Matrix> Bdiff = cie::otFly::concatenate(Sdiff, Rdiff);

// 	std::vector<double> v = { 0.04234617, 0.9941601 , 0.39384291, 0.78736183, 0.86274709,
// 		0.25537733, 0.18095811, 0.12164062, 0.25661357 };

// 	std::vector<double> algorithm84 = alg_84(2, B,Bdiff, M, v);

// 	std::vector<double> result84 = { -0.72691823,  0.72956597, -0.23241883,  0.34839383,  2.29768615,
// 		-1.02363937, -0.24664702, -1.2099807 ,  0.06395819 };

// 	for (int i = 0; i < algorithm84.size(); ++i) {
// 		CHECK(algorithm84[i] == Approx(result84[i]).margin(eps));
// 	}

// }