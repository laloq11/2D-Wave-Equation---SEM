#include "catch.hpp"

#include "alias.hpp"

#include "ontheFly.hpp" 
#include "LagrangeBasisFunctions.hpp"
#include "GaussLobatto.hpp" 
#include "basisfunctions.hpp"
#include "spectralcm.hpp"
#include <fstream>
#include <stdexcept>

namespace cie
{
	namespace otFly
	{


TEST_CASE("kroneckertest"){
	double eps = 1e-12;

	double rho = 1.0;
	double dx =  1.0;
	double dy =  1.0;

	std::vector<std::vector<double>> result_newaxis = { {1.,0.,	0.},
															{0.,	1.,	0.},
															{0.,0.,1.} };
	std::vector<double> result_kron = { 0.02777778, 0.11111111, 0.02777778, 
										0.11111111, 0.44444444,0.11111111, 
										0.02777778, 0.11111111, 0.02777778 };
	size_t polynomialDegree = 2;
	adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1);

	//Matrix shapesS_diff0(cie::spectralkernel::get_ShapesDiff(gauss.getCoordinates(), 0));
	//Matrix shapesS_diff1(cie::spectralkernel::get_ShapesDiff(gauss.getCoordinates(), 1));
	//Matrix shapesR_diff0_newaxis(); //pending because i dont understand the purpose.
	std::vector<double> G = kron(gauss.getWeights(), gauss.getWeights(),(rho * (dx/2.0) * (dy/2.0)) );

	for (size_t t = 0; t < G.size(); ++t) {
		CHECK(G[t] == Approx(result_kron[t]).margin(eps));

	}
}//kroneckertest

TEST_CASE("concatenateTest") {
	double eps = 1e-12;

	// double rho = 1.0;
	// double dx = 1.0;
	// double dy = 1.0;

	Matrix M1 ({ {1., 1.},
		{0., 0.},
		{0., 0.} });
	Matrix M2 ({ {0., 0.},
		{1., 1.},
		{0., 0.} });
	Matrix M3({ {0., 0.},
		{0., 0.},
		{1., 1.} });
	std::vector<Matrix> result_concatenate = { M1, M2, M3 };

	size_t polynomialDegree = 2;
	adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1);
	Matrix S = cie::spectralkernel::get_ShapesDiff(gauss.getCoordinates(), 0);
	Matrix R = cie::spectralkernel::get_ShapesDiff(gauss.getCoordinates(), 0);
	std::vector<Matrix> B = cie::otFly::concatenate(S, R);
	// std::cout<<"--------------------------------------------"<<std::endl;
	// std::cout<<"# matrices in vector: "<<B.size()<<std::endl;
	// std::cout<<"B rows :"<<B[0].size1()<<std::endl;
	// std::cout<<"B[0] cols :"<<B[0].size2()<<std::endl;
	// std::cout<<"--------------------------------------------"<<std::endl;
	size_t rows = result_concatenate[0].size1();
	size_t cols = result_concatenate[0].size2();
	for (size_t v = 0; v < result_concatenate.size(); ++v) {
		for (size_t c = 0; c < cols; ++c) {
			for (size_t r = 0; r < rows; ++r) {
				CHECK(B[v](r, c) == Approx(result_concatenate[v](r, c)).margin(eps));
			}
		}
	}
}//concatenateTest

TEST_CASE("CONCATENATETEST2") {
	Matrix M1({ {-3., -3.},{ 4.,  4. },{ -1., -1. } });
	Matrix M2({ {-1., -1.},{ 0.,  0.},{ 1.,  1.} });
	Matrix M3({ { 1.,  1.},{-4., -4.},{ 3.,  3.} });
	double eps = 1e-12;

	double rho = 1.0;
	double dx = 1.0;
	double dy = 1.0;


	std::vector<Matrix> result_concatenate = { M1, M2, M3 };

	size_t polynomialDegree = 2;
	adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1);
	Matrix S = cie::spectralkernel::get_ShapesDiff(gauss.getCoordinates(), 1) * (2.0 / dx);
	Matrix R = cie::spectralkernel::get_ShapesDiff(gauss.getCoordinates(), 1) * (2.0 / dy);
	std::vector<Matrix> B = cie::otFly::concatenate(S, R);

	size_t rows = result_concatenate[0].size1();
	size_t cols = result_concatenate[0].size2();
	for (size_t v = 0; v < result_concatenate.size(); ++v) {
		for (size_t c = 0; c < cols; ++c) {
			for (size_t r = 0; r < rows; ++r) {
				CHECK(B[v](r, c) == Approx(result_concatenate[v](r, c)).margin(eps));
			}
		}
	}
}//concatenateTest2




TEST_CASE("Algorithm57lightTest") {
	std::cout<<"entering new testing for alg57"<<std::endl;

	// initialization
	double eps = 1e-12;

	double rho = 1.0;
	double dx = 1.0;
	double dy = 1.0;

	//G
	size_t polynomialDegree = 2;
	adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1);
	std::vector<double> G = kron(gauss.getWeights(), gauss.getWeights(),
		(rho * (dx / 2.0) * (dy / 2.0)));
	// B

	Matrix R1({ {0.22007998, 0.26355171},
		{0.77065457, 0.91751922},
		{0.05943222, 0.93462959} });

	Matrix R2({ {0.16564331, 0.59244317},
		{0.70164211, 0.32450572},
		{0.29389116, 0.22698156} });

	Matrix R3({ {0.1430465 , 0.41063075},
		{0.94203222, 0.11368537},
		{0.1389887 , 0.65897952} });

	std::vector<Matrix> R = { R1,R2,R3 };// B = cie::otFly::concatenate(S, R);

	std::vector<double> P = { 1., 1., 1., 1., 1., 1., 1., 1., 1. };

	std::vector<double> result = {2.2218384, 2.4567019, 2.58976008,
								1.20131775, 1.32830525,
								1.40024799, 1.24265777, 
								1.37401518, 1.44843364};
	// std::vector<double> alg57 = alg_57(2, R, P);
	alg_57(2, R, P);
	for (int i = 0; i < P.size(); ++i) {
		CHECK(P[i] == Approx(result[i]).margin(eps));
	}

}

TEST_CASE("Algorithm84newTest") {
	std::cout<<"entering new testing for alg84"<<std::endl;
	// initialization
	double eps = 1e-12;

	double rho = 1.0;
	double dx = 1.0;
	double dy = 1.0;

	//G
	size_t polynomialDegree = 2;
	adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1);
	std::vector<double> M = kron(gauss.getWeights(), gauss.getWeights(),
		(rho * (dx / 2.0) * (dy / 2.0)));
	// B
	Matrix S = cie::spectralkernel::get_ShapesDiff(gauss.getCoordinates(), 0);
	Matrix R = cie::spectralkernel::get_ShapesDiff(gauss.getCoordinates(), 0);
	Matrix Sdiff = cie::spectralkernel::get_ShapesDiff(gauss.getCoordinates(), 1) * (2.0/dx);//* 2 / dx
	Matrix Rdiff = cie::spectralkernel::get_ShapesDiff(gauss.getCoordinates(), 1) * (2.0 / dy);// * 2 / dy
	//Matrix Sdiff = Sdiff_deta * (2.0 / dx);
	//Matrix Rdiff = Rdiff_deta * (2.0 / dy);

	std::vector<Matrix> B = cie::otFly::concatenate(S, R);
	std::vector<Matrix> Bdiff = cie::otFly::concatenate(Sdiff, Rdiff);

	std::vector<double> v = { 0.04234617, 0.9941601 , 0.39384291, 0.78736183, 0.86274709,
		0.25537733, 0.18095811, 0.12164062, 0.25661357 };

	std::vector<double> algorithm84 = alg_84(2, B,Bdiff, M, v);

	std::vector<double> result84 = { -0.72691823,  0.72956597, -0.23241883,  0.34839383,  2.29768615,
		-1.02363937, -0.24664702, -1.2099807 ,  0.06395819 };

	for (int i = 0; i < algorithm84.size(); ++i) {
		CHECK(algorithm84[i] == Approx(result84[i]).margin(eps));
	}
}


TEST_CASE("Algorithm82newTest") {
	// initialization
	double eps = 1e-12;

	double rho = 1.0;
	double dx = 1.0;
	double dy = 1.0;

	//G
	size_t polynomialDegree = 2;
	adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1);
	std::vector<double> G = kron(gauss.getWeights(), gauss.getWeights(),
										(rho * (dx / 2.0) * (dy / 2.0)));
	// B
	Matrix S = cie::spectralkernel::get_ShapesDiff(gauss.getCoordinates(), 0);
	Matrix R = cie::spectralkernel::get_ShapesDiff(gauss.getCoordinates(), 0);

	std::vector<Matrix> B = cie::otFly::concatenate(S, R);
	std::vector<double> P = { 1., 1., 1., 1., 1., 1., 1., 1., 1. };

	// std::vector<double> algorithm82 = alg_82(2,B,G,P);
	alg_82(2,B,G,P);

	std::vector<double> result82 = { 0.02777778, 0.11111111, 0.02777778, 0.11111111, 0.44444444,
		0.11111111, 0.02777778, 0.11111111, 0.02777778 };

	for (int i = 0; i < P.size(); ++i) {
		CHECK(P[i] == Approx(P[i]).margin(eps));
	}

}
//add it here

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



TEST_CASE("Write simulation") {
	// double eps = 1e-12;
	// //initializing
	// std::array<size_t, 2> nelements = { 5, 5 };
	// std::array<double, 2> lengths = { 2.0, 2.0 }; //x, y dimension
	// std::array<double, 2>center = { 1.0, 1.0 };

	// int nsteps = 1000;
	// size_t polynomialDegree = 2;

	// double duration = 12;
	// double radius = 0.4;
	// double frequency = 1.0;
	// double sigmaS = 0.06;
	// double E = 1.0;
	// double rho = 1.0;
	// double domain = 1.0;
	// double alpha = 1.0;
	// double c = 1.0;

	// cie::spectralkernel::Params params(nelements, lengths, center, E, rho, domain, alpha, c,
	// 	duration, radius, frequency, sigmaS,
	// 	nsteps, polynomialDegree);

	// cie::otFly::ontheFly solver(params.nsteps(), params.ndofglobal());

	// Matrix solution1 =solver.otF_Solve(params,true);
	// std::string name = "test2.csv";

	// cie::otFly::writeU_(solution1, name);

}

TEST_CASE("sliceTest") {
	// slice must return u_(it, locationMap[k]) and throw for rows u_ does not hold.
	cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(3, 3, 2);
	size_t ndof = params.ndofglobal();
	size_t nsteps = params.nsteps();

	ontheFly solver(nsteps, ndof);
	Matrix u = solver.otF_Solve(params, false);

	bool nonzero = false;
	for (size_t it : {size_t(0), size_t(1), nsteps / 2, nsteps}) {
		std::vector<double> row(ndof);
		for (size_t k = 0; k < ndof; ++k) row[k] = u(it, k);

		for (size_t i = 0; i < params.nelements(0); ++i) {
			for (size_t j = 0; j < params.nelements(1); ++j) {
				std::vector<double> locationMap = params.createLocationMap(i, j);
				std::vector<double> u_m = solver.slice(it, locationMap);
				std::vector<double> u_m_free = slice(row, locationMap);
				REQUIRE(u_m.size() == locationMap.size());
				REQUIRE(u_m_free.size() == locationMap.size());
				for (size_t k = 0; k < locationMap.size(); ++k) {
					double ref = u(it, static_cast<size_t>(locationMap[k]));
					if (ref != 0.0) nonzero = true;
					CHECK(u_m[k] == ref);
					CHECK(u_m_free[k] == ref);
				}
			}
		}
	}
	// guards against the comparisons passing on an all-zero solution
	CHECK(nonzero);

	std::vector<double> locationMap = params.createLocationMap(0, 0);
	CHECK_THROWS_AS(solver.slice(nsteps + 1, locationMap), std::out_of_range);

	// single-row storage mode: row 0 exists, row 1 must throw instead of falling back to row 0
	ontheFly storage(nsteps, ndof, "strg_1");
	for (double value : storage.slice(0, locationMap)) CHECK(value == 1.0);
	CHECK_THROWS_AS(storage.slice(1, locationMap), std::out_of_range);
}


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

// TEST_CASE("swapaxesTest") {
// 	double eps = 1e-12;

// 	Matrix R1({ {1,2}  ,{3,4}  ,{5,6} });
// 	Matrix R2({ {7,8}  ,{9,10} ,{11,12} });
// 	Matrix R3({ {13,14},{15,16},{17,18} });

// 	std::vector<Matrix> R = { R1,R2,R3 };

// 	std::vector<Matrix> Rswap = swapaxes(R);

// 	Matrix R1n({ {1,2}  ,{7,8}  ,{13,14} });
// 	Matrix R2n({ {3,4}  ,{9,10} ,{15,16} });
// 	Matrix R3n({ {5,6},{11,12},{17,18} });
// 	std::vector<Matrix> Rresult = { R1n,R2n,R3n };

// 	for (size_t t = 0; t < Rresult.size(); ++t) {
// 		for (size_t i = 0; i < Rresult[0].size1(); ++i) {
// 			for (size_t j = 0; j < Rresult[0].size2(); ++j) {
// 				CHECK(Rswap[t](i, j) == Approx(Rresult[t](i,j)).margin(eps));

// 			}
// 		}
// 	}
// }

// TEST_CASE("Algorithm82Test") {
// 	// initialization
// 	double eps = 1e-12;

// 	double rho = 1.0;
// 	double dx = 1.0;
// 	double dy = 1.0;

// 	//G
// 	size_t polynomialDegree = 2;
// 	adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1);
// 	std::vector<double> G = kron(gauss.getWeights(), gauss.getWeights(),
// 										(rho * (dx / 2.0) * (dy / 2.0)));
// 	// B
// 	Matrix S = cie::spectralkernel::get_ShapesDiff(gauss.getCoordinates(), 0);
// 	Matrix R = cie::spectralkernel::get_ShapesDiff(gauss.getCoordinates(), 0);

// 	std::vector<Matrix> B = cie::otFly::concatenate(S, R);
// 	std::vector<double> P = { 1., 1., 1., 1., 1., 1., 1., 1., 1. };

// 	std::vector<double> algorithm82 = alg_82(2,B,G,P);

// 	std::vector<double> result82 = { 0.02777778, 0.11111111, 0.02777778, 0.11111111, 0.44444444,
// 		0.11111111, 0.02777778, 0.11111111, 0.02777778 };

// 	for (int i = 0; i < algorithm82.size(); ++i) {
// 		CHECK(algorithm82[i] == Approx(result82[i]).margin(eps));
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