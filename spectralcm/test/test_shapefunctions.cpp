#include "catch.hpp"

#include <vector>
#include "alias.hpp"
#include "GaussLobatto.hpp" 
#include "LagrangeBasisFunctions.hpp" 
#include "params.hpp" 

namespace cie
{
	namespace spectralkernel
	{

TEST_CASE("GLLcoordinates")
{
	std::vector<double> coords2{ -1.0, 0.0, 1.0 };
	std::vector<double> weights2{ 0.33333333, 1.33333333, 0.33333333 };

	std::vector<double> coords3{ -1.       , -0.4472136,  0.4472136,  1. };
	std::vector<double> weights3{ 0.16666667, 0.83333333, 0.83333333, 0.16666667 };

	std::vector<double> coords4{ -1.        , -0.65465367,  0.        ,  0.65465367,  1. };
	std::vector<double> weights4{ 0.1       , 0.54444444, 0.71111111, 0.54444444, 0.1 };
	//adhocpp::numeric::GaussLobatto

	//CHECK_THROWS( adhocpp::numeric::GaussLobatto(polynomialDegree + 1) );
	// //this is to check if throws are working

	// First basis function
	size_t polynomialDegree = 2;
	adhocpp::numeric::GaussLobatto gauss(polynomialDegree + 1);
	for (size_t i = 0; i < coords2.size(); ++i) {
		CHECK(gauss.getCoordinates()[i] == Approx(coords2[i]));
		REQUIRE(gauss.getWeights()[i] == Approx(weights2[i]));
	}
	polynomialDegree = 3;
	adhocpp::numeric::GaussLobatto gauss3(polynomialDegree + 1);
	for (size_t i = 0; i < coords3.size(); ++i) {
		CHECK(gauss3.getCoordinates()[i] == Approx(coords3[i]));
		REQUIRE(gauss3.getWeights()[i] == Approx(weights3[i]));
	}
	polynomialDegree = 4;
	adhocpp::numeric::GaussLobatto gauss4(polynomialDegree + 1);
	for (size_t i = 0; i < coords4.size(); ++i) {
		CHECK(gauss4.getCoordinates()[i] == Approx(coords4[i]));
		REQUIRE(gauss4.getWeights()[i] == Approx(weights4[i]));
	}
}

TEST_CASE("ShapeFunctions_Derivative_Order0")
{
	double eps = 1e-12;
	size_t polynomialDegree = 2;
	adhocpp::numeric::GaussLobatto gll(polynomialDegree + 1);
	std::vector<double> coords = gll.getCoordinates();
	adhocpp::numeric::LagrangeBasisFunctions shapes0(coords);

	Matrix result({ { 1,0,0}, {0,1,0},{0,0,1} });
	for (size_t i = 0; i < coords.size(); ++i) {
		double shape_p0 = shapes0.evaluateSingle(0, 1, 0, coords[i]);
		double shape_p1 = shapes0.evaluateSingle(1, 1, 0, coords[i]);
		double shape_p2 = shapes0.evaluateSingle(2, 1, 0, coords[i]);
		//std::cout << "Point 0: " << coords[0] << " evaluated in " << i << " = " << shape_p0 << std::endl;
		//std::cout << "Point 1: " << coords[1] << " evaluated in " << i << " = " << shape_p1 << std::endl;
		//std::cout << "Point 2: " << coords[2] <<" evaluated in " << i << " = " << shape_p2<<std::endl;
		CHECK(shape_p0 == Approx(result(0, i)).margin(eps));
		CHECK(shape_p1 == Approx(result(2, i)).margin(eps));
		CHECK(shape_p2 == Approx(result(1, i)).margin(eps));
		//REQUIRE(gauss.getWeights()[i] == Approx(weights[i]));
	}

	polynomialDegree = 3;
	adhocpp::numeric::GaussLobatto gll3(polynomialDegree + 1);
	std::vector<double> coords3 = gll3.getCoordinates();
	adhocpp::numeric::LagrangeBasisFunctions shapes0_3(coords3);

	//Matrix result3({ { 1.0,0.0,0.0,0.0}, {0.0,0.0,0.0,1.0},{0.0,1.0,0.0,0.0}, {0.0,0.0,1.0,0.0} });
	Matrix result3({	{1.0,0.0,0.0,0.0},  
						{0.0,0.0,1.0,0.0},
						{0.0,0.0,0.0,1.0}, 
						{0.0,1.0,0.0,0.0}, });
	std::vector<double> lagrange_SamplingPoint = shapes0_3.getMySamplingPoint();
	for (size_t i = 0; i < coords3.size(); ++i) {
		double shape_p0 = shapes0_3.evaluateSingle(0, 0, 0, coords3[i]);
		double shape_p1 = shapes0_3.evaluateSingle(1, 0, 0, coords3[i]);
		double shape_p2 = shapes0_3.evaluateSingle(2, 0, 0, coords3[i]);
		double shape_p3 = shapes0_3.evaluateSingle(3, 0, 0, coords3[i]);

		CHECK(shape_p0 == Approx(result3(i,0)).margin(eps));
		CHECK(shape_p1 == Approx(result3(i,1)).margin(eps));
		CHECK(shape_p2 == Approx(result3(i,2)).margin(eps));
		CHECK(shape_p3 == Approx(result3(i,3)).margin(eps));
		//REQUIRE(gauss.getWeights()[i] == Approx(weights[i]));
	}

	polynomialDegree = 4;
	adhocpp::numeric::GaussLobatto gll4(polynomialDegree + 1);
	std::vector<double> coords4 = gll4.getCoordinates();
	adhocpp::numeric::LagrangeBasisFunctions shapes0_4(coords4);
	std::vector<double> lagrange_SamplingPoint4 = shapes0_4.getMySamplingPoint();

	Matrix result4({ { 1.0,0.0,0.0,0.0,0.0},
		{0.0,0.0,0.0,0.0,1.0},
		{0.0,1.0,0.0,0.0,0.0},
		{0.0,0.0,1.0,0.0,0.0}, 
		{0.0,0.0,0.0,1.0,0.0} });

	for (size_t i = 0; i < coords3.size(); ++i) {
		double shape_p0 = shapes0_4.evaluateSingle(i, 1, 0, coords4[0]);
		double shape_p1 = shapes0_4.evaluateSingle(i, 1, 0, coords4[1]);
		double shape_p2 = shapes0_4.evaluateSingle(i, 1, 0, coords4[2]);
		double shape_p3 = shapes0_4.evaluateSingle(i, 1, 0, coords4[3]);
		double shape_p4 = shapes0_4.evaluateSingle(i, 1, 0, coords4[4]);
		//std::cout << "Point 0: " << coords3[0] << " evaluated in " << i << " = " << shape_p0 << std::endl;
		//std::cout << "Point 1: " << coords3[1] << " evaluated in " << i << " = " << shape_p1 << std::endl;
		//std::cout << "Point 2: " << coords3[2] << " evaluated in " << i << " = " << shape_p2 << std::endl;
		//std::cout << "Point 3: " << coords3[3] << " evaluated in " << i << " = " << shape_p3 << std::endl;
		//std::cout << "----" << std::endl;
		CHECK(shape_p0 == Approx(result4(i, 0)).margin(eps));
		CHECK(shape_p1 == Approx(result4(i, 1)).margin(eps));
		CHECK(shape_p2 == Approx(result4(i, 2)).margin(eps));
		CHECK(shape_p3 == Approx(result4(i, 3)).margin(eps));
		CHECK(shape_p4 == Approx(result4(i, 4)).margin(eps));
		//REQUIRE(gauss.getWeights()[i] == Approx(weights[i]));
	}


}

TEST_CASE("ShapeFunctions_Derivative_OrderOne")
{
	double eps = 1e-12;
	size_t polynomialDegree = 2;
	adhocpp::numeric::GaussLobatto gll(polynomialDegree + 1);
	std::vector<double> coords = gll.getCoordinates();
	adhocpp::numeric::LagrangeBasisFunctions shapes0(coords);


	Matrix result({ { -1.5,  2. , -0.5 }, { -0.5,  0. ,  0.5 }, { 0.5, -2. ,  1.5 } });

	for (size_t i = 0; i < coords.size(); ++i) {
		double shape_p0 = shapes0.evaluateSingle(0, 0, 1, coords[i]);
		double shape_p2 = shapes0.evaluateSingle(1, 0, 1, coords[i]);
		double shape_p1 = shapes0.evaluateSingle(2, 0, 1, coords[i]);

		CHECK(shape_p0 == Approx(result(i,0)).margin(eps));
		CHECK(shape_p1 == Approx(result(i,1)).margin(eps));
		CHECK(shape_p2 == Approx(result(i,2)).margin(eps));
		//REQUIRE(gauss.getWeights()[i] == Approx(weights[i]));
	}

	polynomialDegree = 3;
	adhocpp::numeric::GaussLobatto gll3(polynomialDegree + 1);
	std::vector<double> coords3 = gll3.getCoordinates();
	adhocpp::numeric::LagrangeBasisFunctions shapes0_3(coords3);


	Matrix result3({	{-3.        ,  4.04508497, -1.54508497,  0.5       }, 
						{-0.80901699,  0.        ,  1.11803399, -0.30901699},
						{ 0.30901699, -1.11803399,  0.        ,  0.80901699},
						{-0.5       ,  1.54508497, -4.04508497,  3.        }});

	std::vector<double> lagrange_SamplingPoint = shapes0_3.getMySamplingPoint();
	for (size_t i = 0; i < coords3.size(); ++i) {
		double shape_p0 = shapes0_3.evaluateSingle(0, 0, 1, coords3[i]);
		double shape_p1 = shapes0_3.evaluateSingle(2, 0, 1, coords3[i]);
		double shape_p2 = shapes0_3.evaluateSingle(3, 0, 1, coords3[i]);
		double shape_p3 = shapes0_3.evaluateSingle(1, 0, 1, coords3[i]);

		CHECK(shape_p0 == Approx(result3(i, 0)).margin(eps));
		CHECK(shape_p1 == Approx(result3(i, 1)).margin(eps));
		CHECK(shape_p2 == Approx(result3(i, 2)).margin(eps));
		CHECK(shape_p3 == Approx(result3(i, 3)).margin(eps));
		//REQUIRE(gauss.getWeights()[i] == Approx(weights[i]));
	}

	polynomialDegree = 4;
	adhocpp::numeric::GaussLobatto gll4(polynomialDegree + 1);
	std::vector<double> coords4 = gll4.getCoordinates();
	adhocpp::numeric::LagrangeBasisFunctions shapes0_4(coords4);
	std::vector<double> lagrange_SamplingPoint4 = shapes0_4.getMySamplingPoint();

	Matrix result4({{-5.        ,  6.75650249, -2.66666667,  1.41016418, -0.5       }, 
					{-1.24099025e+00,  8.88178420e-16,  1.74574312e+00, -7.63762616e-01,2.59009747e-01}, 
					{ 3.75000000e-01, -1.33658458e+00, -2.22044605e-16,  1.33658458e+00,-3.75000000e-01}, 
					{-2.59009747e-01,  7.63762616e-01, -1.74574312e+00, -1.33226763e-15,1.24099025e+00}, 
					{ 0.5       , -1.41016418,  2.66666667, -6.75650249,  5.        } });

	for (size_t i = 0; i < coords4.size(); ++i) {
		double shape_p0 = shapes0_4.evaluateSingle(0, 0, 1, coords4[i]);
		double shape_p1 = shapes0_4.evaluateSingle(2, 0, 1, coords4[i]);
		double shape_p2 = shapes0_4.evaluateSingle(3, 0, 1, coords4[i]);
		double shape_p3 = shapes0_4.evaluateSingle(4, 0, 1, coords4[i]);
		double shape_p4 = shapes0_4.evaluateSingle(1, 0, 1, coords4[i]);

		//std::cout << shape_p0 << "," << shape_p1 << "," << shape_p2 << "," << shape_p3 << "," << shape_p4<< std::endl;

		CHECK(shape_p0 == Approx(result4(i, 0)).margin(eps));
		CHECK(shape_p1 == Approx(result4(i, 1)).margin(eps));
		CHECK(shape_p2 == Approx(result4(i, 2)).margin(eps));
		CHECK(shape_p3 == Approx(result4(i, 3)).margin(eps));
		CHECK(shape_p4 == Approx(result4(i, 4)).margin(eps));
		//REQUIRE(gauss.getWeights()[i] == Approx(weights[i]));
	}


}//ShapeFunctions_Derivative_OrderOne

TEST_CASE("GetShapesFunction_test")
{
	/*
	We need to have 
	N = [N1,N2,N3] 
	to be evaluated by 
	P = [P1,P2,P3]
	so a correct order from the shape functions is needed.
	*/

	double eps = 1e-12;
	size_t polynomialDegree = 2;

	adhocpp::numeric::GaussLobatto gll(polynomialDegree + 1);
	std::vector<double> coords = gll.getCoordinates();
	//adhocpp::numeric::LagrangeBasisFunctions shapes0(coords);

	Matrix shapes0(get_ShapesDiff(coords,0));

	Matrix result({ { 1,0,0}, {0,1,0},{0,0,1} });
	for (size_t i = 0; i < coords.size(); ++i) {
		for (size_t s = 0; s < coords.size(); ++s) {
			CHECK(shapes0(i,s) == Approx(result(i,s)).margin(eps));
		}
		//REQUIRE(gauss.getWeights()[i] == Approx(weights[i]));
	}

	polynomialDegree = 3;
	adhocpp::numeric::GaussLobatto gll3(polynomialDegree + 1);
	std::vector<double> coords3 = gll3.getCoordinates();
	Matrix shapes3(get_ShapesDiff(coords3,0));

	//Matrix result3({ { 1.0,0.0,0.0,0.0}, {0.0,0.0,0.0,1.0},{0.0,1.0,0.0,0.0}, {0.0,0.0,1.0,0.0} });
	Matrix result3({{1.0,0.0,0.0,0.0},
					{0.0,1.0,0.0,0.0},
					{0.0,0.0,1.0,0.0},
					{0.0,0.0,0.0,1.0}});

	for (size_t i = 0; i < coords3.size(); ++i) {
		for (size_t s = 0; s < coords3.size(); ++s) {
			CHECK(shapes3(i, s) == Approx(result3(i, s)).margin(eps));
		}
		//REQUIRE(gauss.getWeights()[i] == Approx(weights[i]));
	}
	// 
	


	polynomialDegree = 4;
	adhocpp::numeric::GaussLobatto gll4(polynomialDegree + 1);
	std::vector<double> coords4 = gll4.getCoordinates();
	Matrix shapes4(get_ShapesDiff(coords4,0));

	Matrix result4({ { 1.0,0.0,0.0,0.0,0.0},
				{0.0,1.0,0.0,0.0,0.0},
				{0.0,0.0,1.0,0.0,0.0},
				{0.0,0.0,0.0,1.0,0.0},
				{0.0,0.0,0.0,0.0,1.0} });

	for (size_t i = 0; i < coords4.size(); ++i) {
		for (size_t s = 0; s < coords4.size(); ++s) {
			CHECK(shapes4(i, s) == Approx(result4(i, s)).margin(eps));
		}
	}

}


	}//spectralkernel
}//cie