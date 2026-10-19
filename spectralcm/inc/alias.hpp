#pragma once

#include <tuple>
#include <functional>
#include <array>
#include <vector>

#include "linalg.hpp"



namespace cie
{
	namespace otFly {

		class ontheFly;
		using SpatialFunction = std::function<double(double, double)>;
		using TimeFunction = std::function<double(double)>;

		using Matrix = linalg::Matrix;
		using LocationMap = std::vector<size_t>;
		using LocationMaps = std::vector<LocationMap>;
		using displacementVector = std::vector<std::vector<double> >;
		// class uniformMesh;
		// using uniformMesh = spectralkernel::uniformMesh;


	}//otFly
} // namespace cie

// namespace cie
// {
// 	namespace otFly_omp {

// 		class ontheFly;
// 		using SpatialFunction = std::function<double(double, double)>;
// 		using TimeFunction = std::function<double(double)>;

// 		using Matrix = linalg::Matrix;
// 		using LocationMap = std::vector<size_t>;
// 		using LocationMaps = std::vector<LocationMap>;
// 		using displacementVector = std::vector<std::vector<double> >;


// 	}//otFly_omp
// } // namespace cie


namespace cie
{
	namespace spectralkernel
	{
		class CompressedSparseRowMatrix;
		class Params;
		class uniformMesh;

		using ontheFly = otFly::ontheFly;

		using SpatialFunction = std::function<double(double, double)>;
		using TimeFunction = std::function<double(double)>;
		using IntegrationPoints = std::array<std::vector<double>, 2>;
		using IntegrationPointProvider = std::function<IntegrationPoints(size_t)>;

		using KnotVectors = std::array<std::vector<double>, 2>;

		using LocationMap = std::vector<size_t>;
		using LocationMaps = std::vector<LocationMap>;

		using Matrix = linalg::Matrix;

		using VectorOfMatrices = std::vector<Matrix>;
		using ElementLinearSystem = std::pair<Matrix, std::vector<double>>;

		using GlobalLinearSystem = std::pair<CompressedSparseRowMatrix, std::vector<double>>;

		using ShapeFunctions = std::pair<std::vector<double>, std::vector<double>>;

		using displacementVector = std::vector<std::vector<double> >;

	} // namespace splinekernel

}//namespace cie

namespace cie{
	namespace otFlyGPU {
		using ontheFly = cie::otFly::ontheFly;
		using Matrix = cie::linalg::Matrix;
		using Params = cie::spectralkernel::Params;//cie::spectralkernel::Params;
		
		template <typename ValueType>
		using Matrixt = std::vector<std::vector<ValueType>>;
}//ontheFlyGPU
}//cie

namespace adhocpp
{
	namespace numeric
	{
		class GaussLobatto;
		// class uniformMesh;
	}//numeric

}//adhocpp
