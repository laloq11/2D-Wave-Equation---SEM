/*                  ___        __ __            ______                         *
 *                 /   |  ____/ // /_   ____   / ____/ __     __               *
 *                / /| | / __  // __ \ / __ \ / /   __/ /_ __/ /_              *
 *               / ___ |/ /_/ // / / // /_/ // /___/_  __//_  __/              *
 *              /_/  |_|\__,_//_/ /_/ \____/ \____/ /_/    /_/                 *
 *                                                                             *
 *   AdhoC++, 2012 Chair for Computation in Engineering, All Rights Reserved.  */
#pragma once

#ifdef _INCLUDE_CSTDDEF
#include <cstddef>
#endif

#ifndef numeric_LAGRANGEBASISFUNCTIONS_HPP_
#define numeric_LAGRANGEBASISFUNCTIONS_HPP_

// --- Internal Includes ---
//#include "AbsPolynomialBasisFunctions.hpp"
#include <vector>

namespace adhocpp
{
namespace numeric
{

class LagrangeBasisFunctions //final : public AbsPolynomialBasisFunctions
{
public:
  LagrangeBasisFunctions( std::vector<double> samplingPoints );
  ~LagrangeBasisFunctions( ) ;

  double evaluateSingle( const size_t& index,
                         const size_t& ansatzOrder,
                         const size_t& differentialOrder,
                         const double& coordinate ) ;

  std::vector<double> getMySamplingPoint() const;

private:
  explicit LagrangeBasisFunctions( const LagrangeBasisFunctions& );
  LagrangeBasisFunctions& operator=( const LagrangeBasisFunctions& );

  void calculateCoefficients();

  std::vector<double> mySamplingPoints;

  std::vector< std::vector<double> > myCoefficients;
};

} // namespace numeric 
} // namespace adhocpp
#endif // numeric_LAGRANGEBASISFUNCTIONS_HPP_ 
