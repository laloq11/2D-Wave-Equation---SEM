/*                  ___        __ __            ______                         *
 *                 /   |  ____/ // /_   ____   / ____/ __     __               *
 *                / /| | / __  // __ \ / __ \ / /   __/ /_ __/ /_              *
 *               / ___ |/ /_/ // / / // /_/ // /___/_  __//_  __/              *
 *              /_/  |_|\__,_//_/ /_/ \____/ \____/ /_/    /_/                 *
 *                                                                             *
 *       Copyright (C) Chair for Computation in Engineering, TU Muenchen       *
 *                           All Rights Reserved.                              *
 *     This file is part of the high order finite element framework AdhoC++.   */
#pragma once
#ifndef NUMERIC_GAUSSLOBATTO_H_
#define NUMERIC_GAUSSLOBATTO_H_

// --- Internal Includes ---
//#include "AbsIntegrationScheme.hpp"
#include <cmath>

// --- Standard Includes ---
#include <vector>

namespace adhocpp
{
namespace numeric
{

class GaussLobatto 
{
public:
  GaussLobatto( size_t order,
                double epsilon = 0 );

  ~GaussLobatto( ) ;

  std::vector<double> getCoordinates( ) ;

  std::vector<double> getWeights( ) ;

  size_t getNumberOfIntegrationPoints( ) ;

private:
  std::vector<double> myCoordinates;
  std::vector<double> myWeights;
  size_t myOrder;
  double myEpsilon;

  void initialize( );
};

} // namespace numeric
} // namespace adhocpp

#endif // NUMERIC_GAUSSLOBATTO_H_
