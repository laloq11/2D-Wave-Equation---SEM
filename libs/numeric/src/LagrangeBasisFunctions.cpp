/*                  ___        __ __            ______                         *
 *                 /   |  ____/ // /_   ____   / ____/ __     __               *
 *                / /| | / __  // __ \ / __ \ / /   __/ /_ __/ /_              *
 *               / ___ |/ /_/ // / / // /_/ // /___/_  __//_  __/              *
 *              /_/  |_|\__,_//_/ /_/ \____/ \____/ /_/    /_/                 *
 *                                                                             *
 *   AdhoC++, 2012 Chair for Computation in Engineering, All Rights Reserved.  */

// --- Internal Includes ---
#include "LagrangeBasisFunctions.hpp"

// --- Numeric Includes ---
//#include "numeric/inc/constants.hpp"

// --- Utilities Includes ---
//#include "utilities/inc/functionessentials.hpp"
//#include "utilities/inc/checks.hpp"
//#include "utilities/inc/logging.hpp"


namespace adhocpp
{
namespace numeric
{

LagrangeBasisFunctions::LagrangeBasisFunctions( std::vector<double> samplingPoints ) :
        mySamplingPoints( samplingPoints )
{

  //ADHOCPP_FUNCTION_START;

  // swap the last and the second entry because, by convention, the first
  // two entries are the nodal/blending terms

  double tmp = mySamplingPoints[mySamplingPoints.size( ) - 1];

  for ( size_t i = mySamplingPoints.size( ) - 1; i > 1; --i )
  {
    mySamplingPoints[i] = mySamplingPoints[i - 1];
  } // end of i-loop

  mySamplingPoints[1] = tmp;

  //Calculate polynomial's coefficients for later evaluation with Horner scheme
  calculateCoefficients( );

  //ADHOCPP_CHECK_NEAR_MSG_DBG( mySamplingPoints[0], -1.0, numeric::GEOPRECISION,
  //    "The first entry has to be at the left node!" );

  //ADHOCPP_CHECK_NEAR_MSG_DBG( mySamplingPoints[1], 1.0, numeric::GEOPRECISION,
  //    "The second entry has to be at the right node!" );

  //ADHOCPP_FUNCTION_END;

}

LagrangeBasisFunctions::~LagrangeBasisFunctions( )
{
}

void LagrangeBasisFunctions::calculateCoefficients( )
{

  //ADHOCPP_FUNCTION_START;

  std::vector<double> tmpCoefficients;

  size_t n = mySamplingPoints.size( ) - 1;

  //Determine temporal/help coefficients
  for ( size_t i = 0; i <= n; i++ )
  {
    tmpCoefficients.push_back( 0 );
  }
  tmpCoefficients[n] = -mySamplingPoints[0];

  for ( size_t i = 1; i <= n; i++ )
  {
    for ( size_t j = n - i; j <= n - 1; j++ )
    {
      tmpCoefficients[j] -= mySamplingPoints[i] * tmpCoefficients[j + 1];
    }

    tmpCoefficients[n] -= mySamplingPoints[i];
  }

  //Calculate the actual coefficients
  for ( size_t index = 0; index <= n; index++ )
  {
    std::vector<double> vec;
    myCoefficients.push_back( vec );
    for ( size_t i = 0; i <= n; i++ )
    {
      myCoefficients[index].push_back( 0.0 );
    }

    double phi, ff, b;

    for ( size_t j = 0; j <= n; j++ )
    {
      phi = n + 1;
      for ( size_t k = n; k >= 1; k-- )
      {
        phi = k * tmpCoefficients[k] + mySamplingPoints[j] * phi;
      }

      if ( j == index )
      {
        ff = 1.0 / phi;
      }
      else
      {
        ff = 0.0;
      }

      b = 1.0;
      for ( int k = n; k >= 0; k-- )
      {
        myCoefficients[index][k] += b * ff;
        b = tmpCoefficients[k] + mySamplingPoints[j] * b;
      }

    }

  }

  //ADHOCPP_FUNCTION_END;

}

double LagrangeBasisFunctions::evaluateSingle( const size_t& index,
                                               const size_t& ansatzOrder,
                                               const size_t& differentialOrder,
                                               const double& coordinate )
{

  // static size_t called=0;

  //ADHOCPP_FUNCTION_START;

  //ADHOCPP_CHECK_EQUAL_MSG_DBG( ansatzOrder, mySamplingPoints.size( ) - 1,
  //    "The number of sampling points does not fit to the passed polynomial order." );

  //ADHOCPP_CHECK_LESS_EQUAL_MSG_DBG( index, mySamplingPoints.size( ),
  //    "The passe index exceeds the number of sampling points." );

  //ADHOCPP_CHECK_LESS_EQUAL_MSG_DBG( differentialOrder, 2, "Derivatives are only implemented up to order 2" );

  double result = 0.0;

  if ( differentialOrder == 0 )
  {

    // Lagrange interpolation formula (slow but accurate)
    /*
     result = 1.0;

     for ( size_t j = 0; j < mySamplingPoints.size( ); ++j )
     {
     if ( j != index )
     {
     result *= ( coordinate - mySamplingPoints[j] ) / ( mySamplingPoints[index] - mySamplingPoints[j] );
     }
     }
     */

    //Coefficients and Horner scheme (fast but sensible to round off errors)
    result = myCoefficients[index][mySamplingPoints.size( ) - 1];
    for ( int j = mySamplingPoints.size( ) - 2; j >= 0; j-- )
    {
      result = result * coordinate + myCoefficients[index][j];
    }

  }
  else if ( differentialOrder == 1 )
  {

    //Lagrange interpolation formula (slow but accurate)
    /*
     result = 0;
     double product;

     for ( size_t i = 0; i < mySamplingPoints.size( ); ++i )
     {

     if ( i != index )
     {
     product = 1.0 / ( mySamplingPoints[index] - mySamplingPoints[i] );

     for ( size_t j = 0; j < mySamplingPoints.size( ); ++j )
     {
     if ( j != index && j != i )
     {
     product *= ( coordinate - mySamplingPoints[j] ) / ( mySamplingPoints[index] - mySamplingPoints[j] );
     }
     }
     result += product;
     }

     }
     */

    //Coefficients and Horner scheme (fast but sensible to round off errors)
    result = myCoefficients[index][mySamplingPoints.size( ) - 1] * ( mySamplingPoints.size( ) - 1 );
    for ( int j = mySamplingPoints.size( ) - 2; j >= 1; j-- )
    {
      result = result * coordinate + myCoefficients[index][j] * j;
    }
  }
  else if ( differentialOrder == 2 )
  {
    //Coefficients and Horner scheme (fast but sensible to round off errors)
    result = myCoefficients[index][mySamplingPoints.size( ) - 1] * ( mySamplingPoints.size( ) - 1 ) * ( mySamplingPoints.size( ) - 2 );
    for ( int j = mySamplingPoints.size( ) - 2; j >= 2; j-- )
    {
      result = result * coordinate + myCoefficients[index][j] * j * (j - 1);
    }
  }

  return result;

  //ADHOCPP_FUNCTION_END;
}

std::vector<double> LagrangeBasisFunctions::getMySamplingPoint( ) const
{
  //ADHOCPP_FUNCTION_START;

    return mySamplingPoints;

  //ADHOCPP_FUNCTION_END;
}



} // namespace numeric
} // namespace adhocpp


