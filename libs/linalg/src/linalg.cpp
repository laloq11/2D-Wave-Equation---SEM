

#include "linalg.hpp"

namespace cie
{
namespace linalg
{
namespace linalghelper
{

void runtime_check( bool result, const char message[] )
{
    if( !result )
    {
        throw std::runtime_error{ message };
    }
}

} // linalghelper

Matrix::Matrix( size_t size1, size_t size2, double value ) :
    size1_( size1 ), size2_( size2 ), data_( size1 * size2, value )
{
}

Matrix::Matrix(size_t size1, size_t size2) :
	Matrix(size1, size2, 0.0)
{
}

Matrix::Matrix() :
	Matrix(0, 0)
{
}

Matrix::Matrix( const std::vector<double>& rowMajorData, size_t size1 ) :
    size1_( size1 ), size2_( 0 ), data_( rowMajorData )
{
    if( size1 != 0 )
    {
        size2_ = rowMajorData.size( ) / size1;
    }
    else
    {
        linalghelper::runtime_check( rowMajorData.size( ) == 0, "Zero size with non-zero sized data." );
    }

    linalghelper::runtime_check( size1_ * size2_ == rowMajorData.size( ), "Inconsistent sizes!" );
}

Matrix::Matrix( const std::vector<Vector>& vectorOfRows ) :
    size1_( vectorOfRows.size( ) ), size2_( 0 )
{
    if( size1_ != 0 )
    {
        size2_ = vectorOfRows.front( ).size( );
        data_.resize( size1_ * size2_ );

        for( size_t i = 0; i < size1_; ++i )
        {
            linalghelper::runtime_check( vectorOfRows[i].size( ) == size2_,
                                         "Inconsistent input data!" );

            std::copy( vectorOfRows[i].begin( ), vectorOfRows[i].end( ), data_.begin( ) + ( i * size2_ ) );
        }
    }
}
Matrix::Matrix (const Matrix& B): size1_( B.size1( ) ), size2_( B.size2() ) {
    // std::vector<double> data_;
    for (size_t i=0;i<B.size1();i++){
        for (size_t j=0;j<B.size2();j++){
            // std::copy( B.begin( ), vectorOfRows[i].end( ), data_.begin( ) + ( i * size2_ ) );
            data_.push_back(B(i,j));
        }
    }
}



Vector Matrix::operator*(const std::vector<double>& vector) {
    //size_t globalSize = indptr_.size() - 1;
    /*
        std::transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(), std::plus<int>());

    */
    linalghelper::runtime_check(vector.size() == size2_, "Invalid rhs size.");

    std::vector<double> result(vector.size(), 0.0);

    for (size_t iRow = 0; iRow < size1_; ++iRow)
    {
        for (size_t jCol = 0; jCol < size2_; ++jCol)
        {
            result[iRow] += data_[iRow*size2_+jCol] * vector[jCol];
        }
    }
    return result;
}
Matrix Matrix::operator*(const double& factor)  {
    Matrix Result(size1_, size2_);
    for (size_t iRow = 0; iRow < size1_; ++iRow)
    {
        for (size_t jCol = 0; jCol < size2_; ++jCol)
        {
            Result(iRow,jCol) = data_[iRow * size2_ + jCol] * factor;
        }
    }
    return Result;
}
Matrix Matrix::operator+(const Matrix& M2) {
    Matrix Result(size1_, size2_);
    for (size_t iRow = 0; iRow < size1_; ++iRow)
    {
        for (size_t jCol = 0; jCol < size2_; ++jCol)
        {
            Result(iRow, jCol) = data_[iRow * size2_ + jCol] + M2(iRow,jCol);
        }
    }
    return Result;
}

namespace linalghelper
{

// template<typename VectorType>
// void writeRow( const VectorType& vector, size_t size, std::ostream& out, size_t digits )
// {
//     auto precision = out.precision( );

//     out << std::setprecision( digits - 4 );

//     for( size_t i = 0; i < size; ++i )
//     {
//         out << std::setw( digits ) << vector( i );
//         if (i < size - 1) {
//             out<< ", ";
//         }
//     }

//     out << std::endl << std::setprecision( precision );
// }

} // linalghelper

void write( const Vector& vector, std::ostream& out )
{
    linalghelper::writeRow( [&]( size_t i ){ return vector[i]; }, vector.size( ), out, 12 );
}

void write( const Matrix& matrix, std::ostream& out )
{
    size_t size1 = matrix.size1( );
    size_t size2 = matrix.size2( );

    for( size_t i = 0; i < size1; ++i )
    {
        linalghelper::writeRow( [&]( size_t j ){ return matrix( i, j ); }, size2, out, 12 );
    }
}

double norm( const Vector& vector )
{
    return std::sqrt( std::inner_product( std::begin( vector ),
                                          std::end( vector ),
                                          std::begin( vector ), 0.0 ) );
}

double norm( const Matrix& matrix )
{
    double result = 0.0;

    for( size_t i = 0; i < matrix.size1( ); ++i )
    {
        for( size_t j = 0; j < matrix.size2( ); ++j )
        {
            result += matrix( i, j ) * matrix( i, j );
        }
    }

    return std::sqrt( result );
}


namespace linalghelper
{

using PermutationVector = std::vector<size_t>;

void updatePermutation( const Matrix& matrix,
                        PermutationVector& permutation,
                        size_t index,
                        double singularTolerance )
{
    auto compare = [&]( size_t i1, size_t i2 )
    {
        return std::abs( matrix( permutation[i1], index ) ) <
               std::abs( matrix( permutation[i2], index ) );
    };

    // Search for max element in column starting at (index, index)
    auto pivot = std::max_element( permutation.begin( ) + index, permutation.end( ), compare );

    linalghelper::runtime_check( std::abs( matrix( *pivot, index ) ) > singularTolerance,
                                 "Matrix is singular!" );

    // Swap indices in permutation vector
    std::iter_swap( permutation.begin( ) + index, pivot );
}

} // linalghelper

Vector solve( const Matrix& matrix,
              const Vector& rhs )
{
    size_t n = matrix.size1( );

    if( n == 0 )
    {
        return { };
    }

    linalghelper::runtime_check( matrix.size2( ) == n, "Solve needs a square matrix!" );
    linalghelper::runtime_check( rhs.size( ) == n, "Matrix and vector sizes don't match!" );

    // Instead of swapping rows in our matrix we use a permutation vector
    linalghelper::PermutationVector permute( n );

    std::iota( permute.begin( ), permute.end( ), 0 ); // initialize  with 1 ... n

    Vector x( n ), tmpB = rhs;
    Matrix tmpM = matrix;

    // Create two convenience lambda functions to access and permute tmpM and tmpP
    auto permuteMatrix = [&]( size_t i, size_t j ) -> double& { return tmpM( permute[i], j ); };
    auto permuteVector = [&]( size_t i ) -> double& { return tmpB[permute[i]]; };

    double tolerance = 1e-10 * norm( matrix );

    // LU decomposition with partial pivoting
    for( size_t k = 0; k < n - 1; k++ )
    {
        linalghelper::updatePermutation( tmpM, permute, k, tolerance );

        for( size_t i = k + 1; i < n; i++ )
        {
            permuteMatrix( i, k ) /= permuteMatrix( k, k );

            for( size_t j = k + 1; j < n; j++)
            {
                permuteMatrix( i, j ) -= permuteMatrix( i, k ) * permuteMatrix( k, j );
            }
        }
    }

    // Just for the singularity check
    linalghelper::updatePermutation( tmpM, permute, n - 1, tolerance );

    // forward substitution
    for( size_t i = 1; i < n; i++ )
    {
        for( size_t j = 0; j < i; j++ )
        {
            permuteVector( i ) -= permuteMatrix( i, j ) * permuteVector( j );
        }
    }

    // backward substitution
    for( size_t i = n; i > 0; i-- )
    {
        for( size_t j = i; j < n; j++ )
        {
            permuteVector( i - 1 ) -= permuteMatrix( i - 1, j ) * permuteVector( j );
        }

        permuteVector( i - 1 ) /= permuteMatrix( i - 1, i - 1 );
    }

    // permute solution and write into x
    std::transform( permute.begin( ), permute.end( ), x.begin( ), [&]( size_t i ){ return tmpB[i]; } );

    return x;
}

void Matrix::fill( double value )
{
    std::fill( data_.begin( ), data_.end( ), value );
}

void print(Vector v) {
    std::cout<<"[";
    for (size_t i=0;i<v.size();++i){
        std::cout<<" "<<v[i]<<",";
    }
    std::cout<<"]"<<std::endl;
}
void Matrix::print() {
    std::cout<<"[";
    for (size_t iRow = 0; iRow < size1_; ++iRow){
        for (size_t jCol = 0; jCol < size2_; ++jCol){
            // Result(iRow, jCol) = data_[iRow * size2_ + jCol] + M2(iRow,jCol);
            std::cout<<" "<<data_[iRow * size2_ + jCol]<<",";
        }
        std::cout<<std::endl;
    }
    std::cout<<"]"<<std::endl;

}
} // namespace linalg
} // namespace cie
