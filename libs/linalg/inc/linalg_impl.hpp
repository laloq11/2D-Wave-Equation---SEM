/*
 * Although we are not using templates we implement trivial
 * functions inline if they are called very often, such that
 * the compiler can optimize these function calls. The conse-
 * quence is that when we change something in this file we
 * have to recompile everything that includes linalg.hpp.
 */
#include <array>


namespace cie
{
namespace linalg
{

inline double& Matrix::operator()( size_t i, size_t j )
{
     return data_[i * size2_ + j];
}

inline double Matrix::operator()( size_t i, size_t j ) const
{
     return data_[i * size2_ + j];
}

inline size_t Matrix::size1( ) const
{
    return size1_;
}

inline size_t Matrix::size2( ) const
{
    return size2_;
}

inline std::array<size_t, 2> Matrix::sizes( ) const
{
    return { size1_, size2_ };
}

} // namespace linalg
} // namespace cie
