#pragma once

#include <cassert>
#include <climits>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>


using int32 = int;

using int64 = long int;

using size_type = std::size_t;

/**
 * Instantiates a template for each non-complex value type compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take one argument, which is replaced by the
 *                value type.
 */
#define INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(_macro) \
    template _macro(float);                                 \
    template _macro(double)


/**
 * Instantiates a template for each value type compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take one argument, which is replaced by the
 *                value type.
 */
#define INSTANTIATE_FOR_EACH_VALUE_TYPE(_macro)          \
    INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(_macro); \
    template _macro(std::complex<float>);                \
    template _macro(std::complex<double>)


/**
 * Instantiates a template for each index type compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take one argument, which is replaced by the
 *                value type.
 */
#define INSTANTIATE_FOR_EACH_INDEX_TYPE(_macro) \
    template _macro(int32);                     \
    template _macro(int64)


/**
 * Instantiates a template for each index type compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take one argument, which is replaced by the
 *                value type.
 */
#define INSTANTIATE_FOR_EACH_INDEX_AND_SIZE_TYPE(_macro) \
    template _macro(int32);                              \
    template _macro(int64);                              \
    template _macro(unsigned long);                      \
    template _macro(unsigned int)


/**
 * Instantiates a template for each value and index type compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take two arguments, which are replaced by the
 *                value and index types.
 */
#define INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(_macro) \
    template _macro(float, int32);                        \
    template _macro(double, int32);                       \
    template _macro(std::complex<float>, int32);          \
    template _macro(std::complex<double>, int32);         \
    template _macro(float, int64);                        \
    template _macro(double, int64);                       \
    template _macro(std::complex<float>, int64);          \
    template _macro(std::complex<double>, int64)


/**
 * Instantiates a template for each value and index type compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take two arguments, which are replaced by the
 *                value and index types.
 */
#define INSTANTIATE_FOR_EACH_POD_TYPE(_macro) \
    template _macro(float);                   \
    template _macro(double);                  \
    template _macro(std::complex<float>);     \
    template _macro(std::complex<double>);    \
    template _macro(size_type);               \
    template _macro(int32);                   \
    template _macro(int64)
