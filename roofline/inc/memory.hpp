#pragma once

#include <cstdlib>
#include <memory>

#include <matrix.hpp>
#include <types.hpp>


template <typename ValueType, typename IndexType = int>
size_type calculate_required_bytes(int n_valuetype, int n_indextype = 0)
{
    return sizeof(ValueType) * n_valuetype + sizeof(IndexType) * n_indextype;
}
