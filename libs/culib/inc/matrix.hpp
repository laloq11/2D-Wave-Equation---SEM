#pragma once

#include <vector>

#include "devices.hpp"
#include "types.hpp"

namespace cie {
    namespace otFlyGPU {
        template <typename ValueType>
        struct Matrix {
            size_type num_rows;
            size_type num_cols;
            size_type stride;
            ValueType* data;
            ATTRIBUTES INLINE size_type num_stored_elems() { return num_rows * stride; }
        };


        template <typename ValueType>
        using Vector = Matrix<ValueType>;

        template <typename ValueType>
        struct Vectorcu {
            size_type num_rows;
            size_type num_cols;
            // size_type stride;
            ValueType* data;
            ATTRIBUTES INLINE size_type num_stored_elems() { return num_rows * num_cols; }
        };

        template <typename ValueType>
        struct VMat {
            size_type num_rows;
            size_type num_cols;
            size_type dim;
            // size_type stride;
            ValueType* data;
            ATTRIBUTES INLINE size_type num_stored_elems() { return num_rows * num_cols * dim; }
        };



}
}