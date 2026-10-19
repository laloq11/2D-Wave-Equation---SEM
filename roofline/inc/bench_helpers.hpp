#pragma once

#include <chrono>
#include <cstring>

#include "devices.hpp"
#include "error_handling.hpp"
#include "matrix.hpp"


#define TIME_FUNC(__time, ...)                                               \
    {                                                                        \
        auto start = std::chrono::steady_clock::now();                       \
        __VA_ARGS__;                                                         \
        auto finish = std::chrono::steady_clock::now();                      \
        __time += std::chrono::duration_cast<std::chrono::duration<double>>( \
                      finish - start)                                        \
                      .count();                                              \
    }


template <typename ValueType>
void reset_data(Vector<ValueType>& data, std::string& device)
{
    if (device == "cuda") {
        ASSERT_NO_CUDA_ERRORS(cudaMemset(
            data.data, 0.0, sizeof(ValueType) * data.num_stored_elems()));
    } else {
        std::memset(data.data, 0, sizeof(ValueType) * data.num_stored_elems());
    }
}
