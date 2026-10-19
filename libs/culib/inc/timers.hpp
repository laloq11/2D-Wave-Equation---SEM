#pragma once

#include <chrono>

#define TIME_FUNC(__time, ...)                                               \
    {                                                                        \
        auto start = std::chrono::steady_clock::now();                       \
        __VA_ARGS__;                                                         \
        auto finish = std::chrono::steady_clock::now();                      \
        __time += std::chrono::duration_cast<std::chrono::duration<double>>( \
                      finish - start)                                        \
                      .count();                                              \
    }
