cmake_minimum_required(VERSION 3.15)

project(fashion_mnist_inference VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(fashion_mnist
    main.cpp
)

target_compile_features(fashion_mnist PRIVATE cxx_std_17)

if (MSVC)
    target_compile_options(fashion_mnist PRIVATE /W4 /permissive-)
else()
    target_compile_options(fashion_mnist PRIVATE -Wall -Wextra -Wpedantic)
endif()
