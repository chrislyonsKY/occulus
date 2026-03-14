cmake_minimum_required(VERSION 3.15)
project(occulus_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find pybind11 — prefer pip-installed version
find_package(pybind11 CONFIG REQUIRED)

# Core library (shared across all bindings)
add_library(occulus_core STATIC
    src/kdtree.cpp
    src/filters.cpp
    src/normals.cpp
    src/registration.cpp
    src/segmentation.cpp
    src/features.cpp
)

target_include_directories(occulus_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Compiler warnings
target_compile_options(occulus_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -O2>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
)

# pybind11 extension module → occulus._cpp._core
pybind11_add_module(_core src/bindings.cpp)

target_link_libraries(_core PRIVATE occulus_core)

# Install into the correct Python package location
install(TARGETS _core LIBRARY DESTINATION occulus/_cpp)
