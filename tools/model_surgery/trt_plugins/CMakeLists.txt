cmake_minimum_required(VERSION 3.18)
project(gather_elements_axis1_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED TENSORRT_ROOT)
    if(DEFINED ENV{TENSORRT_ROOT})
        set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
    elseif(DEFINED ENV{Tensorrt_DIR})
        set(TENSORRT_ROOT $ENV{Tensorrt_DIR})
    endif()
endif()

if(NOT TENSORRT_ROOT)
    message(FATAL_ERROR "Set TENSORRT_ROOT (or env TENSORRT_ROOT/Tensorrt_DIR) to TensorRT install path.")
endif()

message(STATUS "TENSORRT_ROOT=${TENSORRT_ROOT}")
include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)

add_library(gather_elements_axis1_plugin SHARED gather_elements_axis1_plugin.cu)
target_include_directories(gather_elements_axis1_plugin PRIVATE ${TENSORRT_ROOT}/include)
target_link_libraries(gather_elements_axis1_plugin PRIVATE nvinfer cudart)

