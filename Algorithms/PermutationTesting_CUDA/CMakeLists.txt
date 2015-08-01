cmake_minimum_required (VERSION 2.8)

project(PermTesting_CUDA)


find_package("CUDA")
cuda_include_directories(${CUDA_SDK_ROOT_DIR}/common/inc)
cuda_include_directories(/opt/cuda/4.1/cuda/include)
include_directories(${CUDA_SDK_ROOT_DIR}/common/inc)
cuda_include_directories(${CUDA_SDK_ROOT_DIR}/../shared/inc)
MATH (EXPR arch 8*${CMAKE_SIZEOF_VOID_P})
find_library(LIBSHRUTIL NAMES shrUtils${arch} shrutil_${CMAKE_SYSTEM_PROCESSOR} PATHS ${CUDA_SDK_LIB_DIR})
find_library(LIBSHRUTIL_DBG NAMES shrUtils${arch}D shrutil_${CMAKE_SYSTEM_PROCESSOR}D PATHS ${CUDA_SDK_LIB_DIR})
macro(set_compute_capability cc)
        list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${cc},code=sm_${cc}")
        list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${cc},code=compute_${cc}")
endmacro(set_compute_capability)
set_compute_capability("20")



cuda_add_executable(PermTesting_CUDA PermTesting.cpp PermTesting-Kernel.cu)









