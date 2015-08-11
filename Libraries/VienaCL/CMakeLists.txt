cmake_minimum_required (VERSION 2.8)

project(TestVienaCL)

set(OPENMP_BACKEND 0)
set(CUDA_BACKEND 0)
set(OPENCL_BACKEND 0)

add_definitions( -DCUDA_ENABLED=${CUDA_ENABLED} )
add_definitions( -DOPENMP_ENABLED=${OPENMP_ENABLED} )
add_definitions( -DUNIFIEDMEM_ENABLED=${UNIFIEDMEM_ENABLED} )

# Find VienaCL


add_executable(testVienaCL TestVienaCL.cpp)


