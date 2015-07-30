cmake_minimum_required (VERSION 2.8)

project(pi)

# Find both CUDA and OPENMP
find_package("OpenMP")
find_package("CUDA")

# Set if we want to use CUDA, OPENMP or neither
set(CUDA_ENABLED 0)
set(OPENMP_ENABLED 1)


add_definitions( -DCUDA_ENABLED=${CUDA_ENABLED} )
add_definitions( -DOPENMP_ENABLED=${OPENMP_ENABLED} )
add_definitions( -DNUM_RECTS=${NUM_RECTS} )


# If OPENMP was found and it is enabled compile with its flags
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

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


#Print all variables. Useful for debugging
#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()


cuda_add_executable(piUniMem pi-UnifiedMem.cpp pi-UnifiedMem-kernel.cu)
cuda_add_executable(piNonUniMem pi-NonUnifiedMem.cpp pi-NonUnifiedMem-kernel.cu)

