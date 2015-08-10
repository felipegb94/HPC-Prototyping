cmake_minimum_required (VERSION 2.8)

project(pi)

# Find both CUDA and OPENMP
find_package("OpenMP")
find_package("CUDA")

# Set if we want to use CUDA, OPENMP or neither
set(CUDA_ENABLED 1)
set(OPENMP_ENABLED 0)
set(UNIFIEDMEM_ENABLED 0)


# Disable compile optimizations for gcc and nvcc
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O0")


add_definitions( -DCUDA_ENABLED=${CUDA_ENABLED} )
add_definitions( -DOPENMP_ENABLED=${OPENMP_ENABLED} )
add_definitions( -DUNIFIEDMEM_ENABLED=${UNIFIEDMEM_ENABLED} )

# If OPENMP was found and it is enabled compile with its flags
if(CUDA_ENABLED)
    if(CUDA_FOUND)
        message("CUDA is enabled and it was found. Compiling with CUDA flags")
        message("Including CUDA directories...")
        cuda_include_directories(${CUDA_SDK_ROOT_DIR}/common/inc)
        cuda_include_directories(/opt/cuda/4.1/cuda/include)
        include_directories(${CUDA_SDK_ROOT_DIR}/common/inc)
        cuda_include_directories(${CUDA_SDK_ROOT_DIR}/../shared/inc)

        MATH (EXPR arch 8*${CMAKE_SIZEOF_VOID_P})

        message("Finding Libraries...")
        find_library(LIBSHRUTIL NAMES shrUtils${arch} shrutil_${CMAKE_SYSTEM_PROCESSOR} PATHS ${CUDA_SDK_LIB_DIR})
        find_library(LIBSHRUTIL_DBG NAMES shrUtils${arch}D shrutil_${CMAKE_SYSTEM_PROCESSOR}D PATHS ${CUDA_SDK_LIB_DIR})

        message("Setting nvcc flags...")
        macro(set_compute_capability cc)
                list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${cc},code=sm_${cc}")
                list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${cc},code=compute_${cc}")
        endmacro(set_compute_capability)

        set_compute_capability("20")
        cuda_add_executable(pi pi.cpp pi-kernel.cu)
        cuda_add_executable(piThrust pi.cpp pi-kernel-useThrust.cu)

    else()
        message("CUDA is enabled but it was not found. Compiling serially")
        set(CUDA_ENABLED 0)
        set(OPENMP_ENABLED 0)
        add_definitions( -DCUDA_ENABLED=${CUDA_ENABLED} )
        add_definitions( -DOPENMP_ENABLED=${OPENMP_ENABLED} )
        add_executable(pi pi.cpp pi-kernel.cpp)
    endif()

elseif(OPENMP_ENABLED)
    if (OPENMP_FOUND)
        message("OpenMP is enabled and it was found. Compiling with OpenMP flags")
        message(${OpenMP_C_FLAGS})
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    else() 
        message("OpenMP is enabled but it was not found. Compiling Serially...")
        set(CUDA_ENABLED 0)
        set(OPENMP_ENABLED 0)
        add_definitions( -DCUDA_ENABLED=${CUDA_ENABLED} )
        add_definitions( -DOPENMP_ENABLED=${OPENMP_ENABLED} )
    endif()
    add_executable(pi pi.cpp pi-kernel.cpp)  

else()
    message("Neither CUDA or OpenMP are enabled. Compiling Serially")
    add_executable(pi pi.cpp pi-kernel.cpp)
endif()



