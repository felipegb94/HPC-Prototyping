cmake_minimum_required (VERSION 2.8)
project(PermTesting_GPU)

# Compile with C++11
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# Find ArrayFire and Armadillo
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_PATH})

if(${MPI_C_FOUND})
    message("MPI for C was found!")
endif()
if(${MPI_CXX_FOUND})
    message("MPI for C++ was found!")
endif()
if(${CUDA_FOUND})
    message("CUDA was found!")
endif()

if(MPI_COMPILE_FLAGS)
  set_target_properties(hello PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(hello PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

add_executable(pi-mpi-1 pi-mpi-1.c pi-io.c pi-sync-data-1.c pi-mpi.c)
add_executable(pi-mpi-2 pi-mpi-2.c pi-io.c pi-sync-data-2.c pi-mpi.c)
cuda_add_executable(pi-mpi-cuda pi-mpi-cuda.cpp pi-io.c pi-sync-data-2.c pi-mpi-cuda.cu)

target_link_libraries(pi-mpi-1 ${MPI_LIBRARIES})
target_link_libraries(pi-mpi-2 ${MPI_LIBRARIES})
target_link_libraries(pi-mpi-cuda ${MPI_LIBRARIES} ${CUDA_LIBRARIES})

