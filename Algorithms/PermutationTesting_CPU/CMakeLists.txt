cmake_minimum_required (VERSION 2.8)

project(PermTesting_OpenMP CXX)

# Armadillo needs to be compiled for C++11 in order to use large matrices
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Finda CUDA and OPENMP
#if(OPENMP_ENABLED)
#    find_package("OpenMP")
#    if (OPENMP_FOUND)
#        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    endif()
#endif()


#add_executable(TestMatrixMult TestMatrixMult.cpp)
add_executable(PermTesting_Iterative PermTesting_Iterative.cpp)
target_link_libraries(PermTesting_Iterative ${ARMADILLO_LIBRARIES})

add_executable(PermTesting_Matrix PermTesting_Matrix.cpp)
target_link_libraries(PermTesting_Matrix ${ARMADILLO_LIBRARIES})

add_executable(PermTesting_SparseMatrix PermTesting_SparseMatrix.cpp)
target_link_libraries(PermTesting_SparseMatrix ${ARMADILLO_LIBRARIES})
#add_executable(TestMaxT TestMaxT.cpp)
#target_link_libraries(TestMaxT ${ARMADILLO_LIBRARIES})









