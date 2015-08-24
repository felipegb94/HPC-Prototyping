cmake_minimum_required (VERSION 2.8)

project(TestArrayFire)

# Find VienaCL
find_package(ArrayFire)
include_directories(${ArrayFire_INCLUDE_DIRS})
if(${ArrayFire_FOUND})
    message("ArrayFire was found!")
    if(${ArrayFire_CPU_FOUND})
        message("ArrayFire CPU libraries were found!")
    endif()
    if(${ArrayFire_OpenCL_FOUND})
        message("ArrayFire OpenCL libraries were found!")
    endif()

endif()


add_executable(TestMatrixMult TestMatrixMult.cpp)
target_link_libraries(TestMatrixMult ${ArrayFire_LIBRARIES})



