/* Pi - CUDA version 1 - uses integers for CUDA kernels
 * Author: Felipe Gutierrez, SBEL, July 2015
 */
#include <iostream>
#include <stdio.h> /* fprintf()  */
#include <cstdlib> /*  malloc and free */
#include <float.h> /* DBL_EPSILON() */
#include <math.h> /* sqrt() */
#include <ctime>

#include "pi-kernel.h"

/* Only add openmp if it will be used */
#if OPENMP_ENABLED
    #include <omp.h>
#endif


/**
 * @brief CUDA macro
 * @details 
 *         If CUDA is enabled we need to define:
 *         * nthreads = Number of threads per block we want.
 *         
 *         * NUMBLOCKS = Gives the number of blocks we want to use to parallelize a problem of 
 *         size n.
 *         
 *         * KERNEL =  KERNEL(n) to specified the number of blocks and the number of threads 
 *         per block if CUDA is ENABLED. If CUDA is not enabled then KERNEL(n) is just an empty 
 *         piece of code.
 * 
 */
#if CUDA_ENABLED
    #include "TimerGPU.h"
    #include <thrust/reduce.h>
    #include <thrust/system/cuda/execution_policy.h>
    #include <thrust/system/omp/execution_policy.h>
    #define nthreads 1024
    #define getGridDim(n) (int)ceil(sqrt(n/nthreads))
    #define GRID(n) dim3(getGridDim(n), getGridDim(n))
    #define BLOCK(n) dim3(nthreads)
    #define KERNEL(n) <<<GRID(n), BLOCK(n)>>> /* Necessary for kernels */
#else
    #include "TimerCPU.h"
    #define KERNEL(n) /* Empty code */
#endif

/**
 * @brief calculateAreas kernel
 * @details 
 *         * threadId: Index in the areas area. Tells us where to store the calculated area. With 
 *         CUDA this is calculated with threadId and blockId. In serial and OpenMP this is the 
 *         obtained by the for loop counter.       
 *         * x: Current x coordinate
 *         * heightSq: height of rectangle squared
 * 
 * @param numRects numRects we are going to use to estimate the area under the curve. This defines
 * how big our problem size will be. This is the n in KERNEL(n).
 * 
 * @param width of rectangle
 * 
 * @param areas Pre allocated array that will contain  areas. --> This array was allocated with 
 * cudaMallocManaged() function which is what leads to UnifiedMemory.
 * 
 * @return fills the areas array
 */

#if CUDA_ENABLED
__global__ 
#endif
void calculateAreas(const long numRects, const double width, double *dev_areas) 
{
/* If cuda is enabled calculate the threadId which gives us the index in dev_areas */   
#if CUDA_ENABLED
    /* Calculate threadId for 1D grid 1D block*/
    //int threadId = threadIdx.x + (blockIdx.x * blockDim.x);
    /* Calculate threadId for 2D grid 1D block*/
    int threadId = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;
    if(threadId >= numRects)
    {
        return;
    }
#else
    /* We don't have to delete the #pragma clause for the serial version of the code. If the program is not compiled with -fopenmp and omp.h is not included the compiler will just ignore the #pragma clause. */
    #pragma omp parallel for
    /* Define the for loop if cuda is not enable. This is used in both the serial and openmp version */
    for(int threadId = 0;threadId < numRects;threadId++)
#endif
    {
        double x = threadId * width;
        double heightSq = 1 - (x*x);
        double height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));
        dev_areas[threadId] = (width * height); 
    }
}

void calculateArea(const long numRects, double *area) 
{
    double *unifiedAreas;
    int i;

/////////////////////////////// MEMORY ALLOCATION SECTION ////////////////////////////////////////

/* If CUDA is enabled allocate memory in device either using cudaMalloc or cudaMallocManaged */
    cudaError_t err;

    if(getGridDim(numRects) >= 65535)
    {
        fprintf(stderr, "Error: WAY TOO MANY RECTANGLES. Do you really want to compute more than 4.3979123e+12 rectangles!!!! Please input less rectangles");
        return;
    }

    std::cout << "Grid Dimensions = " << getGridDim(numRects) << std::endl;
    printf("Unified Memory is Enabled. Allocating using cudaMallocManaged \n");
    err = cudaMallocManaged(&unifiedAreas, numRects * sizeof(double));

    /* Check for error in device memory allocation */
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc or cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
    }
    
/////////////////////////////// KERNEL CALL SECTION ////////////////////////////////////////

/* If CUDA is enabled do the kernel and reduce call either with unifiedMemory or with device memory*/
/* If CUDA is not enabled calculateAreas is not a kernel but a normal function. */

    calculateAreas KERNEL(numRects) (numRects, (1.0 / numRects), unifiedAreas);

    (*area) = 0.0;

#if CUDA_ENABLED
    (*area) = thrust::reduce(thrust::cuda::par, unifiedAreas, unifiedAreas + numRects);
#else
    for (i = 0; i < numRects; i++) 
    {
        (*area) += unifiedAreas[i];
    }
#endif

    cudaFree(unifiedAreas);

///////////////////// GPU OR CPU FREE THE MEMORY ////////////////////

}

#if CUDA_ENABLED
void printDeviceInfo()
{
    int device;
    struct cudaDeviceProp props;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);

    std::cout << "Device info: " <<std::endl;
    std::cout << "Name: " << props.name <<std::endl;
    std::cout << "version: " << props.major << "," <<  props.minor <<std::endl;
}
#endif