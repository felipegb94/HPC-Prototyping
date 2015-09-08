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
    #include "math.h"
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

#if CUDA_ENABLED
void printDeviceInfo()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    std::cout << "Number of devices available = " << nDevices << std::endl;
    for (int j = 0; j < nDevices; j++) 
    {

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, j);
        printf("Device Number: %d\n", j);
        printf("Device name: %s\n", prop.name);
        printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
}
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

__global__ 
void calculateAreas(const long numRects, 
                    const double width, 
                    double *dev_areas, 
                    int currDevice, 
                    int totalNumRects) 
{
    double xOffset = currDevice * width * (ceil(float(totalNumRects/2)));

    /* If cuda is enabled calculate the threadId which gives us the index in dev_areas */   

    /* Calculate threadId for 1D grid 1D block*/
    //int threadId = threadIdx.x + (blockIdx.x * blockDim.x);
    /* Calculate threadId for 2D grid 1D block*/
    int threadId = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;
    if(threadId >= numRects)
    {
        return;
    }
    double x = (threadId * width) + xOffset;
    double heightSq = 1 - (x*x);
    double height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq)); 
    /* Add Extra computations in order to be able to see the performance difference between CPU and GPU */
    // x = sqrt((float)threadId) * pow(width,3);
    // heightSq = 1 - (x*x);
    // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
    
    x = threadId * pow(width,3);
    heightSq = 1 - (x*x);
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
    x = threadId * pow(width,3);
    heightSq = 1 - (sqrt((float)x)*pow(width,3));
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
    x = sqrt((float)x) * sqrt((float)x);
    heightSq = 1 - (pow(x,4)*x);
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
    x = threadId * width;
    heightSq = 1 - (x*x);
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
    x = threadId * width;
    heightSq = 1 - (x*x);
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));
    x = sqrt((float)threadId) * pow(width,3);
    heightSq = 1 - (x*x);
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
    x = threadId * pow(width,3);
    heightSq = 1 - (x*x);
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
    x = threadId * pow(width,3);
    heightSq = 1 - (sqrt((float)x)*pow(width,3));
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
    x = sqrt((float)x) * sqrt((float)x);
    heightSq = 1 - (pow(x,4)*x);
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
    x = threadId * width;
    heightSq = 1 - (x*x);
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
    x = threadId * width;
    heightSq = 1 - (x*x);
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));
    x = sqrt((float)threadId) * pow(width,3);
    heightSq = 1 - (x*x);
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
    x = threadId * pow(width,3);
    heightSq = 1 - (x*x);
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
    x = threadId * pow(width,3);
    heightSq = 1 - (sqrt((float)x)*pow(width,3));
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
    x = sqrt((float)x) * sqrt((float)x);
    heightSq = 1 - (pow(x,4)*x);
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
    x = threadId * width;
    heightSq = 1 - (x*x);
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
    x = threadId * width;
    heightSq = 1 - (x*x);
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));
    x = sqrt((float)threadId) * pow(width,3);
    heightSq = 1 - (x*x);
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
    x = threadId * pow(width,3);
    heightSq = 1 - (x*x);
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
    x = threadId * pow(width,3);
    heightSq = 1 - (sqrt((float)x)*pow(width,3));
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));  
    x = sqrt((float)x) * sqrt((float)x);
    heightSq = 1 - (pow(x,4)*x);
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
    x = threadId * width;
    heightSq = 1 - (x*x);
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
    x = (threadId * width) + xOffset;
    heightSq = 1 - (x*x);
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));

    heightSq = 1 - (x*x);
    height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));

    dev_areas[threadId] = (width * height); 
}

void calculateArea(const long totalNumRects, double *totalArea) 
{

    double *deviceAreas[2];
    int gpu[2] = {0,1};
    long numRects[2] = {0,0};
    double area[2] = {0.0,0.0};
    int i;

    if((totalNumRects % 2) == 0)
    {
    numRects[0] = totalNumRects/2;
        numRects[1] = totalNumRects/2;
    }
    else
    {
        numRects[0] = ceil(totalNumRects/2);
        numRects[1] = floor(totalNumRects/2);
    }


/////////////////////////////// MEMORY ALLOCATION SECTION ////////////////////////////////////////

/* If CUDA is enabled allocate memory in device either using cudaMalloc or cudaMallocManaged */
    cudaError_t err;
    cudaStream_t stream[2];

    if(getGridDim(numRects[0]) >= 65535)
    {
        fprintf(stderr, "Error: WAY TOO MANY RECTANGLES. Do you really want to compute more than 4.3979123e+12 rectangles!!!! Please input less rectangles");
        return;
    }
    std::cout << "Grid Dimensions 1 = " << getGridDim(numRects[0]) << std::endl;
    std::cout << "Grid Dimensions 2 = " << getGridDim(numRects[1]) << std::endl;

    for(i = 0; i < 2; i++)
    {
        cudaSetDevice(gpu[i]);
        cudaStreamCreate(&stream[i]);
        err = cudaMalloc(&deviceAreas[i], numRects[i] * sizeof(double));
        /* Check for error in device memory allocation */
        if (err != cudaSuccess) 
        {
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        }
    }
/////////////////////////////// KERNEL CALL SECTION ////////////////////////////////////////

/* If CUDA is enabled do the kernel and reduce call either with unifiedMemory or with device memory */
    for(i = 0;i < 2; i++)
    {
        cudaSetDevice(gpu[i]);

        calculateAreas<<<GRID(numRects[i]), BLOCK(numRects[i]), 0, stream[i]>>>(numRects[i], 
                                                                            (1.0 / totalNumRects), 
                                                                            deviceAreas[i], 
                                                                            gpu[i], 
                                                                            totalNumRects);
        area[i] =  thrust::reduce(thrust::cuda::par.on(stream[i]), deviceAreas[i], deviceAreas[i] + numRects[i]);
    }

    cudaDeviceSynchronize();

    (*totalArea) = (area[0]) + (area[1]);

/* If CUDA is not enabled calculateAreas is not a kernel but a normal function. */

///////////////////// GPU OR CPU FREE THE MEMORY ////////////////////
    for(i = 0;i < 2; i++)
    {
        cudaSetDevice(gpu[i]);
        cudaStreamDestroy(stream[i]);
        cudaFree(deviceAreas[i]);
    }
}

