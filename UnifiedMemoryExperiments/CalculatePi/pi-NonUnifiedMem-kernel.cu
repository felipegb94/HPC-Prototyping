/* Pi - CUDA version 1 - uses integers for CUDA kernels
 * Author: Aaron Weeden, Shodor, May 2015
 * Modified By: Felipe Gutierrez
 */
#include <stdio.h> /* fprintf() */
#include <iostream>
#include <float.h> /* DBL_EPSILON() */
#include <math.h> /* sqrt() */

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
    #define nthreads 1024
    #define getGridDim(n) (int)ceil(sqrt(n/nthreads))
    #define GRID(n) dim3(getGridDim(n), getGridDim(n))
    #define BLOCK(n) dim3(nthreads)
    #define KERNEL(n) <<<GRID(n), BLOCK(n)>>> /* Necessary for kernels */
#else
    #define KERNEL(n) /* Empty code */
#endif

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
#elif OPENMP_ENABLED
    #pragma omp parallel for
#endif

#if !CUDA_ENABLED
    /* Define the for loop if cuda is not enable. This is used in both the serial and openmp version */
    for(int threadId = 0;threadId < numRects;threadId++)
#endif
    {
        double x = threadId * width;
        double heightSq = 1 - (x*x);
        double height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));
        dev_areas[threadId] = (width * height);   

        /* Add Extra computations in order to be able to see the performance difference between CPU and GPU */
    }
}

void calculateArea(const long numRects, double *area) {

    cudaError_t err;

    if(getGridDim(numRects) >= 65535)
    {
        fprintf(stderr, "Error: WAY TOO MANY RECTANGLES. Do you really want to compute more than 4.3979123e+12 rectangles!!!! Please input less rectangles");
        return;
    }
    
    std::cout << "Grid Dimensions = " << getGridDim(numRects) << std::endl;

    /* Allocate areas in host */
    double *areas = (double*)malloc(numRects * sizeof(double));
    double *dev_areas;
    int i = 0;

    /* Check for error in allocation*/
    if (areas == NULL) 
    {
        fprintf(stderr, "malloc failed!\n");
    }

    /* Allocate areas in device */
    err = cudaMalloc((void**)&dev_areas, (numRects * sizeof(double)));

    /* Check for error in allocation in device*/
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
    }

#if CUDA_ENABLED
    calculateAreas KERNEL(numRects) (numRects, (1.0 / numRects), dev_areas);
    err = cudaMemcpy(areas, dev_areas, (numRects * sizeof(double)), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
    }
#else 
    calculateAreas KERNEL(numRects) (numRects, (1.0 / numRects), areas);
#endif

    (*area) = 0.0;
    for (i = 0; i < numRects; i++) 
    {
        (*area) += areas[i];
    }
    cudaFree(dev_areas);
    free(areas);
}
