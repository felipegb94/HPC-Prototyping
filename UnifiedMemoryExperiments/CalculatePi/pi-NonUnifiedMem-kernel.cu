/* Pi - CUDA version 1 - uses integers for CUDA kernels
 * Author: Aaron Weeden, Shodor, May 2015
 * Modified By: Felipe Gutierrez
 */
#include <stdio.h> /* fprintf() */
#include <iostream>
#include <float.h> /* DBL_EPSILON() */
#include <math.h> /* sqrt() */

#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>

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
        // x = sqrt((float)threadId) * pow(width,3);
        // heightSq = 1 - (x*x);
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        // dev_areas[threadId] = (width * height); 

        // x = threadId * pow(width,3);
        // heightSq = 1 - (x*x);
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        // dev_areas[threadId] = (width * height); 

        // x = threadId * pow(width,3);
        // heightSq = 1 - (sqrt((float)x)*pow(width,3));
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        // dev_areas[threadId] = (width * height); 

        // x = sqrt((float)x) * sqrt((float)x);
        // heightSq = 1 - (pow(x,4)*x);
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        // dev_areas[threadId] = (width * height); 

        // x = threadId * width;
        // heightSq = 1 - (x*x);
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        // dev_areas[threadId] = (width * pow(x,0.5)); 

        // x = threadId * width;
        // heightSq = 1 - (x*x);
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));
        // dev_areas[threadId] = (width * height); 

        // x = sqrt((float)threadId) * pow(width,3);
        // heightSq = 1 - (x*x);
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        // dev_areas[threadId] = (width * height); 

        // x = threadId * pow(width,3);
        // heightSq = 1 - (x*x);
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        // dev_areas[threadId] = (width * height); 

        // x = threadId * pow(width,3);
        // heightSq = 1 - (sqrt((float)x)*pow(width,3));
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        // dev_areas[threadId] = (width * height); 

        // x = sqrt((float)x) * sqrt((float)x);
        // heightSq = 1 - (pow(x,4)*x);
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        // dev_areas[threadId] = (width * height); 

        // x = threadId * width;
        // heightSq = 1 - (x*x);
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        // dev_areas[threadId] = (width * pow(x,0.5)); 

        // x = threadId * width;
        // heightSq = 1 - (x*x);
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));
        // dev_areas[threadId] = (width * height); 

        // x = sqrt((float)threadId) * pow(width,3);
        // heightSq = 1 - (x*x);
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        // dev_areas[threadId] = (width * height); 

        // x = threadId * pow(width,3);
        // heightSq = 1 - (x*x);
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        // dev_areas[threadId] = (width * height); 

        // x = threadId * pow(width,3);
        // heightSq = 1 - (sqrt((float)x)*pow(width,3));
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        // dev_areas[threadId] = (width * height); 

        // x = sqrt((float)x) * sqrt((float)x);
        // heightSq = 1 - (pow(x,4)*x);
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        // dev_areas[threadId] = (width * height); 

        // x = threadId * width;
        // heightSq = 1 - (x*x);
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        // dev_areas[threadId] = (width * pow(x,0.5)); 

        // x = threadId * width;
        // heightSq = 1 - (x*x);
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));
        // dev_areas[threadId] = (width * height); 

        // x = sqrt((float)threadId) * pow(width,3);
        // heightSq = 1 - (x*x);
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        // dev_areas[threadId] = (width * height); 

        // x = threadId * pow(width,3);
        // heightSq = 1 - (x*x);
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        // dev_areas[threadId] = (width * height); 

        // x = threadId * pow(width,3);
        // heightSq = 1 - (sqrt((float)x)*pow(width,3));
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        // dev_areas[threadId] = (width * height); 

        // x = sqrt((float)x) * sqrt((float)x);
        // heightSq = 1 - (pow(x,4)*x);
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        // dev_areas[threadId] = (width * height); 

        // x = threadId * width;
        // heightSq = 1 - (x*x);
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        // dev_areas[threadId] = (width * pow(x,0.5)); 

        // x = threadId * width;
        // heightSq = 1 - (x*x);
        // height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));
        // dev_areas[threadId] = (width * height); 
    }
}

void calculateArea(const long numRects, double *area) {

    cudaError_t err;
#if CUDA_ENABLED
    if(getGridDim(numRects) >= 65535)
    {
        fprintf(stderr, "Error: WAY TOO MANY RECTANGLES. Do you really want to compute more than 4.3979123e+12 rectangles!!!! Please input less rectangles");
        return;
    }
    std::cout << "Grid Dimensions = " << getGridDim(numRects) << std::endl;
#endif   

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

    cudaEvent_t startKernel, stopKernel, stopSync, stopAll;
    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);
    cudaEventCreate(&stopSync);
    cudaEventCreate(&stopAll);

    cudaEventRecord(startKernel);
    calculateAreas KERNEL(numRects) (numRects, (1.0 / numRects), dev_areas);
    cudaEventRecord(stopKernel);

    // err = cudaMemcpy(areas, dev_areas, (numRects * sizeof(double)), cudaMemcpyDeviceToHost);
    // if (err != cudaSuccess) 
    // {
    //     fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
    // }
    //cudaEventRecord(stopSync);

    cudaEventSynchronize(stopKernel);
    cudaEventSynchronize(stopSync);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startKernel, stopKernel);
    std::cout << "Kernel execution time = " << milliseconds/1000 << std::endl;
    //cudaEventElapsedTime(&milliseconds, startKernel, stopSync);
    //std::cout << "Kernel execution + sync time = " << milliseconds/1000 << std::endl;

    (*area) = thrust::reduce(thrust::cuda::par, dev_areas, dev_areas + numRects);
    cudaEventRecord(stopAll);
    
    cudaEventSynchronize(stopAll);
    cudaEventElapsedTime(&milliseconds, startKernel, stopAll);
    std::cout << "Kernel execution + thrust reduce time = " << milliseconds/1000 << std::endl;

#else 
    calculateAreas KERNEL(numRects) (numRects, (1.0 / numRects), areas);
    (*area) = 0.0;
    for (i = 0; i < numRects; i++) 
    {
        (*area) += areas[i];
    }
#endif


    cudaFree(dev_areas);
    free(areas);
}
