/* Pi - CUDA unified memory version
 * Author: Felipe Gutierrez, SBEL, July 2015
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
void calculateAreas(const long numRects, const double width, double *areas) 
{
/* If cuda is enabled calculate the threadId which gives us the index in areas */   
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
        areas[threadId] = (width * height);   

        /* Add Extra computations in order to be able to see the performance difference between CPU and GPU */
        x = sqrt((float)threadId) * pow(width,3);
        heightSq = 1 - (x*x);
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        areas[threadId] = (width * height); 

        x = threadId * pow(width,3);
        heightSq = 1 - (x*x);
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        areas[threadId] = (width * height); 

        x = threadId * pow(width,3);
        heightSq = 1 - (sqrt((float)x)*pow(width,3));
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        areas[threadId] = (width * height); 

        x = sqrt((float)x) * sqrt((float)x);
        heightSq = 1 - (pow(x,4)*x);
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        areas[threadId] = (width * height); 

        x = threadId * width;
        heightSq = 1 - (x*x);
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        areas[threadId] = (width * pow(x,0.5)); 

        x = threadId * width;
        heightSq = 1 - (x*x);
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));
        areas[threadId] = (width * height); 

        x = sqrt((float)threadId) * pow(width,3);
        heightSq = 1 - (x*x);
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        areas[threadId] = (width * height); 

        x = threadId * pow(width,3);
        heightSq = 1 - (x*x);
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        areas[threadId] = (width * height); 

        x = threadId * pow(width,3);
        heightSq = 1 - (sqrt((float)x)*pow(width,3));
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        areas[threadId] = (width * height); 

        x = sqrt((float)x) * sqrt((float)x);
        heightSq = 1 - (pow(x,4)*x);
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        areas[threadId] = (width * height); 

        x = threadId * width;
        heightSq = 1 - (x*x);
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        areas[threadId] = (width * pow(x,0.5)); 

        x = threadId * width;
        heightSq = 1 - (x*x);
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));
        areas[threadId] = (width * height); 

        x = sqrt((float)threadId) * pow(width,3);
        heightSq = 1 - (x*x);
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        areas[threadId] = (width * height); 

        x = threadId * pow(width,3);
        heightSq = 1 - (x*x);
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        areas[threadId] = (width * height); 

        x = threadId * pow(width,3);
        heightSq = 1 - (sqrt((float)x)*pow(width,3));
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        areas[threadId] = (width * height); 

        x = sqrt((float)x) * sqrt((float)x);
        heightSq = 1 - (pow(x,4)*x);
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        areas[threadId] = (width * height); 

        x = threadId * width;
        heightSq = 1 - (x*x);
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        areas[threadId] = (width * pow(x,0.5)); 

        x = threadId * width;
        heightSq = 1 - (x*x);
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));
        areas[threadId] = (width * height); 

        x = sqrt((float)threadId) * pow(width,3);
        heightSq = 1 - (x*x);
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        areas[threadId] = (width * height); 

        x = threadId * pow(width,3);
        heightSq = 1 - (x*x);
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        areas[threadId] = (width * height); 

        x = threadId * pow(width,3);
        heightSq = 1 - (sqrt((float)x)*pow(width,3));
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        areas[threadId] = (width * height); 

        x = sqrt((float)x) * sqrt((float)x);
        heightSq = 1 - (pow(x,4)*x);
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        areas[threadId] = (width * height); 

        x = threadId * width;
        heightSq = 1 - (x*x);
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt((float)heightSq));
        areas[threadId] = (width * pow(x,0.5)); 

        x = threadId * width;
        heightSq = 1 - (x*x);
        height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));
        areas[threadId] = (width * height); 

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
    /* Allocate areas in unified memory */
    double *unifiedAreas;
    err = cudaMallocManaged(&unifiedAreas, numRects * sizeof(double));

    /* Check for unified memory error*/
    if(err != cudaSuccess)
    {
        fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(err)); 
    }

    /* Call calculateAreas function. This function can be a CUDA kernel or a normal function depending of what is specified in CMakeLists.txt.
    Note: Unified memory allows us to send the same pointer to the allocated memory, no matter if we plan to use GPU memory or CPU memory in the function.
    */
   
    cudaEvent_t startKernel, stopKernel, stopSync, stopAll;
    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);
    cudaEventCreate(&stopSync);
    cudaEventCreate(&stopAll);

    float milliseconds = 0;


    cudaEventRecord(startKernel);
    calculateAreas KERNEL(numRects) (numRects, (1.0 / numRects), unifiedAreas);
    cudaEventRecord(stopKernel);


#if CUDA_ENABLED
    /* If cuda is enabled we want to do the reduce in GPU */
    cudaDeviceSynchronize(); // Synchronize the valued calculated in the kernel.
    cudaEventRecord(stopSync);
    
    cudaEventSynchronize(stopKernel);
    cudaEventSynchronize(stopSync);

    cudaEventElapsedTime(&milliseconds, startKernel, stopKernel);
    std::cout << "Kernel execution time = " << milliseconds/1000 << std::endl;
    cudaEventElapsedTime(&milliseconds, startKernel, stopSync);
    std::cout << "Kernel execution + sync time = " << milliseconds/1000 << std::endl;

    (*area) = thrust::reduce(thrust::cuda::par, unifiedAreas, unifiedAreas + numRects);

    cudaEventRecord(stopAll);
    cudaEventSynchronize(stopAll);

    cudaEventElapsedTime(&milliseconds, startKernel, stopAll);
    std::cout << "Kernel execution + thrust reduce time = " << milliseconds/1000 << std::endl;

#elif OPENMP_ENABLED
    /* If cuda is not enabled but openmp is we want to do the reduce in the cpu with openmp */
    (*area) = thrust::reduce(thrust::omp::par, unifiedAreas, unifiedAreas + numRects);
#else
    /* If neither is enabled we do it serially*/
    // (*area) = thrust::reduce(thrust::cpp::par, unifiedAreas, unifiedAreas + numRects);
    for (int i = 0; i < numRects; i++) 
    {
        (*area) += unifiedAreas[i];
    }
#endif



    cudaFree(unifiedAreas);
}
