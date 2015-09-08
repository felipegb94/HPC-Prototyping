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

#if CUDA_ENABLED
__global__ 
#endif
void calculateAreas(const long numRects, const double width, double *dev_areas, int currDevice, int totalNumRects); 
{
    int xOffset = currDevice * width * (ceil(totalNumRects/2));

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
        double x = (threadId * width) + xOffset;
        double heightSq = 1 - (x*x);
        double height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));
        
        dev_areas[threadId] = (width * height); 
    }
}

void calculateArea(const long numRects, double *area) {

    double *hostAreas1;
    double *hostAreas2;
    double *deviceAreas1;
    double *deviceAreas2;
    double (*area1) = 0.0;
    double (*area2) = 0.0;

    const long numRects1;
    const long numRects2;

    if((numRects % 2) == 0)
    {
        numRects1 = numRects/2;
        numRects2 = numRects/2;
    }
    else
    {
        numRects1 = ceil(numRects/2);
        numRects2 = floor(numRects/2);
    }

    int i;

/////////////////////////////// MEMORY ALLOCATION SECTION ////////////////////////////////////////

/* If CUDA is enabled allocate memory in device either using cudaMalloc or cudaMallocManaged */
#if CUDA_ENABLED
    cudaError_t err;
    int gpu1 = 0;
    int gpu2 = 1;
    if(getGridDim(numRects) >= 65535)
    {
        fprintf(stderr, "Error: WAY TOO MANY RECTANGLES. Do you really want to compute more than 4.3979123e+12 rectangles!!!! Please input less rectangles");
        return;
    }
    std::cout << "Grid Dimensions 1 = " << getGridDim(numRects1) << std::endl;
    std::cout << "Grid Dimensions 2 = " << getGridDim(numRects2) << std::endl;
    cudaSetDevice(gpu1);
    err = cudaMalloc(&deviceAreas1, numRects1 * sizeof(double));
    cudaSetDevice(gpu2);
    err = cudaMalloc(&deviceAreas2, numRects2 * sizeof(double));

    /* Check for error in device memory allocation */
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
    }

/* If CUDA is not enabled we are running on the CPU either serially or with openmp so we allocate memory in the host */
#else 
    hostAreas1 = (double*)malloc(numRects * sizeof(double));
    if (hostAreas1 == NULL) 
    {
        fprintf(stderr, "malloc failed!\n");
    }
#endif


/////////////////////////////// KERNEL CALL SECTION ////////////////////////////////////////

/* If CUDA is enabled do the kernel and reduce call either with unifiedMemory or with device memory*/
#if CUDA_ENABLED
    /* Start all cudaEvents so we can record timings */
    //GpuTimer kernelTimer("Kernel");
    //GpuTimer reduceTimer("Reduce");
    //GpuTimer allTimer("All");

    //allTimer.Start();
    //kernelTimer.Start();

    calculateAreas KERNEL(numRects2) (numRects2, (1.0 / numRects), deviceAreas2, gpu2, numRects);
    cudaSetDevice(gpu1);
    calculateAreas KERNEL(numRects1) (numRects1, (1.0 / numRects), deviceAreas1, gpu1, numRects);



    //kernelTimer.Stop();

    //reduceTimer.Start();
    (*area1) = thrust::reduce(thrust::cuda::par, deviceAreas1, deviceAreas1 + numRects1);
    cudaSetDevice(gpu2);
    (*area2) = thrust::reduce(thrust::cuda::par, deviceAreas2, deviceAreas2 + numRects2);

    cudaDeviceSync();
    (*area) = (*area1) + (*area2);

    //reduceTimer.Stop();
    //allTimer.Stop();
    //kernelTimer.print();
    //reduceTimer.print();
    //allTimer.print();
    cudaFree(deviceAreas2);
    cudaSetDevice(gpu1);
    cudaFree(deviceAreas1);

/* If CUDA is not enabled calculateAreas is not a kernel but a normal function. */
#else 
    /* This kernel call could also be given unifiedMemory as argument but for organization purposes it is called with hostAreas1 */
    CpuTimer kernelTimer("Kernel");
    CpuTimer reduceTimer("Reduce");
    CpuTimer allTimer("All");

    allTimer.Start();
    allTimer.Start_cputimer();

    kernelTimer.Start();
    kernelTimer.Start_cputimer();
    calculateAreas KERNEL(numRects) (numRects, (1.0 / numRects), hostAreas1);
    kernelTimer.Stop_cputimer();
    kernelTimer.Stop();

    (*area) = 0.0;
    reduceTimer.Start();
    reduceTimer.Start_cputimer();
    for (i = 0; i < numRects; i++) 
    {
        (*area) += hostAreas1[i];
    }
    reduceTimer.Stop_cputimer();
    reduceTimer.Stop();

    allTimer.Stop_cputimer();
    allTimer.Stop();

    kernelTimer.print();    
    reduceTimer.print();    
    allTimer.print();    

    free(hostAreas1);
#endif

///////////////////// GPU OR CPU FREE THE MEMORY ////////////////////

}

