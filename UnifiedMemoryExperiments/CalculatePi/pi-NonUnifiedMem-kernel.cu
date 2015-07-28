/* Pi - CUDA version 1 - uses integers for CUDA kernels
 * Author: Aaron Weeden, Shodor, May 2015
 * Modified By: Felipe Gutierrez
 */
#include <stdio.h> /* fprintf() */
#include <iostream>
#include <float.h> /* DBL_EPSILON() */
#include <math.h> /* sqrt() */

#if OPENMP_ENABLED
    #include <omp.h>
#endif

#define nthreads 1000

#if CUDA_ENABLED
    #define NUMBLOCKS(n) ceil(n/nthreads)
    #define KERNEL(n) <<<NUMBLOCKS(n), nthreads>>>
#else
    #define KERNEL(n)
#endif

#if CUDA_ENABLED
__global__ 
#endif
void calculateAreas(const int numRects, const double width, double *dev_areas) 
{
#if CUDA_ENABLED
    int threadId = threadIdx.x + (blockIdx.x * blockDim.x);
    if(threadId >= numRects)
    {
        return;
    }
#elif OPENMP_ENABLED
    #pragma omp parallel 
#endif

#if !CUDA_ENABLED
    for(int threadId = 0;threadId < numRects;threadId++)
#endif

    {
        double x = threadId * width;
        double heightSq = 1 - (x*x);
        double height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));
        dev_areas[threadId] = (width * height);   
    }
}

void calculateArea(const int numRects, double *area) {

    /* Allocate areas in host */
    double *areas = (double*)malloc(numRects * sizeof(double));
    double *dev_areas;
    int i = 0;
    cudaError_t err;

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
