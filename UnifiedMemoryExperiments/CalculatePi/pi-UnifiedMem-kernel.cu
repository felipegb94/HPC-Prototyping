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
#include <thrust/system/cpp/execution_policy.h>


#if OPENMP_ENABLED
    #include <omp.h>
#endif

#define nthreads 512

#if CUDA_ENABLED

    #define NUMBLOCKS(n) ceil(n/nthreads)
    #define KERNEL(n) <<<NUMBLOCKS(n), nthreads>>>
#else
    #define KERNEL(n)
#endif

#if CUDA_ENABLED
__global__ 
#endif
void calculateAreas(const int numRects, const double width, double *areas) 
{
#if CUDA_ENABLED
    int threadId = threadIdx.x + (blockIdx.x * blockDim.x);
    if(threadId >= numRects)
    {
        return;
    }
#elif OPENMP_ENABLED
    #pragma omp parallel for
#endif

#if !CUDA_ENABLED
    for(int threadId = 0;threadId < numRects;threadId++)
#endif
    {
        double x = threadId * width;
        double heightSq = 1 - (x*x);
        double height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));
        areas[threadId] = (width * height);   

    //     x = threadId * width;
    //     heightSq = 1 - (x*x);
    //     height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));
    //     areas[threadId] = (width * height);   

    //     x = threadId * width;
    //     heightSq = 1 - (x*x);
    //     height = (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));
    //     areas[threadId] = (width * height);  

    //     double extraOp = threadId;
    //     extraOp = extraOp*2.876/5.2;
    //     extraOp = sqrt(extraOp)/23.2;

    //     extraOp = extraOp*2.876/5.2;
    //     extraOp = sqrt(extraOp)/23.2;

    //     extraOp = extraOp*2.876/5.2;
    //     extraOp = sqrt(extraOp)/23.2;

    //     extraOp = extraOp*2.876/5.2;
    //     extraOp = sqrt(extraOp)/23.2;

    //     extraOp = extraOp*2.876/5.2;
    //     extraOp = sqrt(extraOp)/23.2;

    //     extraOp = pow(extraOp,2)/23.2;
    //     extraOp = pow(extraOp,0.5)/23.2;
    //     extraOp = pow(extraOp,3)/23.2;

    //     extraOp = extraOp*2.876/5.2;
    //     extraOp = sqrt(extraOp)/23.2;

    //     extraOp = extraOp*2.876/5.2;
    //     extraOp = sqrt(extraOp)/23.2;

    //     extraOp = extraOp*2.876/5.2;
    //     extraOp = sqrt(extraOp)/23.2;

    //     extraOp = pow(extraOp,2)/23.2;
    //     extraOp = pow(extraOp,0.5)/23.2;
    //     extraOp = pow(extraOp,3)/23.2;
    }
}

void calculateArea(const int numRects, double *area) {

    cudaError_t err;
    dim3 blockDims(32,32);

    /* Allocate areas in unified memory */
    double *unifiedAreas;
    err = cudaMallocManaged(&unifiedAreas, numRects * sizeof(double));

    /* Check for unified memory error*/
    if(err != cudaSuccess)
    {
        fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(err)); 
    }

    calculateAreas KERNEL(numRects) (numRects, (1.0 / numRects), unifiedAreas);


#if CUDA_ENABLED
    /* If cuda is enabled we want to do the reduce in GPU */

    cudaDeviceSynchronize(); // Synchronize the valued calculated in the kernel.
    (*area) = thrust::reduce(thrust::cuda::par, unifiedAreas, unifiedAreas + numRects);

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
