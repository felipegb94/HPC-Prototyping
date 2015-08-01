/* PermTesting - CUDA Implementation
 * Author: Felipe Gutierrez, July 2015
 */

#include <stdio.h> /* fprintf() */
#include <iostream>
#include <float.h> /* DBL_EPSILON() */
#include <math.h> /* sqrt() */ 
#include <vector>

#include "PermTesting-Kernel.cuh"

#define NUMTHREADS 1024
#define NUMBLOCKS(n) ceil(n/NUMTHREADS)

void printInfo(stdvecvec &data, stdvecvec &permutations);
void printDeviceInfo();
__global__ void PermTestingDevice(int numPermutations, int N, int V, int N_gp1,
                                  double *dataDevice, 
                                  int *permutationsDevice,
                                  double *MaxTDevice)
{
    int threadId = threadIdx.x + (blockIdx.x * blockDim.x); // Current Permutation
    printf("Starting thread: %d \n", threadId);
    int permutationsStart = threadId * N;
    int i,j;
    int N_gp2 = N - N_gp1; // Size of group 2
    double group1Sum = 0;
    double group2Sum = 0;
    double group1SumSquared = 0; // Sum of all terms squared of group1 : x1^2 + x2^2 ...
    double group2SumSquared = 0; // Sum of all terms squared of group2 
    double group1Mean = 0;
    double group2Mean = 0;
    double group1Var = 0;
    double group2Var = 0;
    double meanDifference = 0; // t-statistics numerator
    double denominator = 0; // t-statistic denominator
    double MaxT = 0;
    double tStat = 0;

    double voxelStatistic = 0;
    int currSubject = 0;

    /* For each voxel calculate a t-statistic*/
    for(i = 0; i < V;i++)
    {
        group1Sum = 0;
        group2Sum = 0;
        group1SumSquared = 0;
        group2SumSquared = 0;     

        /* Add statistics of the first group */
        for(j = 0;j < N_gp1;j++)
        {
            currSubject = permutationsDevice[permutationsStart + j] - 1;
            voxelStatistic = dataDevice[currSubject*V];
            group1Sum = group1Sum + voxelStatistic;
            group1SumSquared = group1SumSquared + voxelStatistic*voxelStatistic;
        }

        /* Add statistics of second group */
        for(j = N_gp1; j < N; j++)
        {
            currSubject = permutationsDevice[permutationsStart + j] - 1;
            voxelStatistic = dataDevice[currSubject*V];
            group2Sum = group2Sum + voxelStatistic;
            group2SumSquared = group2SumSquared + voxelStatistic*voxelStatistic;
        }

        group1Mean = group1Sum/N_gp1;
        group2Mean = group2Sum/N_gp2;

        group1Var = (group1SumSquared/N_gp1) - (group1Mean*group1Mean);
        group2Var = (group2SumSquared/N_gp2) - (group2Mean*group2Mean);

        meanDifference = group1Mean - group2Mean;
        denominator = sqrt((group1Var / N_gp1) + (group2Var / N_gp2));

        tStat = meanDifference/denominator;
        if(tStat > MaxT)
        {
            MaxT = tStat;
        }        
    }

    MaxTDevice[threadId] = MaxT;
}


void PermTesting(stdvecvec &dataVec, stdvecvec &permutationsVec, int N_gp1)
{
    std::cout << "Starting Permutation Testing..." << std::endl;
    printInfo(dataVec, permutationsVec);
    int i, j;
    cudaError_t err;

    /* Get dimensions for arrays */
    int numPermutations = permutationsVec.size();
    int N = dataVec.size();
    int V = dataVec.front().size();

    /* Allocate memory for result */
    double *data = (double*)malloc(N*V * sizeof(double));
    int *permutations = (int*)malloc(N*numPermutations * sizeof(int));
    double *MaxT = (double*)malloc(numPermutations * sizeof(double));

    /* Copy data and permutations to a 1D array. */
    for(i = 0; i < dataVec.size(); i++)
    {
        for(j = 0; j < dataVec.front().size();j++)
        {
            data[i*V + j] = dataVec[i][j];
        }
    }
    for(i = 0; i < permutationsVec.size(); i++)
    {
        for(j = 0; j < permutationsVec.front().size();j++)
        {
            permutations[i*N + j] = permutationsVec[i][j];
        }
    }
    /* Test if the copy was correct */
    //for(i  = 0; i < 10;i++)
    //{
    //    std::cout << "data[i] = " << data[i+40*V] << std::endl;
    //    std::cout << "dataVec[0][i] = " << dataVec[40][i] << std::endl;
    //    std::cout << "permutations[i] = " << permutations[i+N] << std::endl;
    //    std::cout << "permutationsVec[0][i] = " << permutationsVec[1][i] << std::endl;
    //}


    /**
     * What do we have to allocate on the GPU memory?
     *     1. dataDevice: N*V 2D array --> Large 1D array
     *     2. permutationsDevice: N*numPermutations 2D array --> large 1D array
     *     3. MaxTDevice: 1*numPermutations
     */
    double *dataDevice;
    int *permutationsDevice;
    double *MaxTDevice;
    /* Allocate memory in GPU */
    err = cudaMalloc((void**)&dataDevice, N*V * sizeof(double));
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
    }

    err = cudaMalloc((void**)&permutationsDevice, N*numPermutations * sizeof(int));
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
    }

    err = cudaMalloc((void**)&MaxTDevice, numPermutations * sizeof(double));
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
    }

    /* Copy data to the memory allocated in the GPU */
    err = cudaMemcpy(dataDevice, data, N*V * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(permutationsDevice, permutations, N*numPermutations * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
    }

    printDeviceInfo();


//<<<NUMBLOCKS(1), NUMTHREADS>>>
    PermTestingDevice<<<1, 1>>>(numPermutations, N, V, 
                                                                    N_gp1, dataDevice, 
                                                                    permutationsDevice,
                                                                    MaxTDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "kernel call failed: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(MaxT, MaxTDevice, numPermutations * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
    }

    // for(i = 0;i < numPermutations;i++)
    // {
    //     std::cout << "MaxT[i] = " << MaxT[i] << std::endl;
    // }

    free(data);
    free(permutations);
    free(MaxT);
    cudaFree(dataDevice);
    cudaFree(permutationsDevice);
    cudaFree(MaxTDevice);


}


void printInfo(stdvecvec &data, stdvecvec &permutations)
{
    std::cout << "Sample size = " << data.size() << std::endl;
    std::cout << "Number of voxels per sample = " << data.front().size() << std::endl;
    std::cout << "Number of Permutations = " << permutations.size() << std::endl;
}

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

