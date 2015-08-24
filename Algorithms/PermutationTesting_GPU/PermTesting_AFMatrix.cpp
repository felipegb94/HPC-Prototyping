#include <iostream>
#include <vector>
#include <stdlib.h>

#include "arrayfire.h"
#include "armadillo"

int getIntervalDimension(int nVoxels)
{
    double maxMemoryPerInterval = 5e8; //2 gigabytes
    double totalMemoryPerCol = (double)(nVoxels * sizeof(double));
    double colMemMaxMemRatio = totalMemoryPerCol/maxMemoryPerInterval;
    /* Estimate to the nearest hundredth */
    double numPermutationsPerInterval = 100 * floor((1/(100*colMemMaxMemRatio)));
    if((numPermutationsPerInterval*nVoxels*sizeof(double)) > maxMemoryPerInterval)
    {
        std::cout << "Too big of an interval!!!!!!!!!!!" << std::endl;
    }
    return numPermutationsPerInterval;
}

int main()
{
    std::cout << "Starting Permutation Testing with ArrayFire!" << std::endl;

    std::string dataPath = "../../PermutationTesting_CPU/build/data/data.mat";
    std::string dataArmaPath = "../../PermutationTesting_CPU/build/data/data.arma";
    std::string permutationsPath = "../../PermutationTesting_CPU/build/data/permutations.mat";
    std::string permutationsArmaPath = "../../PermutationTesting_CPU/build/data/permutations.arma";
    std::string permutationMatrix1Path = "../../PermutationTesting_CPU/build/data/Matrix1.mat";
    std::string permutationMatrix1ArmaPath = "../../PermutationTesting_CPU/build/data/Matrix1.arma";
    std::string permutationMatrix2Path = "../../PermutationTesting_CPU/build/data/Matrix2.mat";
    std::string permutationMatrix2ArmaPath = "../../PermutationTesting_CPU/build/data/Matrix2.arma";

    arma::mat data;
    arma::mat permutationMatrix1;
    arma::mat permutationMatrix2;
    data.load(dataArmaPath);
    permutationMatrix1.load(permutationMatrix1ArmaPath);
    permutationMatrix2.load(permutationMatrix2ArmaPath);

    /* Get dimensions */   
    int N = data.n_rows; 
    int N_g1 = 25;
    int N_g2 = N-N_g1;
    int V = data.n_cols;
    int nPermutations = permutationMatrix1.n_rows;

    std::cout << "Number of subjects (rows in data matrix): " << N << std::endl;
    std::cout << "Number of voxels per subject (cols in data matrix): " << V << std::endl;
    std::cout << "Number of Permutations (rows in permutations matrix):" << nPermutations << std::endl;
    std::cout << "Size of group1 = " << N_g1 << std::endl;
    std::cout << "Size of group2 = " << N_g2 << std::endl;
    std::cout << "Rows in PermMatrix1 = " << permutationMatrix1.n_rows << std::endl;
    std::cout << "Cols in PermMatrix1 = " << permutationMatrix1.n_cols << std::endl;
    std::cout << "Rows in PermMatrix2 = " << permutationMatrix2.n_rows << std::endl;
    std::cout << "Cols in PermMatrix2 = " << permutationMatrix2.n_cols << std::endl;

    float *dataHostPointer = new float[N*V];
    float *permutationMatrix1Host = new float[N*nPermutations];
    float *permutationMatrix2Host = new float[N*nPermutations];

    for(int i = 0;i < N;i++)
    {
        for(int j = 0; j < V;j++)
        {
            dataHostPointer[i*V + j] = data(i,j);
        }
    } 
    for(int i = 0;i < nPermutations;i++)
    {
        for(int j = 0; j < N;j++)
        {
            permutationMatrix1Host[i*N + j] = permutationMatrix1(i,j);
            permutationMatrix2Host[i*N + j] = permutationMatrix2(i,j);
        }
    }     

    af::array dataDevice(V,N,dataHostPointer);
    af::array dataSquaredDevice = dataDevice*dataDevice;
    af::array permutationMatrix1Device(N,nPermutations,permutationMatrix1Host);
    af::array permutationMatrix2Device(N,nPermutations,permutationMatrix2Host);

    int intervalSize = getIntervalDimension(V);
    intervalSize = intervalSize/2;
    int numPasses = nPermutations/intervalSize;
    std::cout << "Interval Size = " << intervalSize << std::endl;
    std::cout << "Number of Passes = " << numPasses << std::endl;


    af::array g1Mean(N, intervalSize);
    af::array g2Mean(N, intervalSize);
    af::array g1Var(N, intervalSize);
    af::array g2Var(N, intervalSize);
    af::array tStatMatrix(N, intervalSize);
    af::array MaxT(1,nPermutations);

    int start, end;
    for(int i = 0;i < numPasses;i++)
    {
        start = intervalSize * i;
        end = (intervalSize * i) + intervalSize - 1;
        std::cout << "Curr Pass = " << i << std::endl;
        g1Mean = af::matmul(dataDevice, permutationMatrix1Device(af::span, af::seq(start, end))) / N_g1 ;
        g2Mean = af::matmul(dataDevice, permutationMatrix2Device(af::span, af::seq(start, end))) / N_g2;
        g1Var = (af::matmul(dataSquaredDevice, permutationMatrix1Device(af::span, af::seq(start, end))) / (N_g1-1)) - (g1Mean*g1Mean);

        g2Var = (af::matmul(dataSquaredDevice, permutationMatrix2Device(af::span, af::seq(start, end))) / (N_g2-1)) - (g2Mean*g2Mean); 

        tStatMatrix = (g1Mean - g2Mean) / (af::sqrt((g1Var/N_g1) + (g2Var/N_g2)));

        MaxT(af::seq(start,end)) = af::max(tStatMatrix,0);

    }
    
    af::saveimage("MaxT_ArrayFire_10000.af",MaxT);

    af_print(MaxT(af::seq(0,intervalSize-1)));
    //
    // af::array A(2,2,hA);
    // af_print(A);
    // af_print(af::sqrt(A));
    // af_print(A + A);
    // af_print(A - A);
    // af_print(A/2);



    return 0;

}