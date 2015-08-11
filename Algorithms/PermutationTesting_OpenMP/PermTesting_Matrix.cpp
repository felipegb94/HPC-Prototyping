#include <iostream>
#include <cstdio>

#include <math.h>
//#include "/home/felipegb94/include/armadillo"
#define ARMA_64BIT_WORD
#include "armadillo"
//#include "../include/armadillo"

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
    std::string dataPath = "../data/RawADRC/adrc_raw.arma";
    std::string permutationsPath = "../data/RawADRC/permutations.arma";
    int N_g1 = 25;
    int i;

    arma::mat data;
    arma::mat dataSquared;
    arma::mat permutations;

    data.load(dataPath);
    //data.save("/Users/felipegb94/sbel/repos/HPC-Prototyping/Algorithms/PermutationTesting_CUDA/data/adrc_raw.csv",arma::csv_ascii);
    permutations.load(permutationsPath);

    //permutations.save("/Users/felipegb94/sbel/repos/HPC-Prototyping/Algorithms/PermutationTesting_CUDA/data/permutations.csv",arma::csv_ascii);
 
    /* Change indexing from matlab to C++ */
    permutations = permutations - 1;

    /* Get dimensions */   
    int N = data.n_rows; 
    int N_g2 = N-N_g1;
    int V = data.n_cols;
    int nPermutations = permutations.n_rows;
    std::cout << "Number of subjects (rows in data matrix): " << N << std::endl;
    std::cout << "Number of voxels per subject (cols in data matrix): " << data.n_cols << std::endl;
    std::cout << "Number of Permutations (rows in permutations matrix):" << nPermutations << std::endl;


    arma::mat g1Permutations = permutations(arma::span::all,arma::span(0,N_g1-1));
    arma::mat g2Permutations = permutations(arma::span::all,arma::span(N_g1,N-1));
    arma::mat permutationMatrix1 = arma::zeros(nPermutations, N);
    arma::mat permutationMatrix2 = arma::zeros(nPermutations, N);
    /* uvec needed for matrix element access. For more info look at Armadillo docs */
    arma::uvec g1Indices;
    arma::uvec g2Indices;
    arma::uvec perm;

    std::cout << "Number of subject in g1: " << g1Permutations.n_cols << std::endl;
    std::cout << "Number of subject in g2: " << g2Permutations.n_cols << std::endl;

    perm << 0; 
    int g1Counter = 0;
    int g2Counter = 0;

    for(i = 0;i < nPermutations;i++)
    {
        perm(0,0) = i;
        g1Indices = arma::conv_to<arma::uvec>::from(g1Permutations(i, arma::span::all));
        g2Indices = arma::conv_to<arma::uvec>::from(g2Permutations(i, arma::span::all));
        permutationMatrix1(perm,g1Indices) = arma::ones<arma::rowvec>(N_g1);
        permutationMatrix2(perm,g2Indices) = arma::ones<arma::rowvec>(N_g2);
    }
    std::cout << "Num rows in permutationMatrix1: " << permutationMatrix1.n_rows << std::endl;
    std::cout << "Num cols in permutationMatrix1: " << permutationMatrix1.n_cols << std::endl;
    std::cout << "Num rows in permutationMatrix2: " << permutationMatrix2.n_rows << std::endl;
    std::cout << "Num cols in permutationMatrix2: " << permutationMatrix2.n_cols << std::endl;

    
    int intervalDim = getIntervalDimension(V);


    if(nPermutations % intervalDim == 0)
    {
        int nPasses = nPermutations/intervalDim;
        int start, end;
        arma::mat permInterval1;
        arma::mat permInterval2;
        arma::mat sumMatrix1;
        arma::mat sumMatrix2;
        arma::mat sumMatrixSquared1;
        arma::mat sumMatrixSquared2;
        arma::mat g1Mean;
        arma::mat g2Mean;
        arma::mat g1Var;
        arma::mat g2Var;
        arma::mat meanDifference;
        arma::mat denominator;
        arma::mat tStatMatrix;
        arma::mat MaxT = arma::zeros(nPermutations,1);
        dataSquared = data % data;
        std::cout << "nPasses = " << nPasses << std::endl;
        for(i = 0; i < nPasses;i++)
        //for(i = 0; i < 1;i++)
        {
            std::cout << "Current Interval = " << i << std::endl;
            start = intervalDim*i;
            end = (intervalDim*i) + intervalDim - 1;
            permInterval1 = permutationMatrix1(arma::span(start,end),arma::span::all);
            permInterval2 = permutationMatrix2(arma::span(start,end),arma::span::all);

            sumMatrix1 = permInterval1 * data;
            sumMatrix2 = permInterval2 * data;

            sumMatrixSquared1 = permInterval1 * dataSquared;
            sumMatrixSquared2 = permInterval2 * dataSquared;

            g1Mean = sumMatrix1/N_g1;
            g2Mean = sumMatrix2/N_g2;
            g1Var = (sumMatrixSquared1/N_g1) - (g1Mean%g1Mean);
            g2Var = (sumMatrixSquared2/N_g2) - (g2Mean%g2Mean);

            meanDifference = g1Mean - g2Mean;
            denominator = sqrt( (g1Var / N_g1) + (g2Var / N_g2) );
            tStatMatrix = meanDifference / denominator;

            MaxT(arma::span(start,end), arma::span::all) = arma::max(tStatMatrix,1);
        }
        MaxT.save("MaxT.arma");
        MaxT.save("MaxT.ascii",arma::raw_ascii);

    }
    else
    {
        fprintf(stderr, "Invalid intervalDim");
        return 1;
    }



    return 0;
}