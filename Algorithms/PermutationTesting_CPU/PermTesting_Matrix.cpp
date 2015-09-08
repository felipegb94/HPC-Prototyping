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
    std::string dataPath = "/home/felipe/data/RawADRC/data.mat";
    std::string dataArmaPath = "/home/felipe/data/RawADRC/data.arma";
    std::string permutationsPath = "/home/felipe/data/RawADRC/permutations.mat";
    std::string permutationsArmaPath = "/home/felipe/data/RawADRC/permutations.arma";
    std::string permutationMatrix1Path = "/home/felipe/data/RawADRC/Matrix1.mat";
    std::string permutationMatrix1ArmaPath = "/home/felipe/data/RawADRC/Matrix1.arma";
    std::string permutationMatrix2Path = "/home/felipe/data/RawADRC/Matrix2.mat";
    std::string permutationMatrix2ArmaPath = "/home/felipe/data/RawADRC/Matrix2.arma";

    int N_g1 = 25;
    int i;

    arma::mat data;
    arma::mat dataSquared;
    arma::mat permutations;
    arma::mat permutationMatrix1;
    arma::mat permutationMatrix2;

    /* Load data that just came from matlab */
    // permutations.load(permutationsPath, arma::raw_ascii);
    // permutationMatrix1.load(permutationMatrix1Path, arma::raw_ascii);
    // permutationMatrix2.load(permutationMatrix2Path, arma::raw_ascii);
    // permutations.save(permutationsArmaPath);
    // permutationMatrix1.save(permutationMatrix1ArmaPath);
    // permutationMatrix2.save(permutationMatrix2ArmaPath);

    data.load(dataArmaPath);
    permutations.load(permutationsArmaPath);
    permutationMatrix1.load(permutationMatrix1ArmaPath);
    permutationMatrix2.load(permutationMatrix2ArmaPath);
    dataSquared = data % data;

    /* Change indexing from matlab to C++ */

    /* Get dimensions */   
    int N = data.n_rows; 
    int N_g2 = N-N_g1;
    int V = data.n_cols;
    int nPermutations = permutationMatrix1.n_rows;
    std::cout << "Number of subjects (rows in data matrix): " << N << std::endl;
    std::cout << "Number of voxels per subject (cols in data matrix): " << data.n_cols << std::endl;
    std::cout << "Number of Permutations (rows in permutations matrix):" << nPermutations << std::endl;
    std::cout << "Size of group1 = " << N_g1 << std::endl;
    std::cout << "Size of group2 = " << N_g2 << std::endl;
    std::cout << "Rows in PermMatrix1 = " << permutationMatrix1.n_rows << std::endl;
    std::cout << "Cols in PermMatrix1 = " << permutationMatrix1.n_cols << std::endl;
    std::cout << "Rows in PermMatrix2 = " << permutationMatrix2.n_rows << std::endl;
    std::cout << "Cols in PermMatrix2 = " << permutationMatrix2.n_cols << std::endl;

    int intervalSize = getIntervalDimension(V);
    int numPasses = nPermutations/intervalSize;

    std::cout << "Interval Size = " << intervalSize << std::endl;
    std::cout << "Number of Passes = " << numPasses << std::endl;

    arma::mat sumMatrix1 = arma::zeros(intervalSize, N);
    arma::mat sumMatrix2 = arma::zeros(intervalSize, N);
    arma::mat sumMatrixSquared1 = arma::zeros(intervalSize, N);
    arma::mat sumMatrixSquared2 = arma::zeros(intervalSize, N);
    arma::mat g1Mean = arma::zeros(intervalSize, N);
    arma::mat g2Mean = arma::zeros(intervalSize, N);
    arma::mat g1Var = arma::zeros(intervalSize, N);
    arma::mat g2Var = arma::zeros(intervalSize, N);
    arma::mat meanDifference;
    arma::mat denominator;
    arma::mat tStatMatrix = arma::zeros(intervalSize, N);
    arma::mat MaxT = arma::zeros(nPermutations,1);

    if((nPermutations % intervalSize) != 0)
    {
        fprintf(stderr, "Error: Wrong intervalSize \n");
    }
    int start, end;
    for(int i = 0;i < 5;i++)
    {
        std::cout << "Curr Pass = " << i << std::endl;
        start = intervalSize * i;
        end = (intervalSize * i) + intervalSize - 1;
        sumMatrix1 = permutationMatrix1(arma::span(start,end), arma::span::all) * data;
        sumMatrix2 = permutationMatrix2(arma::span(start,end), arma::span::all) * data;
        sumMatrixSquared1 = permutationMatrix1(arma::span(start,end), arma::span::all)  * dataSquared;
        sumMatrixSquared2 = permutationMatrix2(arma::span(start,end), arma::span::all)  * dataSquared;
        g1Mean = sumMatrix1/N_g1;
        g2Mean = sumMatrix2/N_g2;
        g1Var = (sumMatrixSquared1/(N_g1-1)) - (g1Mean%g1Mean); 
        g2Var = (sumMatrixSquared2/(N_g2-1)) - (g2Mean%g2Mean); 
        tStatMatrix = (g1Mean - g2Mean) / (sqrt((g1Var/N_g1) + (g2Var/N_g2)));
        MaxT(arma::span(start,end),arma::span::all) = arma::max(tStatMatrix,1);
        //MaxT(arma::span(start,end),arma::span::all).print();
    }

    std::string armaFileName = "MaxT_Matrix_10000.arma";
    std::string asciiFileName = "MaxT_Matrix_10000.ascii";

    MaxT.save(armaFileName);
    MaxT.save(asciiFileName, arma::raw_ascii);




    return 0;
}