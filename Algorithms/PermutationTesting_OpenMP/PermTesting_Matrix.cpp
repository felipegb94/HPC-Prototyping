#include <iostream>
#include <math.h>
//#include "/home/felipegb94/include/armadillo"
#include "armadillo"
//#include "../include/armadillo"


int main()
{
    std::string dataPath = "../data/RawADRC/adrc_raw.arma";
    std::string permutationsPath = "../data/RawADRC/permutations.arma";
    int N_g1 = 25;

    arma::mat data;
    arma::mat permutations;

    data.load(dataPath);
    //data.save("/Users/felipegb94/sbel/repos/HPC-Prototyping/Algorithms/PermutationTesting_CUDA/data/adrc_raw.csv",arma::csv_ascii);
    permutations.load(permutationsPath);
    /* Change indexing from matlab to C++ */
    permutations = permutations - 1;
    //permutations.save("/Users/felipegb94/sbel/repos/HPC-Prototyping/Algorithms/PermutationTesting_CUDA/data/permutations.csv",arma::csv_ascii);
    
    int N = data.n_rows; 
    int num_permutations = permutations.n_rows;



    std::cout << data.n_rows << std::endl;
    std::cout << data.n_cols << std::endl;
    std::cout << permutations.n_rows << std::endl;
    std::cout << permutations.n_cols << std::endl;

    arma::mat g1Permutations = permutations(arma::span::all,arma::span(0,N_g1-1));
    arma::mat g2Permutations = permutations(arma::span::all,arma::span(N_g1,N-1));


    // std::cout << g1Permutations.n_rows << std::endl;
    // std::cout << g1Permutations.n_cols << std::endl;
    // std::cout << g2Permutations.n_rows << std::endl;
    // std::cout << g2Permutations.n_cols << std::endl;

    arma::uvec perm;
    perm << 0;
    arma::uvec g1Indices;
    arma::uvec g2Indices;
    arma::mat permutationMatrix = arma::zeros(num_permutations*2, N);
    int g1Counter = 0;
    int g2Counter = 0;

    for(int i = 0;i < 2*num_permutations;i++)
    {
        perm(0,0) = i;
        if(i % 2 == 0)
        {
            g1Indices = arma::conv_to<arma::uvec>::from(g1Permutations(g1Counter, arma::span::all));
            permutationMatrix(perm,g1Indices) = arma::ones<arma::rowvec>(N_g1);
            g1Counter++;
        }
        else
        {
            g2Indices = arma::conv_to<arma::uvec>::from(g2Permutations(g2Counter, arma::span::all));
            permutationMatrix(perm,g2Indices) = arma::ones<arma::rowvec>(N-N_g1);   
            g2Counter++;
        }

    }

    // arma::mat t1 = permutationMatrix(0,arma::span::all);
    // arma::mat t2 = permutationMatrix(1,arma::span::all);
    // t2 = t2.t();
    // arma::mat t3 = t1*t2;
    // t3.print();
    // std::cout << t1.n_rows << std::endl;
    // std::cout << t1.n_cols << std::endl;
    // std::cout << t2.n_rows << std::endl;
    // std::cout << t2.n_cols << std::endl;



    std::cout << permutationMatrix.n_rows << std::endl;
    std::cout << permutationMatrix.n_cols << std::endl;

    arma::mat sum = arma::max(data.t() * permutationMatrix.t());

    // arma::mat g1g2Permutations = arma::zeros(10,2);
    // arma::mat rowIndices;
    // arma::uvec colIndices;
    // colIndices << 0;

    // rowIndices << 1 << 3 << 5 << 7 << 9 << arma::endr
    //            << 0 << 2 << 4 << 6 << 8 << arma::endr;
    // rowIndices = rowIndices.t();
    // rowIndices.print();
    // g1g2Permutations.print();
    // std::cout << std::endl;

    // //g1g2Permutations(rowIndices,colIndices) = arma::ones<arma::vec>(5);
    // g1g2Permutations.print();

    // for(int i = 0;i < 2;i++)
    // {
    //     arma::uvec c;
    //     c << i;
    //     colIndices(0,0) = i;
    //     arma::vec r2 = rowIndices(arma::span::all,i);
    //     arma::uvec r = arma::conv_to<arma::uvec>::from(r2);
    //     //rowIndices(arma::span::all, i).print();
    //     //r.print();
    //     g1g2Permutations(r,c) = arma::ones<arma::vec>(5);
    //     g1g2Permutations.print();
    // }

    return 0;
}