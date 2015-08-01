#include <iostream> /* cout, endl*/
#include <iomanip> /* setprecision*/
#include <vector>

#include "/home/felipegb94/include/armadillo"
#include "PermTesting-Kernel.cuh"



stdvecvec mat_to_std_vec(arma::mat &A) {
    stdvecvec V(A.n_rows);
    for (size_t i = 0; i < A.n_rows; ++i) {
        V[i] = arma::conv_to< stdvec >::from(A.row(i));
    };
    return V;
}

int main()
{
    std::string dataPath = "/home/felipegb94/data/RawADRC/adrc_raw.arma";
    std::string permutationsPath = "/home/felipegb94/data/RawADRC/permutations.arma";
    int N_g1 = 25;

    arma::mat armaData;
    arma::mat armaPermutations;
    stdvecvec data;
    stdvecvec permutations;

    armaData.load(dataPath);
    armaPermutations.load(permutationsPath);
    std::cout << armaData.n_rows << std::endl;
    std::cout << armaData.n_cols << std::endl;
    std::cout << armaPermutations.n_rows << std::endl;
    std::cout << armaPermutations.n_cols << std::endl;

    data = mat_to_std_vec(armaData);
    permutations = mat_to_std_vec(armaPermutations);

    PermTesting(data, permutations, N_g1);




    return 0;
}