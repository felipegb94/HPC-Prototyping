#include <iostream>
#include <fstream>
#include <vector>
#include "../../PermutationTesting_OpenMP/include/armadillo"

int main()
{
    std::string dataPath = "../../PermutationTesting_OpenMP/data/RawADRC/adrc_raw.arma";
    std::string permutationsPath = "../../PermutationTesting_OpenMP/data/RawADRC/permutations.arma";
    int N_g1 = 25;

    arma::mat data;
    arma::mat permutations;
    data.load(dataPath);
    permutations.load(permutationsPath);

    std::cout << data.n_rows << std::endl;
    std::cout << data.n_cols << std::endl;
    std::cout << permutations.n_rows << std::endl;
    std::cout << permutations.n_cols << std::endl;


    return 0;
}