#include <math.h>
//#include <omp.h>
#include "/home/felipegb94/include/armadillo"

arma::mat ttest2(arma::mat group1, arma::mat group2)
{

    int n = group1.n_rows;
    int m = group2.n_rows;

    arma::mat tstat = arma::zeros(1, group1.n_cols);
    arma::mat group1_mean = arma::mean(group1); // 1 x Voxels vector
    arma::mat group2_mean = arma::mean(group2); // 1 x Voxels vector
    arma::mat group1_var = arma::var(group1); // 1 x Voxels vector
    arma::mat group2_var = arma::var(group2); // 1 x Voxels vector

    arma::mat mean_difference = group1_mean - group2_mean;
    arma::mat denominator = sqrt( (group1_var / n) + (group2_var / m) );

    arma::mat t_stat = mean_difference / denominator;

    return t_stat;
}


int main()
{
    std::string dataPath = "/home/felipegb94/data/RawADRC/adrc_raw.arma";
    std::string permutationsPath = "/home/felipegb94/data/RawADRC/permutations.arma";
    int N_g1 = 25;

    arma::mat data;
    arma::mat permutations;

    data.load(dataPath);
    //data.save("/Users/felipegb94/sbel/repos/HPC-Prototyping/Algorithms/PermutationTesting_CUDA/data/adrc_raw.csv",arma::csv_ascii);
    permutations.load(permutationsPath);
    //permutations.save("/Users/felipegb94/sbel/repos/HPC-Prototyping/Algorithms/PermutationTesting_CUDA/data/permutations.csv",arma::csv_ascii);


    std::cout << data.n_rows << std::endl;
    std::cout << data.n_cols << std::endl;
    std::cout << permutations.n_rows << std::endl;
    std::cout << permutations.n_cols << std::endl;

    /* Do Permutation Testing */
    /* N x V matrix*/
    int N = data.n_rows; 
    int num_permutations = permutations.n_rows;
    arma::mat T = arma::zeros(permutations.n_rows, data.n_cols);
    arma::mat MaxT = arma::zeros(permutations.n_rows, 1);
    /* Permutation loop */
    #pragma omp parallel for
    for(int i = 0;i < num_permutations ;i++ )
    {
        std::cout << "Permutation " << i << std::endl;
        arma::urowvec label_j(1, N);
        for(int j = 0; j < N; j++)
        {
            label_j(j) = permutations(i, j) - 1;
        }
        arma::mat group1 = data.rows(label_j(arma::span(0, N_g1-1)));
        arma::mat group2 = data.rows(label_j(arma::span(N_g1, N-1)));   

        arma::mat tstat = ttest2(group1, group2);
        T(i, arma::span::all) = tstat;
    }
    MaxT = arma::max(T,1);
    MaxT.save("RawADRC_MaxT_10000.csv", arma::csv_ascii);

    std::cout << T.n_rows << std::endl;
    std::cout << T.n_cols << std::endl;




    return 0;
}



