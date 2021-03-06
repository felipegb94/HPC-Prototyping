#include <math.h>
#if OPENMP_ENABLED
    #include <omp.h>
#endif
#define ARMA_64BIT_WORD
#include "armadillo"
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
    std::string dataPath = "/home/felipe/data/RawADRC/data.mat";
    std::string dataArmaPath = "/home/felipe/data/RawADRC/data.arma";
    std::string permutationsPath = "/home/felipe/data/RawADRC/permutations.mat";
    std::string permutationsArmaPath = "/home/felipe/data/RawADRC/permutations.arma";

    int N_g1 = 25;

    arma::mat data;
    arma::mat permutations;
    arma::mat T;
    arma::mat MaxT;

    data.load(dataArmaPath);
    //data.save(dataArmaPath);
    permutations.load(permutationsArmaPath);
    //permutations.save(permutationsArmaPath);

    /* Get dimensions */   
    int N = data.n_rows; 
    int N_g2 = N - N_g1;
    int V = data.n_cols;
    int nPermutations = permutations.n_rows;
    std::cout << "Number of subjects (rows in data matrix): " << N << std::endl;
    std::cout << "Number of voxels per subject (cols in data matrix): " << data.n_cols << std::endl;
    std::cout << "Number of Permutations (rows in permutations matrix):" << nPermutations << std::endl;

    //T = arma::zeros(nPermutations, V);
    MaxT = arma::zeros(nPermutations, 1);
    /* Do Permutation Testing */
    arma::urowvec label_j(1, N);
    arma::mat group1 = arma::zeros(N_g1, V);
    arma::mat group2 = arma::zeros(N_g2, V);
    arma::mat tstat = arma::zeros(1, V);


    /* Permutation loop */
    #pragma omp parallel for
    for(int i = 0;i < nPermutations ;i++ )
    {
        std::cout << "Permutation " << i << std::endl;
        for(int j = 0; j < N; j++)
        {
            label_j(j) = permutations(i, j) - 1;
        }
        group1 = data.rows(label_j(arma::span(0, N_g1-1)));
        group2 = data.rows(label_j(arma::span(N_g1, N-1)));   
        tstat = ttest2(group1, group2);
        MaxT(i, arma::span::all) = arma::max(tstat,1);
    }
    std::cout << T.n_rows << std::endl;
    std::cout << T.n_cols << std::endl;

    MaxT.save("MaxT_Iterative_10000.arma");
    MaxT.save("MaxT_Iterative_10000.ascii", arma::raw_ascii);

    std::cout << T.n_rows << std::endl;
    std::cout << T.n_cols << std::endl;




    return 0;
}



