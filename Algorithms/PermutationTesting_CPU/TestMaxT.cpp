#include <iostream>
#include "armadillo"

int main()
{
    std::cout << "Starting MaxT test" << std::endl;
    arma::mat MaxT_Iterative;
    arma::mat MaxT_Matrix;
    arma::mat permutationMatrix1;
    arma::mat permutationMatrix2;
    arma::mat permutations;
    std::string permutationsArmaPath = "data/permutations.arma";
    std::string permutationMatrix1ArmaPath = "data/Matrix1.arma";
    std::string permutationMatrix2ArmaPath = "data/Matrix2.arma";

    MaxT_Iterative.load("MaxT_Iterative_100.arma");
    MaxT_Matrix.load("MaxT_Matrix_100.arma");

    std::cout << "MaxT_Iterative numRows = " << MaxT_Iterative.n_rows << std::endl;
    std::cout << "MaxT_Iterative numCols = " << MaxT_Iterative.n_cols << std::endl;
    std::cout << "MaxT_Iterative(0,0) = " << MaxT_Iterative(0,0) << std::endl;
    std::cout << "MaxT_Iterative(10,0) = " << MaxT_Iterative(10,0) << std::endl;
    std::cout << "MaxT_Iterative(99,0) = " << MaxT_Iterative(99,0) << std::endl;

    std::cout << "MaxT_Matrix numRows = " << MaxT_Matrix.n_rows << std::endl;
    std::cout << "MaxT_Matrix numCols = " << MaxT_Matrix.n_cols << std::endl;
    std::cout << "MaxT_Matrix(0,0) = " << MaxT_Matrix(0,0) << std::endl;
    std::cout << "MaxT_Matrix(10,0) = " << MaxT_Matrix(10,0) << std::endl;
    std::cout << "MaxT_Matrix(99,0) = " << MaxT_Matrix(99,0) << std::endl;

    permutations.load(permutationsArmaPath);
    permutationMatrix1.load(permutationMatrix1ArmaPath);
    permutationMatrix2.load(permutationMatrix2ArmaPath);
    std::cout << "permutations numRows = " << permutations.n_rows << std::endl;
    std::cout << "permutations numCols = " << permutations.n_cols << std::endl;
    std::cout << "permutationMatrix1 numRows = " << permutationMatrix1.n_rows << std::endl;
    std::cout << "permutationMatrix1 numCols = " << permutationMatrix1.n_cols << std::endl;
    permutations(1,arma::span(0,24)).print();
    std::cout << "---------------------------------------" << std::endl;
    permutations(1,arma::span(25,49)).print();
    std::cout << "---------------------------------------" << std::endl;
    permutationMatrix1(1,arma::span::all).print();
    std::cout << "---------------------------------------" << std::endl;
    permutationMatrix2(1,arma::span::all).print();



    std::cout << "End test" << std::endl;
    return 0;
}