#include <iostream>
#include "armadillo"
int main()
{
    std::cout << "Starting arma multiplication test" << std::endl;
    arma::mat m1 = arma::randu(10000,10000);
    arma::mat m2 = arma::randu(10000,10000);
    arma::mat m3 = m1*m2;
    std::cout << "End test" << std::endl;
    return 0;
}