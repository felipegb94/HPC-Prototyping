#include <iostream>

#include "arrayfire.h"
#include <af/util.h>

int main()
{
    std::cout << "Testing ArrayFire" << std::endl;
    af::info();
    af::array r = af::randn(2, 2);
    af::array r2 = af::randn(2, 2);
    af::array result = r*r2;
    //std::cout << result(0,0) << std::endl;
    af_print(result);

    // Generate 10,0000 random values
    af::array a = af::randu(1000000);
    // Sum the values and copy the result to the CPU:
    af::timer t = af::timer::start();
    double sum = af::sum<float>(a);
    printf("elapsed seconds: %g\n", af::timer::stop(t));

    double sum2 = 0;
    t = af::timer::start();
    for(int i = 0;i < 1000000;i++)
    {
        sum2 = sum2 + a(i);
    }
    printf("elapsed seconds: %g\n", af::timer::stop(t));


    printf("sum: %g\n", sum);
    printf("sum2: %g\n", sum2);

    return 0;
}