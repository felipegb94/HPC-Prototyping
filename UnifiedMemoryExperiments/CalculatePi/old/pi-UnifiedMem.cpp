#include <stdio.h> /* fprintf() */
#include <iostream> 

#include "pi-kernel.cuh"

/************************
 * FUNCTION DEFINITIONS *
 ************************/
int main(int argc, char **argv) 
{
    long numRects = 1e9;
    std::cout << "NumRects being used = " << numRects << std::endl;

    double area = 0.0;

    if(CUDA_ENABLED){
        printf("CUDA is enabled\n");
    }
    else if(OPENMP_ENABLED){
        printf("OMP is enabled\n");     
    }
    else{
        printf("Neitheer CUDA or OMP are enabled. Running serially\n");
    }

    calculateArea(numRects, &area);

    std::cout << "Pi = " << 4*area << std::endl;

  return 0;
}
