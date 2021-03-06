#include <stdio.h> /* fprintf() */
#include <iostream> 
#include "pi-kernel.h"

/************************
 * FUNCTION DEFINITIONS *
 ************************/
int main(int argc, char **argv) 
{
    long numRects = 5e8;
    std::cout << "NumRects being used = " << numRects << std::endl;

    double area = 0.0;

    if(CUDA_ENABLED){
        printf("CUDA is enabled\n");
        printDeviceInfo();
    }
    else if(OPENMP_ENABLED){
        printf("OMP is enabled..\n");     
    }
    else{
        printf("Neither CUDA or OMP are enabled. Running serially\n");
    }

    calculateArea(numRects, &area);
    std::cout << "Pi = " << 4*area << std::endl;

  return 0;
}
