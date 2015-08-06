/* Pi - CUDA version
 * Author: Aaron Weeden, Shodor, May 2015
 * Modified By: Felipe Gutierrez
 *
 * Approximate pi using a Left Riemann Sum under a quarter unit circle.
 *
 * When running the program, the number of rectangles can be passed using the
 * -r option, e.g. 'pi-cuda-1 -r X', where X is the number of rectangles.
 */

/*************
 * LIBRARIES *
 *************/

/* Pi - CUDA version 1 - uses integers for CUDA kernels
 * Author: Aaron Weeden, Shodor, May 2015
 */
#include <stdio.h> /* fprintf() */
#include <iostream> 

#include "pi-kernel.cuh"

/************************
 * FUNCTION DEFINITIONS *
 ************************/
int main(int argc, char **argv) 
{
    long numRects = 1e9;
    double area = 0.0;
    std::cout << "NumRects being used = " << numRects << std::endl;

    if(CUDA_ENABLED){
        printf("CUDA is enabled\n");
    }
    else{
        printf("Cuda is not enabled\n");
    }

    calculateArea(numRects, &area);
    std::cout << "Pi = " << 4*area << std::endl;

  return 0;
}




