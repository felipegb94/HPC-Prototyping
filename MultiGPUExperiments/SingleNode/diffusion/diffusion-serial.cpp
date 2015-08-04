#include <iostream>
#include <vector>

#include "utils.h" /* For printGrid and printToFile */
#include "diffusion-kernel.h" /* For diffuse */


int main(int argc, char *argv[])
{
    int i, j, numSteps, startRow, startCol, endRow, endCol;
    double initialTemp;

    /* Initialization step */
    std::vector< std::vector<double> > currGrid(NROWS, std::vector<double>(NCOLS));
    std::vector< std::vector<double> > nextGrid(NROWS, std::vector<double>(NCOLS));
    /* Initial conditions */
    initialTemp = 100;
    currGrid[0].assign(NCOLS, initialTemp);
    currGrid[NROWS-1].assign(NCOLS, initialTemp);
    nextGrid[0].assign(NCOLS, initialTemp);
    nextGrid[NROWS-1].assign(NCOLS, initialTemp);  


    printGrid(currGrid);

    numSteps = 3;
    startRow = 1;
    startCol = 0;
    endRow = NROWS-1 ;
    endCol = NCOLS ;

    for(i = 0;i < numSteps;i++)
    {
        std::cout << "------------------Step " << i << "-----------------------" << std::endl;
        diffuse(currGrid, nextGrid, startRow, startCol, endRow, endCol, i);

    }


    return 0;
}

















