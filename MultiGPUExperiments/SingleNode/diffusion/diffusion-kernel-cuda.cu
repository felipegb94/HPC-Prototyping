#include <vector>

#include "diffusion-kernel.h"

#include "utils.h"



void diffuse(   std::vector< std::vector<double> > &currGridVector, 
                std::vector< std::vector<double> > &nextGridVector, 
                int startRow, 
                int startCol,
                int endRow,
                int endCol,
                int currStep)
{
    double *currGrid;
    double *nextGrid;

    int numRows = currGridVector.size();
    int numCols = currGridVector[0].size();

    currGrid = gridToArray(currGridVector);
    nextGrid = gridToArray(nextGridVector);

    if(currStep%10 == 0)
    {
        printToFile(currGrid, numRows, numCols, currStep);
    }

}