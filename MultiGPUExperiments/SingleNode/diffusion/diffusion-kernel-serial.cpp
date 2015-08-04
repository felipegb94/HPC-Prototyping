#include <iostream> /* cout , endl */
#include <vector> /* vector */


#include "diffusion-kernel.h"
#include "utils.h"

void calcGrid(std::vector< std::vector<double> > &currGrid, 
              std::vector< std::vector<double> > &nextGrid, 
              int startRow, 
              int startCol,
              int endRow, 
              int endCol)
{
    for(int i = startRow; i < endRow;i++)
    {
        for(int j = startCol;j < endCol;j++)
        {
            double toAdd = 0;
            double toDivide = 0;
            /* UP */
            if(i != 0)
            {
                toAdd = toAdd + currGrid[i-1][j];
                toDivide = toDivide + 1;
            }
            /* DOWN */
            if(i != NROWS-1)
            {
                toAdd = toAdd + currGrid[i+1][j];
                toDivide = toDivide + 1;
            }
            /* LEFT */
            if(j != 0)
            {
                toAdd = toAdd + currGrid[i][j-1];
                toDivide = toDivide + 1;
            }
            /* RIGHT */
            if(j != NCOLS-1)
            {
                toAdd = toAdd + currGrid[i][j+1];
                toDivide = toDivide + 1;
            }
            /* TOP LEFT */
            if(i != 0 && j != 0)
            {
                toAdd = toAdd + currGrid[i-1][j-1];
                toDivide = toDivide + 1;
            }
            /* TOP RIGHT */
            if(i != 0 && j != NCOLS-1)
            {
                toAdd = toAdd + currGrid[i-1][j+1];
                toDivide = toDivide + 1;

            }
            /* BOTTOM LEFT */
            if(i != NROWS-1 && j != 0)
            {
                toAdd = toAdd + currGrid[i+1][j-1];
                toDivide = toDivide + 1;    
            }
            /* BOTTOM RIGHT */
            if(i != NROWS-1 && j != NCOLS-1)
            {
                toAdd = toAdd + currGrid[i+1][j+1];
                toDivide = toDivide + 1;   
            }

            nextGrid[i][j] = toAdd/toDivide;
            if(nextGrid[i][j] > 100000)
            {
                double sum = 0;
                sum = sum + nextGrid[i-1][j]    ;
                sum = sum + nextGrid[i][j-1]    ;
                sum = sum + nextGrid[i-1][j-1]  ;
                sum = sum + nextGrid[i+1][j]    ;
                sum = sum + nextGrid[i][j+1]    ;
                sum = sum + nextGrid[i+1][j+1]  ;
                sum = sum + nextGrid[i+1][j-1]  ;
                sum = sum + nextGrid[i-1][j+1]  ;

                std::cout << "i = " << i << ", j = " << j << std::endl;
                std::cout << nextGrid[i][j]      << std::endl;
                std::cout << nextGrid[i-1][j]    << std::endl;
                std::cout << nextGrid[i][j-1]    << std::endl;
                std::cout << nextGrid[i-1][j-1]  << std::endl;
                std::cout << nextGrid[i+1][j]    << std::endl;
                std::cout << nextGrid[i][j+1]    << std::endl;
                std::cout << nextGrid[i+1][j+1]  << std::endl;
                std::cout << nextGrid[i+1][j-1]  << std::endl;
                std::cout << nextGrid[i-1][j+1]  << std::endl;

                std::cout << "sum = " << sum << std::endl;
                std::cout << "result = " << sum/toDivide << std::endl;

                std::cout << "toAdd = " << toAdd << std::endl;
                std::cout << "toDivide = " << toDivide << std::endl;  
                throw 20;
            }
        }
    } 
}

void updateGrid(std::vector< std::vector<double> > &currGrid, 
                std::vector< std::vector<double> > &nextGrid, 
                int startRow, 
                int startCol,
                int endRow,
                int endCol)
{
    for(int i = startRow; i < endRow;i++)
    {
        for(int j = startCol;j < endCol;j++)
        {
            currGrid[i][j] = nextGrid[i][j];
        }
    } 
}


void diffuse(std::vector< std::vector<double> > &currGrid, 
             std::vector< std::vector<double> > &nextGrid, 
             int startRow, 
             int startCol,
             int endRow,
             int endCol,
             int currStep)
{
    int numRows = currGrid.size();
    int numCols = currGrid[0].size();
    calcGrid(currGrid, nextGrid, startRow, startCol, endRow, endCol);
    updateGrid(currGrid, nextGrid, startRow, startCol, endRow, endCol);

    if(currStep%10 == 0)
    {
        printToFile(currGrid, currStep);
    }
}

