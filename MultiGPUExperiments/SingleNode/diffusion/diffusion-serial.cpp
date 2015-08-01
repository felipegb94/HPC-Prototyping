#include <iostream>

#include "utils.h" /* For printGrid */


void initGrid(  double grid[NROWS][NCOLS]);

void calcGrid(  double currGrid[NROWS][NCOLS], 
                double nextGrid[NROWS][NCOLS], 
                int startRow, 
                int startCol,
                int endRow,
                int endCol);

void updateGrid(double currGrid[NROWS][NCOLS], 
                double nextGrid[NROWS][NCOLS], 
                int startRow, 
                int startCol,
                int endRow,
                int endCol);

void diffuse(   double currGrid[NROWS][NCOLS], 
                double nextGrid[NROWS][NCOLS], 
                int startRow, 
                int startCol,
                int endRow,
                int endCol);


int main(int argc, char *argv[])
{
    int i, j, numSteps, startRow, startCol, endRow, endCol;
    double currGrid[NROWS][NCOLS];
    double nextGrid[NROWS][NCOLS];

    numSteps = 1e2;
    startRow = 1;
    startCol = 1;
    endRow = NROWS ;
    endCol = NCOLS ;

    initGrid(nextGrid);
    initGrid(currGrid);
    printGrid(currGrid);


    for(i = 0;i < 2;i++)
    {
        std::cout << "------------------Step " << i << "-----------------------" << std::endl;
        diffuse(currGrid, nextGrid, startRow, startCol, endRow, endCol);
        //printGrid(currGrid);
    }


    return 0;
}

void initGrid(double grid[NROWS][NCOLS])
{
    for(int i = 0;i < NROWS;i++)
    {
        for(int j = 0;j < NCOLS;j++)
        {
            grid[i][j] = 0;
            if(j == 0)
            {
                grid[i][j] = 25.0;
            }
        }
    } 

}

void calcGrid(double currGrid[NROWS][NCOLS], 
              double nextGrid[NROWS][NCOLS], 
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
                //if(currGrid[i-1][j] > 0.0000000000001)
                {
                    toAdd = toAdd + currGrid[i-1][j];
                }
                toDivide = toDivide + 1;
            }
            /* DOWN */
            if(i != NROWS-1)
            {
                //if(currGrid[i+1][j] > 0.0000000000001)
                {
                    toAdd = toAdd + currGrid[i+1][j];
                }
                toDivide = toDivide + 1;
            }
            /* LEFT */
            if(j != 0)
            {
                //if(currGrid[i][j-1] > 0.0000000000001)
                {
                    toAdd = toAdd + currGrid[i][j-1];
                }
                toDivide = toDivide + 1;
            }
            /* RIGHT */
            if(j != NCOLS-1)
            {
                //if(currGrid[i][j+1] > 0.0000000000001)
                {
                    toAdd = toAdd + currGrid[i][j+1];
                }
                toDivide = toDivide + 1;
            }
            /* TOP LEFT */
            if(i != 0 && j != 0)
            {
                //if(currGrid[i-1][j-1] > 0.0000000000001)
                {
                    toAdd = toAdd + currGrid[i-1][j-1];
                }
                toDivide = toDivide + 1;
            }
            /* TOP RIGHT */
            if(i != 0 && j != NCOLS-1)
            {
                //if(currGrid[i-1][j+1] > 0.0000000000001)
                {
                    toAdd = toAdd + currGrid[i-1][j+1];
                }
                toDivide = toDivide + 1;

            }
            /* BOTTOM LEFT */
            if(i != NROWS-1 && j != 0)
            {
                //if(currGrid[i+1][j-1] > 0.0000000000001)
                {
                    toAdd = toAdd + currGrid[i+1][j-1];
                }
                toDivide = toDivide + 1;    
            }
            /* BOTTOM RIGHT */
            if(i != NROWS-1 && j != NCOLS-1)
            {
                //if(currGrid[i+1][j+1] > 0.0000000000001)
                {
                    toAdd = toAdd + currGrid[i+1][j+1];
                }
                toDivide = toDivide + 1;   
            }

            // std::cout << "i = " << i << ", j = " << j << std::endl;
            // std::cout << "toAdd = " << toAdd << std::endl;
            // std::cout << "toDivide = " << toDivide << std::endl;
            if(toAdd > 0.0000000000001)
            {
                nextGrid[i][j] = toAdd/toDivide;
            }
            if(nextGrid[i][j] > 100)
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

void updateGrid(double currGrid[NROWS][NCOLS], 
                double nextGrid[NROWS][NCOLS], 
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


void diffuse(double currGrid[NROWS][NCOLS], 
             double nextGrid[NROWS][NCOLS], 
             int startRow, 
             int startCol,
             int endRow,
             int endCol)
{
    calcGrid(currGrid, nextGrid, startRow, startCol, endRow, endCol);
    updateGrid(currGrid, nextGrid, startRow, startCol, endRow, endCol);
}
















