#include <iostream>

#include "utils.h"

void printGrid(double grid[NROWS][NCOLS])
{
    /* Print grid */
    for(int i = 0;i < NROWS;i++)
    {
        for(int j = 0;j < NCOLS;j++)
        {
            std::cout << grid[i][j] << "    ";
        }
        std::cout << std::endl;
    }   
}

