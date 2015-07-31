#include <iostream>

#include "utils.h" /* For printGrid */


void initGrid(double grid[NROWS][NCOLS]);


int main(int argc, char *argv[])
{
    int i, j;
    // if(argc != 3)
    // {
    //     fprintf(stderr, "Incorrect number of arguments. Usage: ./diffusion numRows numCols\n");
    //     return 1;
    // }
    // rows = atoi(argv[1]);
    // cols = atoi(argv[2]);

    double grid[NROWS][NCOLS];
    initGrid(grid);
    printGrid(grid);
    return 0;
}

void initGrid(double grid[NROWS][NCOLS])
{
    /* Print grid */
    for(int i = 0;i < NROWS;i++)
    {
        for(int j = 0;j < NCOLS;j++)
        {
            grid[i][j] = 1;
        }
    } 
}


