#include <iostream> /* cout, endl*/
#include <fstream> /* ofstream*/
#include <sstream> /* stringstream */

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

void printToFile(double grid[NROWS][NCOLS], int step)
{
    std::ofstream myFile;
    std::stringstream filename;
    filename << "temperature."  << "csv." << step;
    myFile.open(filename.str());

    myFile << "x,y,z,T\n";
    for(int i = 0;i < NROWS;i++)
    {
        for(int j = 0;j < NCOLS;j++)
        {
            if(grid[i][j] < 0.0000001){
                myFile << i << "," << j << "," << 0 << "," << 0 << "\n";
            }
            else
            {
                myFile << i << "," << j << "," << 0 << "," << grid[i][j] << "\n";
            }
        }
    }  

    myFile.close();

}

