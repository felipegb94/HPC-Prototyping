#include <iostream> /* cout, endl*/
#include <fstream> /* ofstream*/
#include <sstream> /* stringstream */
#include <vector>
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

void printGrid(std::vector< std::vector<double> > &grid)
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
    filename << "results/temperature.csv." << step;
    myFile.open(filename.str());

    myFile << "x,y,T\n";
    for(int i = 0;i < NROWS;i++)
    {
        for(int j = 0;j < NCOLS;j++)
        {
            myFile << i << "," << j << "," << grid[i][j] << "\n";
        }
    }  

    myFile.close();

}

void printToFile(std::vector< std::vector<double> > &grid, int step)
{
    std::ofstream myFile;
    std::stringstream filename;
    filename << "results/temperature.csv." << step;
    myFile.open(filename.str());

    myFile << "x,y,T\n";
    for(int i = 0;i < NROWS;i++)
    {
        for(int j = 0;j < NCOLS;j++)
        {
            myFile << i << "," << j << "," << grid[i][j] << "\n";
        }
    }  

    myFile.close();

}

