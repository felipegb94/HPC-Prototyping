#include <iostream> /* cout, endl*/
#include <fstream> /* ofstream*/
#include <sstream> /* stringstream */
#include <vector>
#include <stdlib.h> /* fprintf() */
#include "utils.h"

double* gridToArray(std::vector< std::vector<double> > &gridVector)
{
    int numRows = gridVector.size();
    int numCols = gridVector[0].size();

    double *grid = (double*)malloc(numRows*numCols * sizeof(double));

    for(int i = 0;i < numRows;i++)
    {
        std::vector<double> currRow = gridVector[i];
        for(int j = 0;j < numCols;j++)
        {
            grid[i*numCols + j] = currRow[j];
        }
    }

    return grid;

}

void printGrid(double grid[NROWS][NCOLS])
{
    /* Print grid */
    for(int i = 0;i < NCOLS;i++)
    {
        for(int j = 0;j < NROWS;j++)
        {
            std::cout << grid[i][j] << "    ";
        }
        std::cout << std::endl;
    }   
}


void printGrid(std::vector< std::vector<double> > &grid)
{
    int numRows = grid.size();
    int numCols = grid[0].size();
    /* Print grid */
    for(int i = 0;i < numRows;i++)
    {
        std::vector<double> currRow = grid[i];

        for(int j = 0;j < numCols;j++)
        {
            std::cout << currRow[j] << "    ";
        }
        std::cout << std::endl;
    }   
}

void printGrid(double *grid, int numRows, int numCols)
{
    std::cout << "Array grid" << std::endl;
    for(int i = 0;i < numRows;i++)
    {
        for(int j = 0;j < numCols;j++)
        {
            std::cout << grid[i*numCols + j] << "      " ;
        }
        std::cout << " "<< std::endl;
    }
}

void printToFile(double grid[NROWS][NCOLS], int step)
{
    std::ofstream myFile;
    std::stringstream filename;
    filename << "results/temperature.csv." << step;
    myFile.open(filename.str().c_str());

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
    int numRows = grid.size();
    int numCols = grid[0].size();
    std::ofstream myFile;
    std::stringstream filename;
    filename << "results/temperature.csv." << step;
    myFile.open(filename.str().c_str());

    myFile << "x,y,T\n";
    for(int i = 0;i < numRows;i++)
    {
        for(int j = 0;j < numCols;j++)
        {
            myFile << i << "," << j << "," << grid[i][j] << "\n";
        }
    }  

    myFile.close();
}


void printToFile(double *grid, int numRows, int numCols, int step)
{
    std::ofstream myFile;
    std::stringstream filename;
    filename << "results/temperature.csv." << step;
    myFile.open(filename.str().c_str());

    myFile << "x,y,T\n";
    for(int i = 0;i < numRows;i++)
    {
        for(int j = 0;j < numCols;j++)
        {
            myFile << i << "," << j << "," << grid[i*numCols + j] << "\n";
        }
    }  

    myFile.close();
}




