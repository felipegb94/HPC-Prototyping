double* gridToArray(std::vector< std::vector<double> > &gridVector);

void printGrid(double grid[NROWS][NCOLS]);
void printGrid(std::vector< std::vector<double> > &grid);
void printGrid(double *grid, int numRows, int numCols);

void printToFile(double grid[NROWS][NCOLS], int step);
void printToFile(std::vector< std::vector<double> > &grid, int step);
void printToFile(double *grid, int numRows, int numCols, int step);





