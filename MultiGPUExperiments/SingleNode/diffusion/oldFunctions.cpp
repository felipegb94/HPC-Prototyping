void initGrid(double grid[NROWS][NCOLS]);


/* This was an essential step when using C's 2D arrays! */
void initGrid(double grid[NROWS][NCOLS])
{
    for(int i = 0;i < NROWS;i++)
    {
        for(int j = 0;j < NCOLS;j++)
        {
            grid[i][j] = 0;
            if(j == 0)
            {
                grid[i][j] = 50.0;
            }
            if(i == 0)
            {
                grid[i][j] = 50.0;
            }
        }
    } 

}