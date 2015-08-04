void calcGrid(  std::vector< std::vector<double> > &currGrid, 
                std::vector< std::vector<double> > &nextGrid, 
                int startRow, 
                int startCol,
                int endRow,
                int endCol);

void updateGrid(std::vector< std::vector<double> > &currGrid, 
                std::vector< std::vector<double> > &nextGrid, 
                int startRow, 
                int startCol,
                int endRow,
                int endCol);

void diffuse(   std::vector< std::vector<double> > &currGrid, 
                std::vector< std::vector<double> > &nextGrid, 
                int startRow, 
                int startCol,
                int endRow,
                int endCol);