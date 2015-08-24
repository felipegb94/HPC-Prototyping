void setupMPI(int *argc, char ***argv, int *myRank, int *numProcs);

void distributeWork(const int numRects, const int myRank, const int numProcs,
    int *myNumRects, int *myDispl);

void calculateHeight(const int i, const double width, double *height);
void calculateArea(const int numRects, const int myNumRects, const double width,
    const int myDispl, double *area);
