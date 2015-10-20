#include <stdio.h>
#include <mpi.h>

#define TAG 5
#define MASTER 0
int main(int argc, char **argv)
{
	
	int myRank;
	int numProcs;

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	
	if(myRank == 0)
	{
		int theirRank;
		
		for (int i = 1; i < numProcs; i++)
		//int counter = 1;
		//while(counter < numProcs) 
		{
 			MPI_Recv(&theirRank, 
				 1, 
				 MPI_INT, 
				 MPI_ANY_SOURCE, 
				 TAG,
				 MPI_COMM_WORLD,
				 MPI_STATUS_IGNORE);
			//counter++;
			printf("helloworld from: %d \n", theirRank);
    		}
	}
	else
	{
		MPI_Send(&myRank, 1, MPI_INT, MASTER, TAG, MPI_COMM_WORLD);
	}


	MPI_Finalize();
	return 0;
}
