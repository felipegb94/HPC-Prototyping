#include "main.decl.h" 
#include "main.h"

// Entry point of Charm++ application
Main::Main(CkArgMsg* msg) 
{
    // Print a message for the user
    CkPrintf("Hello World!\n");
    // Exit the application --> This is needed. It does not work like normal C/C++ that just returning will do the job.
    CkExit();
}

// Constructor needed for chare object migration (ignore
// for now) NOTE: This constructor does not need to
// appear in the ".ci" file
Main::Main(CkMigrateMessage* msg) { }

#include "main.def.h" 