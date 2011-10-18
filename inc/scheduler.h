#ifndef _GAUSSIANELIM
#define _GAUSSIANELIM

#include <iostream>
#include <vector>
#include <float.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#include "clutils.h"

// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif 

float *OpenClGaussianElimination(
	cl_context context,
	int timing);

void printUsage();
int parseCommandline(int argc, char *argv[], char* filename,
                     int *q, int *t, int *p, int *d);
                     
float eventTime(cl_event event,cl_command_queue command_queue);
void runScheduler(cl_context context, int timing);
#endif
