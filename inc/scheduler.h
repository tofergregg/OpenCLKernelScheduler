#ifndef _GAUSSIANELIM
#define _GAUSSIANELIM

#include <iostream>
#include <vector>
#include <float.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "types.h"
#include "clutils.h"

// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif 

enum { GLOBAL_ARG,LOCAL_ARG };

float *OpenClGaussianElimination(
	cl_context context,
	int timing);

void printUsage();
int parseCommandline(int argc, char *argv[], char* filename,
                     int *q, int *t, int *p, int *d);
                     
float eventTime(cl_event event,cl_command_queue command_queue);
void setUpScheduler(cl_context *context,cl_command_queue *command_queue,
                    cl_kernel *scheduler_kernel, 
                    cl_kernel *setArg_kernel, cl_kernel *setArgGlobalUint_kernel,
                    cl_kernel *setArgGlobalFloat_kernel,
                    cl_kernel *setArgLocal_kernel);

int setArg(cl_command_queue command_queue,
            cl_mem taskGPU,
            int taskNum,
            int argIndex,
            cl_kernel setArg_kernel,
            unsigned int argSize,
            void *argPtr);
            
void runKernelScheduler(cl_command_queue command_queue,
                        cl_kernel scheduler_kernel,
                        cl_context context,
                        cl_mem taskGPU,
                        int totalBlocks);
#endif
