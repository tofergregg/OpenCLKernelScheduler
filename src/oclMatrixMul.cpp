/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication with multi GPU support.
 * It has been written for clarity of exposition to illustrate various OpenCL
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * CUBLAS provides high-performance matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Superconducting (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11. 
 *
 */

// project include
#include "matrixMul.h"
#include "scheduler.h"

enum shrBOOL
{
    shrFALSE = 0,
    shrTRUE = 1
};

// max GPU's to manage for multi-GPU parallel compute
const unsigned int MAX_GPU_COUNT = 1;

// Globals for size of matrices
unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
int iSizeMultiple = 1;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
int runTest(int argc, const char** argv);
void matrixMulGPU(cl_uint ciDeviceCount, cl_mem h_A, float* h_B_data, unsigned int mem_size_B, float* h_C,
                    cl_context cxGPUContext,cl_command_queue *commandQueue);


void matrixMulGPU(cl_uint ciDeviceCount, cl_mem h_A, float* h_B_data, unsigned int mem_size_B, float* h_C,
                    cl_context cxGPUContext,cl_command_queue *commandQueue)
{
    cl_mem d_A[MAX_GPU_COUNT];
    cl_mem d_C[MAX_GPU_COUNT];
    cl_mem d_B[MAX_GPU_COUNT];

    cl_event GPUDone[MAX_GPU_COUNT];
    cl_event GPUExecution[MAX_GPU_COUNT];

    // Start the computation on each available GPU
    
    // Create buffers for each GPU
    // Each GPU will compute sizePerGPU rows of the result
    int sizePerGPU = uiHA / ciDeviceCount;

    int workOffset[MAX_GPU_COUNT];
    int workSize[MAX_GPU_COUNT];

    workOffset[0] = 0;
    for(unsigned int i=0; i < ciDeviceCount; ++i) 
    {
        // Input buffer
        workSize[i] = (i != (ciDeviceCount - 1)) ? sizePerGPU : (uiHA - workOffset[i]);        

        d_A[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, workSize[i] * sizeof(float) * uiWA, NULL,NULL);

        // Copy only assigned rows from host to device
        clEnqueueCopyBuffer(commandQueue[i], h_A, d_A[i], workOffset[i] * sizeof(float) * uiWA, 
                            0, workSize[i] * sizeof(float) * uiWA, 0, NULL, NULL);        
        
        // create OpenCL buffer on device that will be initiatlize from the host memory on first use
        // on device
        d_B[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                mem_size_B, h_B_data, NULL);

        // Output buffer
        d_C[i] = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY,  workSize[i] * uiWC * sizeof(float), NULL,NULL);
//               
//         // set the args values
//         clSetKernelArg(multiplicationKernel[i], 0, sizeof(cl_mem), (void *) &d_C[i]);
//         clSetKernelArg(multiplicationKernel[i], 1, sizeof(cl_mem), (void *) &d_A[i]);
//         clSetKernelArg(multiplicationKernel[i], 2, sizeof(cl_mem), (void *) &d_B[i]);
//         clSetKernelArg(multiplicationKernel[i], 3, sizeof(float) * BLOCK_SIZE *BLOCK_SIZE, 0 );
//         clSetKernelArg(multiplicationKernel[i], 4, sizeof(float) * BLOCK_SIZE *BLOCK_SIZE, 0 );
//         clSetKernelArg(multiplicationKernel[i], 5, sizeof(cl_int), (void *) &uiWA);
//         clSetKernelArg(multiplicationKernel[i], 6, sizeof(cl_int), (void *) &uiWB);
// 
//         if(i+1 < ciDeviceCount)
//             workOffset[i + 1] = workOffset[i] + workSize[i];
    }
    
    // Execute Multiplication on all GPUs in parallel
    size_t localWorkSize[] = {BLOCK_SIZE, BLOCK_SIZE};
    size_t globalWorkSize[] = {shrRoundUp(BLOCK_SIZE, uiWC), shrRoundUp(BLOCK_SIZE, workSize[0])};
//     
//     // Launch kernels on devices
// 
//         for(unsigned int i = 0; i < ciDeviceCount; i++) 
//         {
// 			// Multiplication - non-blocking execution:  launch and push to device(s)
// 			globalWorkSize[1] = shrRoundUp(BLOCK_SIZE, workSize[i]);
// 			clEnqueueNDRangeKernel(commandQueue[i], multiplicationKernel[i], 2, 0, globalWorkSize, localWorkSize,
// 				                   0, NULL, &GPUExecution[i]);
//             clFlush(commandQueue[i]);
// 		}


    // sync all queues to host
	for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {
		clFinish(commandQueue[i]);
	}


    for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {    
        // Non-blocking copy of result from device to host
        clEnqueueReadBuffer(commandQueue[i], d_C[i], CL_FALSE, 0, uiWC * sizeof(float) * workSize[i], 
                            h_C + workOffset[i] * uiWC, 0, NULL, &GPUDone[i]);
    }

	// CPU sync with GPU
    clWaitForEvents(ciDeviceCount, GPUDone);


    // Release mem and event objects    
    for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {
        clReleaseMemObject(d_A[i]);
        clReleaseMemObject(d_C[i]);
        clReleaseMemObject(d_B[i]);
	    clReleaseEvent(GPUExecution[i]);
	    clReleaseEvent(GPUDone[i]);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for 
////////////////////////////////////////////////////////////////////////////////
int runTest(int argc, const char** argv, cl_context cxGPUContext, cl_command_queue *commandQueue)
{
    cl_uint ciDeviceCount = 0;
    cl_int ciErrNum = CL_SUCCESS;

    // allocate host memory for matrices A and B
    unsigned int size_A = uiWA * uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A_data = (float*)malloc(mem_size_A);
    unsigned int size_B = uiWB * uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B_data = (float*)malloc(mem_size_B);

    // initialize host memory
    srand(2006);
    shrFillArray(h_A_data, size_A);
    shrFillArray(h_B_data, size_B);

    // allocate host memory for result
    unsigned int size_C = uiWC * uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* h_C = (float*) malloc(mem_size_C);

    // create OpenCL buffer pointing to the host memory
    cl_mem h_A = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
				    mem_size_A, h_A_data, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clCreateBuffer\n");
        return ciErrNum;
    }
        
    // Run multiplication on 1..deviceCount GPUs to compare improvement
    printf("\nRunning Computations on 1 - %d GPU's...\n\n", ciDeviceCount);
    for(unsigned int k = 1; k <= ciDeviceCount; ++k) 
    {
        matrixMulGPU(k, h_A, h_B_data, mem_size_B, h_C,cxGPUContext,commandQueue);
    }

    // clean up OCL resources
    ciErrNum = clReleaseMemObject(h_A);

    if(ciErrNum != CL_SUCCESS)
    {
        printf("Error: Failure releasing OpenCL resources: %d\n", ciErrNum);
        return ciErrNum;
    }

    // clean up memory
    free(h_A_data);
    free(h_B_data);
    free(h_C);
    
    return CL_SUCCESS;
}

// Round Up Division function
size_t shrRoundUp(int group_size, int global_size) 
{
    int r = global_size % group_size;
    if(r == 0) 
    {
        return global_size;
    } else 
    {
        return global_size + group_size - r;
    }
}

// Helper function to init data arrays 
// *********************************************************************
void shrFillArray(float* pfData, int iSize)
{
    int i; 
    const float fScale = 1.0f / (float)RAND_MAX;
    for (i = 0; i < iSize; ++i) 
    {
        pfData[i] = fScale * rand();
    }
}
