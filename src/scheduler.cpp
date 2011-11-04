#ifndef __SCHEDULER__
#define __SCHEDULER__

#define STRINGIFY(s) STRINGIFY_HELPER(s)
#define STRINGIFY_HELPER(s) #s

#include "scheduler.h"
#include "types.h"

// g++ on linux #defines linux to 1. Since we don't use that anywhere, but do
// need "linux" as part of a path, we can undefine it here.

#ifdef linux
#undef linux
#endif

//cl_context context=NULL;
//cl_command_queue command_queue = NULL;

// globals
size_t globalWorksize = 8192;
size_t localWorksize = 256;

int mainOLD(int argc, char *argv[]) {
        
    // args
    char filename[100];
    int quiet=0,timing=0,platform=-1,device=-1;
    
    // parse command line
    if (parseCommandline(argc, argv, filename,
                     &quiet, &timing, &platform, &device)) {
    printUsage();
    return 0;
    }
    
    //for(i=0;i<numRecords;i++)
    //    printf("%s, %f, %f\n",(records[i].recString),locations[i].lat,locations[i].lng);
    

    //begin timing	
    
    // run scheduler
    cl_context context=NULL;
    cl_command_queue command_queue = NULL;
    cl_kernel scheduler_kernel, setArg_kernel, setArgGlobalUint_kernel,setArgGlobalFloat_kernel;
    cl_kernel setArgLocal_kernel;
    setUpScheduler(&context, &command_queue, &scheduler_kernel, 
                   &setArg_kernel,&setArgGlobalUint_kernel,&setArgGlobalFloat_kernel,
                   &setArgLocal_kernel);
    
    //end timing
//     if (!quiet) {
//         printf("The result of matrix m is: \n");
//         
//         PrintMat(m, size, size, size);
//         printf("The result of matrix a is: \n");
//         PrintMat(a, size, size, size);
//         printf("The result of array b is: \n");
//         PrintAry(b, size);
//         
//         BackSub(a,b,finalVec,size);
//         printf("The final solution is: \n");
//         PrintAry(finalVec,size);
//     }

  return 0;
}

/*------------------------------------------------------
 ** runScheduler() -- run the scheduler
 **------------------------------------------------------
 */
void setUpScheduler(cl_context *context,cl_command_queue *command_queue,
                    cl_kernel *scheduler_kernel, cl_kernel *setArg_kernel,
                    cl_kernel *setArgGlobalUint_kernel, cl_kernel *setArgGlobalFloat_kernel,
                    cl_kernel *setArgLocal_kernel){
    // 1. set up kernels
    cl_int status=0;
    cl_program scheduler_program;
    cl_event writeEvent,kernelEvent,readEvent;
    float writeTime=0, kernelTime=0;
    float readTime=0;
    //float writeMB=0,readMB=0;
    
    int quiet=0,platform=-1,device=-1;
    *context = cl_init_context(platform,device,quiet);

    scheduler_program = cl_CompileProgram(
        (char *)"scheduler.cl",
        (char *)"-I " STRINGIFY(RUNLOC) " -I " STRINGIFY(INCLOC)
        );

    *scheduler_kernel = clCreateKernel(
        scheduler_program, "scheduler", &status);
    status = cl_errChk(status, (char *)"Error Creating scheduler kernel");
    if(status)exit(1);
    
    *setArg_kernel = clCreateKernel(
        scheduler_program, "setArg", &status);
    status = cl_errChk(status, (char *)"Error Creating setArg kernel");
    if(status)exit(1);
    
    *setArgGlobalUint_kernel = clCreateKernel(
        scheduler_program, "setArgGlobalUint", &status);
    status = cl_errChk(status, (char *)"Error Creating setArg kernel");
    if(status)exit(1);
    
    *setArgGlobalFloat_kernel = clCreateKernel(
        scheduler_program, "setArgGlobalFloat", &status);
    status = cl_errChk(status, (char *)"Error Creating setArg kernel");
    if(status)exit(1);   
    
    *setArgLocal_kernel = clCreateKernel(
        scheduler_program, "setArgLocal", &status);
    status = cl_errChk(status, (char *)"Error Creating setArg Local kernel");
    if(status)exit(1);    
        
//     logInfoGPU = clCreateBuffer(*context, CL_MEM_READ_WRITE,
//         sizeof(unsigned int) * globalWorksize * 2, NULL, &status);    
//     status = cl_errChk(status, (char *)"Error allocating logInfoGPU buffer");
//     if(status)exit(1);

    *command_queue = cl_getCommandQueue();
    
//     Task task;
//     task.xDim = 3;
//     task.yDim = 2;
//     task.workgroupsLeft = task.xDim * task.yDim;
//     task.xThreads = 2;
//     task.yThreads = 3;
//     task.kernelId = 0;    
//     
//     status = clEnqueueWriteBuffer(command_queue,
//                taskGPU,
//                1, // change to 0 for nonblocking write
//                0, // offset
//                sizeof(Task),
//                &task,
//                0,
//                NULL,
//                &writeEvent);
//     
//     if (timing) writeTime+=eventTime(writeEvent,command_queue);
//     clReleaseEvent(writeEvent);
//     
//     // setArgs
//     
//     // set the arguments for setArgs
//     setArg(command_queue,
//             taskGPU,
//             0, // taskNum
//             0, // argIndex
//             setArg_kernel,
//             sizeof(cl_mem), // argSize
//             (void *)&logInfoGPU);
//     
//     // end setArgs
//     

//     
//     unsigned int *logInfoCPU = (unsigned int *)malloc(sizeof(unsigned int)*globalWorksize*2);
//     for (unsigned int i=0;i<globalWorksize*2;i++){
//         logInfoCPU[i]=0xcafebabe;
//     }
//     status = clEnqueueWriteBuffer(command_queue,
//                logInfoGPU,
//                1, // change to 0 for nonblocking write
//                0, // offset
//                sizeof(unsigned int) * globalWorksize * 2,
//                logInfoCPU,
//                0,
//                NULL,
//                &writeEvent);
//     if (timing) writeTime+=eventTime(writeEvent,command_queue);
//     clReleaseEvent(writeEvent);
// 	
// 	unsigned int queueSize = 1;
// 	unsigned int numberOfTasksToExecute = 1;
// 	
// 	// 4. Setup and Run kernels
//         // kernel args
//         cl_int argchk  = clSetKernelArg(scheduler_kernel, 0, sizeof(cl_mem), (void *)&taskGPU);
//         cl_errChk(argchk,"ERROR in Setting Scheduler kernel args 0");
//         argchk  = clSetKernelArg(scheduler_kernel, 1, sizeof(unsigned int), &queueSize);
//         cl_errChk(argchk,"ERROR in Setting Scheduler kernel args 1");
//         argchk  = clSetKernelArg(scheduler_kernel, 2, sizeof(unsigned int), &numberOfTasksToExecute);
//         cl_errChk(argchk,"ERROR in Setting Scheduler kernel args 2");
//         argchk  = clSetKernelArg(scheduler_kernel, 3, sizeof(cl_mem), (void *)&lockGPU);
//         cl_errChk(argchk,"ERROR in Setting Scheduler kernel args 3");
//         argchk  = clSetKernelArg(scheduler_kernel, 4, sizeof(unsigned int)*localWorksize, NULL);
//         cl_errChk(argchk,"ERROR in Setting Scheduler kernel args 4");
//         argchk  = clSetKernelArg(scheduler_kernel, 5, sizeof(cl_mem), (void *)&spoofingGPU);
//         cl_errChk(argchk,"ERROR in Setting Scheduler kernel args 5");
//         //argchk  = clSetKernelArg(scheduler_kernel, 6, sizeof(cl_mem), (void *)&logInfoGPU);
//         //cl_errChk(argchk,"ERROR in Setting Scheduler kernel args 6");
//         
//         // launch kernel
//         status = clEnqueueNDRangeKernel(
//                   command_queue,  scheduler_kernel, 1, 0,
//                   &globalWorksize,&localWorksize,
//                   0, NULL, &kernelEvent);
// 
//         cl_errChk(status,"ERROR in Executing Scheduler Kernel");
//         if (timing) {
//              kernelTime+=eventTime(kernelEvent,command_queue);
//         }
//         clReleaseEvent(kernelEvent);
// 		
// //5. transfer data off of device
//     status = clEnqueueReadBuffer(command_queue,
//         logInfoGPU,
//         1, // change to 0 for nonblocking write
//         0, // offset
//         sizeof(unsigned int) * globalWorksize * 2,
//         logInfoCPU,
//         0,
//         NULL,
//         &readEvent);
// 
//     cl_errChk(status,"ERROR with clEnqueueReadBuffer");
//     if (timing) readTime+=eventTime(readEvent,command_queue);
//     clReleaseEvent(readEvent);
// 
// // print log info
//     for(unsigned int i=0;i<globalWorksize;i++) {
//         unsigned int x,y;
//         x = logInfoCPU[i*2];
//         y = logInfoCPU[i*2+1];
//         if (x != 0xcafebabe || y != 0xcafebabe)
//           printf("index:%d, x:%d, y:%d\n",i,x,y);
//     }

//     if (timing) {
//         printf("Matrix Size\tWrite(s) [size]\t\tKernel(s)\tRead(s)  [size]\t\tTotal(s)\n");
//         printf("%dx%d      \t",size,size);
//         
//         printf("%f [%.2fMB]\t",writeTime,writeMB);
//         
// 
//         printf("%f\t",kernelTime);
//        
// 
//         printf("%f [%.2fMB]\t",readTime,readMB);
//        
//         printf("%f\n\n",writeTime+kernelTime+readTime);
//    }
    
}

float eventTime(cl_event event,cl_command_queue command_queue){
    cl_int error=0;
    cl_ulong eventStart,eventEnd;
    clFinish(command_queue);
    error = clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,
                                    sizeof(cl_ulong),&eventStart,NULL);
    cl_errChk(error,"ERROR in Event Profiling."); 
    error = clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,
                                    sizeof(cl_ulong),&eventEnd,NULL);
    cl_errChk(error,"ERROR in Event Profiling.");

    return (float)((eventEnd-eventStart)/1e9);
}

int setArg(cl_command_queue command_queue,
            cl_mem taskGPU,
            int taskNum,
            int argIndex,
            cl_kernel setArg_kernel,
            unsigned int argSize,
            void *argPtr){

    cl_int argchk = 0;
    argchk  = clSetKernelArg(setArg_kernel, 0, sizeof(cl_mem), (void *)&taskGPU);
        if (argchk) {
        cl_errChk(argchk,"ERROR in Setting SetArg kernel args 0");
        return argchk;
    }
    argchk = clSetKernelArg(setArg_kernel, 1, sizeof(unsigned int), &taskNum);
        if (argchk) {
        cl_errChk(argchk,"ERROR in Setting SetArg kernel args 1");
        return argchk;
    }
    argchk = clSetKernelArg(setArg_kernel, 2, sizeof(unsigned int), &argIndex);
    if (argchk) {
        cl_errChk(argchk,"ERROR in Setting SetArg kernel args 2");
        return argchk;
    }
    if (argPtr == NULL) { // local memory argument
        argchk = clSetKernelArg(setArg_kernel, 3, sizeof(unsigned int), &argSize);
    }
    else {
        argchk = clSetKernelArg(setArg_kernel, 3, argSize, argPtr);
    }

    if (argchk) {
        cl_errChk(argchk,"ERROR in Setting SetArg kernel args 3");
        return argchk;
    }
    
    // launch kernel
    cl_int status = clEnqueueTask(
              command_queue,  setArg_kernel,
              0, NULL, NULL);
    status = cl_errChk(status, (char *)"Error running setArg clEnqueuTask");
    if(status)exit(1);
    return 0;
}

void runKernelScheduler(cl_command_queue command_queue,
                        cl_kernel scheduler_kernel,
                        cl_context context,
                        cl_mem taskGPU,
                        int totalBlocks){
    unsigned lockInit = 0;
    cl_mem lockGPU,spoofingGPU;
    cl_int status=0;
        
    lockGPU = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(unsigned int), NULL, &status);
    status = cl_errChk(status, (char *)"Error allocating lockGPU buffer");
    if(status)exit(1);

    spoofingGPU = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(SpoofedId) * globalWorksize, NULL, &status);
    status = cl_errChk(status, (char *)"Error allocating spoofingGPU buffer");
    if(status)exit(1);
    
    status = clEnqueueWriteBuffer(command_queue,
               lockGPU,
               1, // change to 0 for nonblocking write
               0, // offset
               sizeof(unsigned int),
               &lockInit,
               0,
               NULL,
               NULL);
               
 	unsigned int queueSize = 1;
	unsigned int numberOfTasksToExecute = 1;
	
	
	// 4. Setup and Run kernels
        // kernel args
        cl_int argchk  = clSetKernelArg(scheduler_kernel, 0, sizeof(cl_mem), (void *)&taskGPU);
        argchk  = clSetKernelArg(scheduler_kernel, 1, sizeof(unsigned int), &queueSize);
        argchk  = clSetKernelArg(scheduler_kernel, 2, sizeof(unsigned int), &numberOfTasksToExecute);
        argchk  = clSetKernelArg(scheduler_kernel, 3, sizeof(cl_mem), (void *)&lockGPU);
        argchk  = clSetKernelArg(scheduler_kernel, 4, 2048, NULL);
        argchk  = clSetKernelArg(scheduler_kernel, 5, sizeof(cl_mem), (void *)&spoofingGPU);
        
        cl_ulong totalKernelTime = 0;
        
        // launch kernel
        cl_event kernelEvent;
        cl_ulong beginKTime,endKTime;
        for (int kernelLoop=totalBlocks;kernelLoop>0;kernelLoop-=32) {
        printf("Running scheduler...\n");

            status = clEnqueueNDRangeKernel(
                      command_queue,  scheduler_kernel, 1, 0,
                      &globalWorksize,&localWorksize,
                      0, NULL, &kernelEvent);
            if (status) {
                cl_errChk(status,"ERROR in Executing Scheduler Kernel");
                exit(0);
            }
        printf("Past scheduler...\n");

            clFinish (command_queue);

            printf("Starting profiling...\n");

            status = clGetEventProfilingInfo (kernelEvent,
                                    CL_PROFILING_COMMAND_START,
                                    sizeof(cl_ulong),
                                    &beginKTime,
                                    NULL);
            status = cl_errChk(status, (char *)"Error with profiling begin");
            

            clGetEventProfilingInfo (kernelEvent,
                                    CL_PROFILING_COMMAND_END,
                                    sizeof(cl_ulong),
                                    &endKTime,
                                    NULL);
            totalKernelTime+= (endKTime - beginKTime);
            status = cl_errChk(status, (char *)"Error with profiling end");
           clReleaseEvent(kernelEvent); 
            printf("Got here...\n");

        }
        printf("Total kernel time: %f\n",totalKernelTime*1e-9);    
}

int parseCommandline(int argc, char *argv[], char* filename,
                     int *q, int *t, int *p, int *d){
    int i;
    //if (argc < 2) return 1; // error
    //strncpy(filename,argv[1],100);
    filename = (char *)"";
    char flag;
    
    for(i=1;i<argc;i++) {
      if (argv[i][0]=='-') {// flag
        flag = argv[i][1];
          switch (flag) {
            case 'h': // help
              return 1;
              break;
            case 'q': // quiet
              *q = 1;
              break;
            case 't': // timing
              *t = 1;
              break;
            case 'p': // platform
              i++;
              *p = atoi(argv[i]);
              break;
            case 'd': // device
              i++;
              *d = atoi(argv[i]);
              break;
        }
      }
    }
    if ((*d >= 0 && *p<0) || (*p>=0 && *d<0)) // both p and d must be specified if either are specified
      return 1;
    return 0;
}

void printUsage(){
  printf("Gaussian Elimination Usage\n");
  printf("\n");
  printf("gaussianElimination [filename] [-hqt] [-p [int] -d [int]]\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./gaussianElimination matrix4.txt\n");
  printf("\n");
  printf("filename     the filename that holds the matrix data\n");
  printf("\n");
  printf("-h           Display the help file\n");
  printf("-q           Quiet mode. Suppress all text output.\n");
  printf("-t           Print timing information.\n");
  printf("\n");
  printf("-p [int]     Choose the platform (must choose both platform and device)\n");
  printf("-d [int]     Choose the device (must choose both platform and device)\n");
  printf("\n");
  printf("\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
  printf("       2. If you declare either the device or the platform,\n");
  printf("          you must declare both.\n\n");
}



#endif
