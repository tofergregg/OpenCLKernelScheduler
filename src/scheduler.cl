#pragma OPENCL EXTENSION cl_amd_printf:enable

#include "types.h"

#ifndef MAXTASKS
#   define MAXTASKS 32
#endif

#define get_global_id(num)   spoofing[(get_global_id)(0)].globalId[ num ]
#define get_local_id(num)    spoofing[(get_global_id)(0)].localId[ num ]
#define get_global_size(num) spoofing[(get_global_id)(0)].globalSize[ num ]
#define get_local_size(num)  spoofing[(get_global_id)(0)].localSize[ num ]
#define get_group_id(num)    spoofing[(get_global_id)(0)].groupId[ num ]
#define get_num_groups(num)  spoofing[(get_global_id)(0)].numGroups[ num ]
#include "matrixMul.cl"
// #include "kernel1.cl"
// #include "kernel2"
#undef get_global_id
#undef get_local_id
#undef get_global_size
#undef get_local_size
#undef get_group_id
#undef get_num_groups

#include "switchKernel.cl"

unsigned int getWork(__global Task *queue, 
             const int queueSize, 
             unsigned int numberOfTasksToExecute,
             WorkItem *workItem,
             __global unsigned int *lock);

__kernel void scheduler(__global Task *queue,
                        const int queueSize,
                        unsigned int numberOfTasksToExecute,
                        __global unsigned int *lock,
                        __local unsigned int *sharedMem,
                        __global SpoofedId *spoofing) {//,
                        //__global unsigned int *logInfo) {
        
    size_t globalId = get_global_id(0);
    size_t workgroups = get_num_groups(0);
    size_t localId = get_local_id(0);
    
    WorkItem workItems[MAXTASKS];
    
    if (localId == 0) {
        sharedMem[0] = getWork(queue,
                               queueSize,
                               numberOfTasksToExecute,
                               workItems,
                               lock);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    numberOfTasksToExecute -= sharedMem[0];
    for (unsigned int i=0; i < numberOfTasksToExecute; i++) {
        __local WorkItem *next = (__local WorkItem *)sharedMem;
        if (localId == 0)
            *next = workItems[i];
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        spoofing[globalId].localId[0] = localId % next->task->xThreads;
        spoofing[globalId].localId[1] = localId / next->task->xThreads;
        spoofing[globalId].localId[2] = 1;
        
        spoofing[globalId].globalId[0] = next->x * next->task->xThreads + spoofing[globalId].localId[0];
        spoofing[globalId].globalId[1] = next->y * next->task->yThreads + spoofing[globalId].localId[1];
        spoofing[globalId].globalId[2] = 1;
        
        spoofing[globalId].globalSize[0] = next->task->xDim * next->task->xThreads;
        spoofing[globalId].globalSize[1] = next->task->yDim * next->task->yThreads;
        spoofing[globalId].globalSize[2] = 1;

        spoofing[globalId].localSize[0] = next->task->xThreads;
        spoofing[globalId].localSize[1] = next->task->yThreads;
        spoofing[globalId].localSize[2] = 1;
        
        spoofing[globalId].groupId[0] = next->x;
        spoofing[globalId].groupId[1] = next->y;
        spoofing[globalId].groupId[2] = 1;
        
        spoofing[globalId].numGroups[0] = next->task->xDim;
        spoofing[globalId].numGroups[1] = next->task->yDim;
        spoofing[globalId].numGroups[2] = 1;
        
        if(next->task->xThreads * next->task->yThreads > get_local_id(0)){
            dispatch(spoofing,next,sharedMem);
        }
    }
}

unsigned int getWork(__global Task *queue, 
             const int queueSize, 
             unsigned int numberOfTasksToExecute,
             WorkItem *workItem,
             __global unsigned int *lock) {
    
    __global Task *task;
    unsigned int index = 0;
    
    // acquire spin-lock
    while (atomic_xchg(lock,1)==1);
    
    // grab numberOfTasksToExecute tasks
    for (int i=0;i<queueSize && numberOfTasksToExecute > 0;i++) {
        task = &queue[i];
        unsigned int workgroupsGrabbed = min(task->workgroupsLeft,numberOfTasksToExecute);
        for (unsigned int workgroup = 0; workgroup < workgroupsGrabbed; workgroup++) {
            unsigned int workgroupId = task->workgroupsLeft - workgroup - 1;
            workItem[index].x = workgroupId % task->xDim;
            workItem[index].y = workgroupId / task->xDim;
            workItem[index].z = 1;
            workItem[index].task = task;
            index++;
        }
        task->workgroupsLeft -= workgroupsGrabbed;
        numberOfTasksToExecute -= workgroupsGrabbed;
    }
    
    // release spin-lock
    atomic_xchg(lock,0);
    
    // return number of extra tasks that didn't get executed
    return numberOfTasksToExecute;
}

__kernel void setArg(__global Task *task,
                     uint taskNum,
                     uint index,
                     ArgType arg) {
    task[taskNum].kernelArgs[index] = arg;
}

__kernel void setArgUint(__global Task *task,
                     uint taskNum,
                     uint index,
                     uint arg) {
    task[taskNum].kernelArgs[index].uintArg = arg;
}

__kernel void setArgGlobalFloat(__global Task *task,
                     uint taskNum,
                     uint index,
                     __global float *arg) {
    task[taskNum].kernelArgs[index].globalFloatArg = arg;
}

__kernel void setArgLocal(__global Task *task,
                     uint taskNum,
                     uint index,
                     uint size) {
    task[taskNum].kernelArgs[index].uintArg = size;
}




