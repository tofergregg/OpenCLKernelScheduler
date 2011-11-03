#define BLOCK_SIZE 16

void dispatch(__global SpoofedId *spoofing,
              __local WorkItem *next,
              __global Task *task) {
    
    __local float *localMemA = (__local float *)next;
    __local float *localMemB = localMemA + BLOCK_SIZE * BLOCK_SIZE;
    
    switch(task->kernelId) {
        case 0:
            //kernel1(task->kernelArgs[0].globalUintArg,spoofing);
            matrixMul(task->kernelArgs[0].globalFloatArg, 
                      task->kernelArgs[1].globalFloatArg,
                      task->kernelArgs[2].globalFloatArg,
                      localMemA,
                      localMemB,
	                  task->kernelArgs[5].uintArg,
	                  task->kernelArgs[6].uintArg,
	                  spoofing);
            break;
    }
}

