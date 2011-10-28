#define LOCALTASKWORKSIZE 256

void dispatch(__global SpoofedId *spoofing,
              __local WorkItem *next, local uint *localMemStart) {
    int localSizeA = next->task->kernelArgs[3].uintArg;
    //int localSizeB = next->task->kernelArgs[4].uintArg;
    
    __local float *localMemA = localMemStart+LOCALTASKWORKSIZE;
    __local float *localMemB = localMemStart+LOCALTASKWORKSIZE+localSizeA;
    
    switch(next->task->kernelId) {
        case 0:
            //kernel1(next->task->kernelArgs[0].globalUintArg,spoofing);
            matrixMul(next->task->kernelArgs[0].globalFloatArg, 
                      next->task->kernelArgs[1].globalFloatArg,
                      next->task->kernelArgs[2].globalFloatArg,
                      localMemA,
                      localMemB,
	                  next->task->kernelArgs[5].globalUintArg,
	                  next->task->kernelArgs[6].globalUintArg,
	                  spoofing);
            break;
    }
}

