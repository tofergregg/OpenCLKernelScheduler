void dispatch(__global SpoofedId *spoofing,
              __local WorkItem *next,
              __global uint *logInfo) {
    switch(next->task->kernelId) {
        case 0:
            kernel1(logInfo,spoofing);
            break;
    }
}
