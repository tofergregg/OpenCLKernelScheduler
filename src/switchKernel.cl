void dispatch(__global SpoofedId *spoofing,
              __local WorkItem *next) {
    switch(next->task->kernelId) {
        case 0:
            kernel1(next->task->kernelArgs[0].globalUintArg,spoofing);
            break;
    }
}

