__kernel void kernel1(__global uint *logInfo, 
                      __global SpoofedId *spoofing){
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    
    uint lidx = get_local_id(0);
    uint lidy = get_local_id(1);
    
    uint index = gidy + gidx * get_global_size(1);
    //uint index = (get_global_id)(0);
    //logInfo[2*index]=gidx;
    //logInfo[2*index+1]=gidy;
    
    //logInfo[2*index]=gidx;
    //logInfo[2*index+1]=gidy;
    
    logInfo[2*index]=lidx;
    logInfo[2*index+1]=lidy;
    //printf("logInfo:%d,%d\n",logInfo[2*index],logInfo[2*index+1]);

}

