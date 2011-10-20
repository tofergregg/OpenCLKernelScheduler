#pragma OPENCL EXTENSION cl_amd_printf:enable

__kernel void kernel1(__global uint *logInfo, 
                      __global SpoofedId *spoofing){
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    
    uint lidx = get_local_id(0);
    uint lidy = get_local_id(1);
    
    uint index = gidy + gidx * get_global_size(1);
    logInfo[2*index]=lidx;
    logInfo[2*index+1]=lidy;
    
    printf("gids:%d,%d; lids:%d,%d,get_global_size(1):%d,groupIds:%d,%d\n",
            gidx,gidy,lidx,lidy,get_global_size(1),get_group_id(0),get_group_id(1));
}


