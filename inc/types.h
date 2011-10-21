#ifndef __TYPES__
#define __TYPES__

#define MAXARGS 5

#ifdef __OPENCL_VERSION__
    // Being read on the device side: need to add cl_ prefixes
#   define VEC( type ) \
    typedef type       cl_ ## type ; \
    typedef type ##  2 cl_ ## type ##  2 ; \
    typedef type ##  3 cl_ ## type ##  3 ; \
    typedef type ##  4 cl_ ## type ##  4 ; \
    typedef type ##  8 cl_ ## type ##  8 ; \
    typedef type ## 16 cl_ ## type ## 16

    VEC( char );
    VEC( uchar );
    VEC( short );
    VEC( ushort );
    VEC( int );
    VEC( uint );
    VEC( long );
    VEC( ulong );
    VEC( float );

    typedef half cl_half;

#   undef VEC
#else
// Being read on the host side: need to include OpenCL
#   if defined (__APPLE__) || defined(MACOSX)
#       include <OpenCL/opencl.h>
#   else
#       include <CL/opencl.h>
#   endif
#endif

typedef struct
    {
        unsigned int globalId[3];
        unsigned int localId[3];
        unsigned int globalSize[3];
        unsigned int localSize[3];
        unsigned int groupId[3];
        unsigned int numGroups[3];
    } SpoofedId;

typedef /*unholy*/ union
    {
        uint uintArg;
        float floatArg;
#ifdef __OPENCL_VERSION__
        __global uint * globalUintArg;
        __global float * globalFloatArg;
        __local uint * localUintArg;
        __local float * localFloatArg;
#endif
        char padding[8];

    } ArgType;

typedef struct
    {
        unsigned int workgroupsLeft; // number of workgroups in each dimension multiplied together
        unsigned int xDim,yDim;
        unsigned int xThreads,yThreads;
        unsigned int kernelId;
        ArgType kernelArgs[MAXARGS];
    } Task; 


#ifdef __OPENCL_VERSION__
typedef struct
    {
        unsigned int x,y,z;
        __global Task *task;
    } WorkItem;
#endif

#endif // __TYPES__

