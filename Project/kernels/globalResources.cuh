#ifndef GLOBAL_RESOURCES_CUH
#define GLOBAL_RESOURCES_CUH

#include <SparseOctree/NodePool.h>

const int maxNodePoolSizeForConstMemory = 1024;
int volumeResolution = 384;

bool constantMemoryValid = false;   // the flag indicates wheather a kernel is allowed to use the constantNodePool
__constant__ node constNodePool[maxNodePoolSizeForConstMemory];
__constant__ int constVolumeResolution[1];
__device__ unsigned int globalNodePoolCounter = 0;
__device__ unsigned int globalBrickPoolCounter = 0;

surface<void, cudaSurfaceType3D> colorBrickPool;
surface<void, cudaSurfaceType3D> normalBrickPool;

#endif