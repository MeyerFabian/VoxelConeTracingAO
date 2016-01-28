#ifndef GLOBAL_RESOURCES_CUH
#define GLOBAL_RESOURCES_CUH

#include <SparseOctree/NodePool.h>

const int maxNodePoolSizeForConstMemory = 8168;
int volumeResolution = 384;

bool constantMemoryValid = false;   // the flag indicates wheather a kernel is allowed to use the constantNodePool
// this memory is unused atm. we might copy the top of our octree to the constant memory to increase the traversal speed
__constant__ node constNodePool[maxNodePoolSizeForConstMemory];
__constant__ uint3 lookup_octants[8];
__constant__ uint3 insertPositions[8];
__device__ unsigned int globalNodePoolCounter = 0; // counter for node pool memory reservation. Gets increased whenever a new node is allocated
__device__ unsigned int globalBrickPoolCounter = 0; // same as node counter but for bricks

surface<void, cudaSurfaceType3D> colorBrickPool; // the surface representation of our colorBrickPool (surface is needed for write access)
surface<void, cudaSurfaceType3D> normalBrickPool; // same as above, but for normals

#endif