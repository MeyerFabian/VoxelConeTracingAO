#ifndef GLOBAL_RESOURCES_CUH
#define GLOBAL_RESOURCES_CUH

#include <SparseOctree/NodePool.h>

int volumeResolution = 384;

struct LevelInterval
{
    unsigned int start;
    unsigned int end;
};

// this memory is unused atm. we might copy the top of our octree to the constant memory to increase the traversal speed
__constant__ uint3 lookup_octants[8];
__constant__ uint3 insertPositions[8];
__constant__ LevelInterval constLevelIntervalMap[10]; // a little more memory than needed..

//
LevelInterval LevelIntervalMap[10]; // we save the memory positions of every octree level. this should guarantee better performance

// the counter for node/brick-adresses
unsigned int *d_counter;

surface<void, cudaSurfaceType3D> colorBrickPool; // the surface representation of our colorBrickPool (surface is needed for write access)
surface<void, cudaSurfaceType3D> normalBrickPool; // same as above, but for normals

#endif