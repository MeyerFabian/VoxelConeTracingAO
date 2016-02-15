#ifndef GLOBAL_RESOURCES_CUH
#define GLOBAL_RESOURCES_CUH

#include <SparseOctree/NodePool.h>

unsigned int volumeResolution = 384;

struct LevelInterval
{
    unsigned int start;
    unsigned int end;
};

// this memory is unused atm. we might copy the top of our octree to the constant memory to increase the traversal speed
__constant__ uint3 lookup_octants[8];
__constant__ uint3 insertPositions[8];
__constant__ LevelInterval constLevelIntervalMap[10]; // a little more memory than needed..
__constant__ uint3 constLookUp1Dto3DIndex[27]; // is used in mipmapping to get all 27 voxels within a brick. Prevents modulo operations

//
LevelInterval LevelIntervalMap[10]; // we save the memory positions of every octree level. this should guarantee better performance

// the counter for node/brick-adresses
unsigned int *d_counter;

surface<void, cudaSurfaceType3D> colorBrickPool; // the surface representation of our colorBrickPool (surface is needed for write access)
surface<void, cudaSurfaceType3D> normalBrickPool; // same as above, but for normals

// threadcounts
const unsigned int threadsPerBlockMipMap = 256;
const unsigned int threadPerBlockSpread = 512;
const unsigned int threadPerBlockReserve = 512;
const unsigned int threadsPerBlockFragmentList = 512;
const unsigned int threadsPerBlockClear = 256;
const unsigned int threadsPerBlockCombineBorders = 1024;
const dim3 block_dim(8,8,8);

unsigned int blockCountSpread;

#endif