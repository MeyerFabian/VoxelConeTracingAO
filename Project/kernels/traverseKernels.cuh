#ifndef TRAVERSE_KERNELS_CUH
#define TRAVERSE_KERNELS_CUH

#include <src/SparseOctree/NodePool.h>
#include "bitUtilities.cuh"

// traverses the nodepool and returns the adress of the node
// foundOnLevel is the level that corresponds to the found node
__device__
unsigned int traverseToCorrespondingNode(const node* nodePool, const float3 position, unsigned int &foundOnLevel, unsigned int maxLevel)
{
    unsigned int nodeOffset = 0;
    unsigned int childPointer = 0;
    unsigned int offset=0;
    unsigned int node = 0;
    unsigned int value = 0;

    float3 pos = position;

    // load first level manually
    node = nodePool[0].nodeTilePointer;
    childPointer = node & 0x3fffffff;

    for (int i = 1; i <= maxLevel+1; i++)
    {
        uint3 nextOctant = make_uint3(0, 0, 0);
        // determine octant for the given voxel
        nextOctant.x = static_cast<unsigned int>(2 * pos.x);
        nextOctant.y = static_cast<unsigned int>(2 * pos.y);
        nextOctant.z = static_cast<unsigned int>(2 * pos.z);

        pos.x = 2 * pos.x - nextOctant.x;
        pos.y = 2 * pos.y - nextOctant.y;
        pos.z = 2 * pos.z - nextOctant.z;

        // make the octant position 1D for the linear memory
        nodeOffset = nextOctant.x + 2 * nextOctant.y + 4 * nextOctant.z;
        offset = nodeOffset + childPointer * 8;

        node = nodePool[offset].nodeTilePointer;
        __syncthreads();
        if(getBit(node,32) == 1)
        {
            childPointer = node & 0x3fffffff;
            foundOnLevel++;
        }
        else if(maxLevel == 6)
        {
            foundOnLevel++;
        }
        else
        {
            foundOnLevel = 0;
            break;
        }
    }

    foundOnLevel--;

    // return our node adress
    return offset;
}

#endif