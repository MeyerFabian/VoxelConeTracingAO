#ifndef TRAVERSE_KERNELS_CUH
#define TRAVERSE_KERNELS_CUH

#include <src/SparseOctree/NodePool.h>

// traverses the nodepool and returns the adress of the node
// foundOnLevel is the level that corresponds to the found node
__device__
unsigned int traverseToCorrespondingNode(const node* nodepool, float3 &pos, unsigned int &foundOnLevel, unsigned int maxLevel)
{
    bool useConst = true;
    unsigned int nodeOffset = 0;
    unsigned int childPointer = 0;
    unsigned int nodeTile = 0;
    unsigned int offset = 0;
    uint3 nextOctant = make_uint3(0,0,0);

    // level 0 we just assume that maxdivided is 1 :P
    nodeTile = constNodePool[offset].nodeTilePointer;//nodepool[offset].nodeTilePointer;
    childPointer = nodeTile & 0x3fffffff;

    for(unsigned int curLevel=1;curLevel<=maxLevel; curLevel++)
    {
        nextOctant.x = static_cast<unsigned int>(2 * pos.x);
        nextOctant.y = static_cast<unsigned int>(2 * pos.y);
        nextOctant.z = static_cast<unsigned int>(2 * pos.z);

        // make the octant position 1D for the linear memory
        nodeOffset = nextOctant.x + 2 * nextOctant.y + 4 * nextOctant.z;
        offset = nodeOffset + childPointer * 8;

        if(offset >= 8168)
            useConst = false;

        if(useConst)
            nodeTile = constNodePool[offset].nodeTilePointer;
        else
            nodeTile = nodepool[offset].nodeTilePointer;

        unsigned int maxDivide = getBit(nodeTile,32);

        foundOnLevel = curLevel;

        if(maxDivide == 1)
            childPointer = nodeTile & 0x3fffffff;
        else
            break;

        pos.x = 2 * pos.x - nextOctant.x;
        pos.y = 2 * pos.y - nextOctant.y;
        pos.z = 2 * pos.z - nextOctant.z;
    }

    // return our node adress
    return offset;
}

#endif