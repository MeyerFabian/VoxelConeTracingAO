#ifndef TRAVERSE_KERNELS_CUH
#define TRAVERSE_KERNELS_CUH

#include <src/SparseOctree/NodePool.h>

// traverses the nodepool and returns the adress of the node
// foundOnLevel is the level that corresponds to the found node
__device__
unsigned int traverseToCorrespondingNode(const node* nodepool, float3 &pos, unsigned int &foundOnLevel, unsigned int maxLevel)
{
    unsigned int nodeOffset = 0;
    unsigned int childPointer = 0;
    unsigned int nodeTile = 0;
    unsigned int offset = 0;
    uint3 nextOctant = make_uint3(0,0,0);

    // level 0 we just assume that maxdivided is 1 :P
    nodeTile = nodepool[offset].nodeTilePointer;
    childPointer = nodeTile & 0x3fffffff;

    for(unsigned int curLevel=1;curLevel<=maxLevel; curLevel++)
    {
        nextOctant.x = static_cast<unsigned int>(2 * pos.x);
        nextOctant.y = static_cast<unsigned int>(2 * pos.y);
        nextOctant.z = static_cast<unsigned int>(2 * pos.z);

        // make the octant position 1D for the linear memory
        nodeOffset = nextOctant.x + 2 * nextOctant.y + 4 * nextOctant.z;
        offset = nodeOffset + childPointer * 8;

        nodeTile = nodepool[offset].nodeTilePointer;

        unsigned int maxDivide = getBit(nodeTile,32);


        if(getBit(nodeTile,32) == 0 && curLevel != 8) // TODO: define global max level
            return 0;

        childPointer = nodeTile & 0x3fffffff;

        if(curLevel == maxLevel)
        {
            foundOnLevel = curLevel;

            if(maxLevel == 8)
            {
                //printf("found: %d, cur: %d, offset:%d, maxLevel:%d \n", foundOnLevel, curLevel, offset, maxLevel);
            }
        }

        pos.x = 2 * pos.x - nextOctant.x;
        pos.y = 2 * pos.y - nextOctant.y;
        pos.z = 2 * pos.z - nextOctant.z;
    }

    // return our node adress
    return offset;
}

#endif