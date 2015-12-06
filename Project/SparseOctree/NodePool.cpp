//
// Created by nils1990 on 03.12.15.
//

#include <driver_types.h>

extern "C" // this is not necessary imho, but gives a better idea on where the function comes from
{
    void updateNodePool(cudaArray_t &voxel);
}

#include "NodePool.h"

void NodePool::init(int nodeCount)
{
    // TODO: call nvcc method that generates the memory
}

void NodePool::updateConstMemory()
{
    // TODO: call nvcc method that maps the global structure of our octree to the const memory
}

void NodePool::fillNodePool(cudaArray_t &voxelList)
{
    updateNodePool(voxelList);
}
