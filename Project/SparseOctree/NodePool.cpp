//
// Created by nils1990 on 03.12.15.
//

#include <driver_types.h>
#include <cuda_runtime.h>
#include <Utilities/errorUtils.h>
#include "NodePool.h"

extern "C" // this is not necessary imho, but gives a better idea on where the function comes from
{
    cudaError_t updateNodePool(cudaArray_t &voxel, node *nodePool, int poolSize);
}

void NodePool::init(int nodeCount)
{
    m_poolSize = nodeCount;

    // just initialise the memory for the nodepool once
    cudaErrorCheck(cudaMalloc((void **)&m_dNodePool,nodeCount*sizeof(node)));
}

void NodePool::updateConstMemory()
{
    // TODO: call nvcc method that maps the global structure of our octree to the const memory
}

void NodePool::fillNodePool(cudaArray_t &voxelList)
{
    cudaErrorCheck(updateNodePool(voxelList, m_dNodePool, m_poolSize));
}

NodePool::~NodePool()
{
    cudaFree(m_dNodePool);
}
