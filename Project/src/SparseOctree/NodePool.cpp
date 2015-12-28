//
// Created by nils1990 on 03.12.15.
//

#include <driver_types.h>
#include <cuda_runtime.h>
#include <Utilities/errorUtils.h>
#include <iostream>
#include "NodePool.h"

extern "C" // this is not necessary imho, but gives a better idea on where the function comes from
{
    cudaError_t updateNodePool(uchar4* colorBufferDevPointer, node *nodePool, int poolSize);
    cudaError_t copyNodePoolToConstantMemory(node *nodePool, int poolSize);
}

void NodePool::init(int nodeCount)
{
    m_poolSize = nodeCount;

    // just initialise the memory for the nodepool once
    cudaErrorCheck(cudaMalloc((void **)&m_dNodePool,sizeof(node)*nodeCount));
}

void NodePool::updateConstMemory()
{
    cudaErrorCheck(copyNodePoolToConstantMemory(m_dNodePool, m_poolSize));
}

void NodePool::fillNodePool(uchar4* colorBufferDevPointer)
{
    cudaErrorCheck(updateNodePool(colorBufferDevPointer, m_dNodePool, m_poolSize));
}

NodePool::~NodePool()
{
    cudaFree(m_dNodePool);
}

int NodePool::getPoolSize()
{
    return m_poolSize;
}
