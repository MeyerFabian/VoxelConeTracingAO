//
// Created by nils1990 on 03.12.15.
//

#include <driver_types.h>
#include <cuda_runtime.h>
#include <Utilities/errorUtils.h>
#include <cuda_gl_interop.h>
#include "NodePool.h"

extern "C" // this is not necessary imho, but gives a better idea on where the function comes from
{
    cudaError_t updateNodePool(uchar4* colorBufferDevPointer, node *nodePool, int poolSize);
    cudaError_t clearNodePoolCuda(node *nodePool, int poolSize);
    cudaError_t copyNodePoolToConstantMemory(node *nodePool, int poolSize);
}

void NodePool::init(int nodeCount)
{
    m_poolSize = nodeCount;

    unsigned int* data = new unsigned int[nodeCount * 2];

    // make sure the node poll starts empty
    for(int i=0;i<nodeCount*2;i++)
        data[i] = 0U;

    // just initialise the memory for the nodepool once
    // Position buffer
    glGenBuffers(1, &mNodePoolOutputBuffer);
    glBindBuffer(GL_TEXTURE_BUFFER, mNodePoolOutputBuffer);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(GLuint) * nodeCount * 2, data, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);

    // nodepool texture
    glGenTextures(1, &mNodePoolOutputTexture);
    glBindTexture(GL_TEXTURE_BUFFER, mNodePoolOutputTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, mNodePoolOutputBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    cudaErrorCheck(cudaGraphicsGLRegisterBuffer(&mNodePoolFragmentList,mNodePoolOutputBuffer,cudaGraphicsMapFlagsReadOnly));
    cudaErrorCheck(cudaGraphicsMapResources(1, &mNodePoolFragmentList, 0));

    size_t sizePosition = sizeof(GLuint) * nodeCount * 2;
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&m_dNodePool,
                                                        &sizePosition, mNodePoolFragmentList));

    cudaGraphicsUnmapResources(1, &mNodePoolFragmentList, 0);

    delete data;

    //cudaErrorCheck(cudaMalloc((void **)&m_dNodePool,sizeof(node)*nodeCount));
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

node *NodePool::getNodePoolDevicePointer()
{
    return m_dNodePool;
}

void NodePool::mapToCUDA()
{
    cudaErrorCheck(cudaGraphicsMapResources(1, &mNodePoolFragmentList, 0));

    size_t sizePosition = sizeof(GLuint) * m_poolSize * 2;
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&m_dNodePool,
                                                        &sizePosition, mNodePoolFragmentList));
}

void NodePool::unmapFromCUDA()
{
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &mNodePoolFragmentList, 0));
}

void NodePool::bind()
{
    glActiveTexture(GL_TEXTURE1);
    glBindImageTexture(1,
                       mNodePoolOutputTexture,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_WRITE,
                       GL_R32UI);
}

void NodePool::clearNodePool()
{
    cudaErrorCheck(clearNodePoolCuda(m_dNodePool, m_poolSize));
}

int NodePool::getNodePoolTextureID()
{
    return mNodePoolOutputTexture;
}

int NodePool::getNodePoolBufferID()
{
    return mNodePoolOutputBuffer;
}
