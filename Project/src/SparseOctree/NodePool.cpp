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
    cudaError_t clearNodePoolCuda(node *nodePool, neighbours* neighbourPool, int poolSize);
}

void NodePool::init(int nodeCount)
{
    m_poolSize = nodeCount;

    unsigned int* data = new unsigned int[nodeCount * 2];
    unsigned int* neighbourData = new unsigned int[nodeCount * 6];

    // make sure the node poll starts empty
    for(int i=0;i<nodeCount*2;i++)
        data[i] = 0U;

    for(int i=0;i<nodeCount*6;i++)
        neighbourData[i] = 0U;

    // just initialise the memory for the nodepool once
    // Position buffer
    glGenBuffers(1, &mNodePoolOutputBuffer);
    glBindBuffer(GL_TEXTURE_BUFFER, mNodePoolOutputBuffer);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(GLuint) * nodeCount * 2, data, GL_STATIC_DRAW);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);

    // nodepool texture
    glGenTextures(1, &mNodePoolOutputTexture);
    glBindTexture(GL_TEXTURE_BUFFER, mNodePoolOutputTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, mNodePoolOutputBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    cudaErrorCheck(cudaGraphicsGLRegisterBuffer(&mNodePoolFragmentList,mNodePoolOutputBuffer,cudaGraphicsMapFlagsNone));
    cudaErrorCheck(cudaGraphicsMapResources(1, &mNodePoolFragmentList, 0));

    size_t sizePosition = sizeof(GLuint) * nodeCount * 2;
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&m_dNodePool,
                                                        &sizePosition, mNodePoolFragmentList));

    cudaGraphicsUnmapResources(1, &mNodePoolFragmentList, 0);

    // create and bin buffers for the neighbourmap
    glGenBuffers(1, &mNeighbourPoolBuffer);
    glBindBuffer(GL_TEXTURE_BUFFER, mNeighbourPoolBuffer);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(GLuint)*nodeCount * 6, neighbourData, GL_STATIC_DRAW);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);

    //neighbourpool texture
    glGenTextures(1,&mNeighbourPoolTexture);
    glBindTexture(GL_TEXTURE_BUFFER, mNeighbourPoolTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI,mNeighbourPoolBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    // register neighbourbuffer to cuda
    cudaErrorCheck(cudaGraphicsGLRegisterBuffer(&mNeighbourPoolResource,mNeighbourPoolBuffer,cudaGraphicsMapFlagsNone));
    cudaErrorCheck(cudaGraphicsMapResources(1, &mNeighbourPoolResource,0));

    size_t size = sizeof(GLuint) * nodeCount * 6;
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&m_dNeighbourPool, &size,mNeighbourPoolResource));

    cudaGraphicsUnmapResources(1,&mNeighbourPoolResource,0);

    delete []data;
    delete []neighbourData;

    //cudaErrorCheck(cudaMalloc((void **)&m_dNodePool,sizeof(node)*nodeCount));
}

void NodePool::updateConstMemory()
{
    // TODO: do this :D
}

NodePool::~NodePool()
{
    cudaFree(m_dNodePool);
    cudaFree(m_dNeighbourPool);
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


    cudaErrorCheck(cudaGraphicsMapResources(1, &mNeighbourPoolResource, 0));

    size_t sizeNeighbour = sizeof(GLuint) * m_poolSize * 6;
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&m_dNeighbourPool,
                                                        &sizeNeighbour, mNeighbourPoolResource));
}

void NodePool::unmapFromCUDA()
{
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &mNodePoolFragmentList, 0));
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &mNeighbourPoolResource, 0));
}

void NodePool::bind(GLuint textureUnit)
{
    glActiveTexture(GL_TEXTURE0+textureUnit);
    glBindImageTexture(0,
                       mNodePoolOutputTexture,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_ONLY,
                       GL_R32UI);
}

void NodePool::clearNodePool()
{
    cudaErrorCheck(clearNodePoolCuda(m_dNodePool, m_dNeighbourPool, m_poolSize));
}

int NodePool::getNodePoolTextureID()
{
    return mNodePoolOutputTexture;
}

int NodePool::getNodePoolBufferID()
{
    return mNodePoolOutputBuffer;
}

void NodePool::bindNeighbourPool(GLuint textureUnit)
{
    glActiveTexture(GL_TEXTURE+textureUnit);
    glBindImageTexture(1,
                       mNeighbourPoolTexture,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_ONLY,
                       GL_R32UI);
}

int NodePool::getNeighbourPoolTextureID()
{
    return mNeighbourPoolTexture;
}

int NodePool::getNeighbourPoolBufferID()
{
    return mNeighbourPoolBuffer;
}

neighbours *NodePool::getNeighbourPoolDevicePointer()
{
    return m_dNeighbourPool;
}
