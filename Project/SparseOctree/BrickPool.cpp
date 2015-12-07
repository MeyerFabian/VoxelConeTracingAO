//
// Created by nils1990 on 03.12.15.
//

#include "BrickPool.h"
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "Utilities/errorUtils.h"

extern "C" // this is not necessary imho, but gives a better idea on where the function comes from
{
    cudaError_t updateBrickPool(cudaArray_t &brickPool, dim3 textureDim);
}


BrickPool::BrickPool()
{

}

BrickPool::~BrickPool()
{
    glDeleteTextures(1,&m_brickPoolID);
}

void BrickPool::init(int width, int height, int depth)
{
    int max;
    glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max);

    m_poolSize.x = static_cast<unsigned int>(width);
    m_poolSize.y = static_cast<unsigned int>(height);
    m_poolSize.z = static_cast<unsigned int>(depth);

    //TODO: implement a method that is able to query the real max size.. depends on type and format..
    //load data into a 3D texture
    glGenTextures(1, &m_brickPoolID);
    glBindTexture(GL_TEXTURE_3D, m_brickPoolID);

    // set the texture parameters
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    glTexImage3D(GL_TEXTURE_3D,0,GL_RGBA32F,width,height,depth,0,GL_RGBA, GL_FLOAT, NULL);

    GLenum error = glGetError();

    switch (error)
    {
        case GL_INVALID_VALUE:

            break;
        case GL_INVALID_ENUM:

            break;
        case GL_INVALID_OPERATION:

            break;
        case GL_INVALID_FRAMEBUFFER_OPERATION:

            break;
        case GL_OUT_OF_MEMORY:

            break;
        case GL_NO_ERROR:
            break;
    }
}


void BrickPool::registerTextureForCUDAWriting()
{
    cudaErrorCheck(cudaGraphicsGLRegisterImage(&m_brickPoolRessource, m_brickPoolID, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
}

void BrickPool::registerTextureForCUDAReading()
{
    cudaErrorCheck(cudaGraphicsGLRegisterImage(&m_brickPoolRessource, m_brickPoolID, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsReadOnly));
}

void BrickPool::unregisterTextureForCUDA()
{
    cudaErrorCheck(cudaGraphicsUnregisterResource(m_brickPoolRessource));
}


void BrickPool::mapRessourceToArray()
{
    cudaErrorCheck(cudaGraphicsMapResources(1, &m_brickPoolRessource, 0));
    cudaErrorCheck(cudaGraphicsSubResourceGetMappedArray(&m_brickPoolArray, m_brickPoolRessource, 0, 0));
}

void BrickPool::unmapRessource() {
    cudaGraphicsUnmapResources(1, &m_brickPoolRessource, 0);
}

cudaArray_t *BrickPool::fillBrickPool(const NodePool &nodepool)
{
    mapRessourceToArray();
    cudaErrorCheck(updateBrickPool(m_brickPoolArray, m_poolSize));
    unmapRessource();
    return &m_brickPoolArray;
}
