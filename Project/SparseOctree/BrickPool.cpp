//
// Created by nils1990 on 03.12.15.
//

#include "BrickPool.h"
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "Utilities/errorUtils.h"


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

void BrickPool::voxelizeMaxDetail()
{
    // TODO: Voxelize the current meshes as stated in: "OpenGL Insights"
}

void BrickPool::subdivideOctree()
{
    // TODO: mark all nodes that should be subdivided
}

void BrickPool::fillBricks()
{
    // TODO: call kernel that fills all "mipmaplevels" manually
}
