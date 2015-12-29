//
// Created by nils1990 on 28.12.15.
//

#include "FragmentList.h"
#include <assert.h>
#include <cuda_runtime_api.h>
#include <src/Utilities/errorUtils.h>
#include <cuda_gl_interop.h>

FragmentList::FragmentList(GLuint maxListSize) : mVoxelCount(0), mMaxListSize(maxListSize)
{
    init(maxListSize);
}

FragmentList::~FragmentList()
{

}

void FragmentList::init(GLuint maxListSize)
{
    mMaxListSize = maxListSize;

    // ### OpenGL ###

    // Position buffer
    glGenBuffers(1, &mPositionOutputBuffer);
    glBindBuffer(GL_TEXTURE_BUFFER, mPositionOutputBuffer);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(GLfloat) * mMaxListSize, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);

    // Position texture
    glGenTextures(1, &mPositionOutputTexture);
    glBindTexture(GL_TEXTURE_BUFFER, mPositionOutputTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, mPositionOutputBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    // Normal buffer
    glGenBuffers(1, &mNormalOutputBuffer);
    glBindBuffer(GL_TEXTURE_BUFFER, mNormalOutputBuffer);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(GLubyte) * 4 * mMaxListSize, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);

    // Normal texture
    glGenTextures(1, &mNormalOutputTexture);
    glBindTexture(GL_TEXTURE_BUFFER, mNormalOutputTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA8, mNormalOutputBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    // Color buffer
    glGenBuffers(1, &mColorOutputBuffer);
    glBindBuffer(GL_TEXTURE_BUFFER, mColorOutputBuffer);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(GLubyte) * 4 * mMaxListSize, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);

    // Color texture
    glGenTextures(1, &mColorOutputTexture);
    glBindTexture(GL_TEXTURE_BUFFER, mColorOutputTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA8, mColorOutputBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    // ### Cuda ###

    // TODO for Nils

    // Register the texture for cuda (just once)
    cudaErrorCheck(cudaGraphicsGLRegisterBuffer(&mFragmentListResource,mColorOutputBuffer,cudaGraphicsMapFlagsReadOnly));
    cudaErrorCheck(cudaGraphicsMapResources(1, &mFragmentListResource, 0));

    size_t size = maxListSize * sizeof(GLubyte) * 4;
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&mDevPointer,
                                         &size,
                                         mFragmentListResource));

    cudaGraphicsUnmapResources(1, &mFragmentListResource, 0);

    glBindBuffer(GL_TEXTURE_BUFFER, 0);
}

void FragmentList::bind()
{
    glActiveTexture(GL_TEXTURE1);
    glBindImageTexture(1,
                       mColorOutputTexture,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_WRITE,
                       GL_R32UI);

    glActiveTexture(GL_TEXTURE2);
    glBindImageTexture(2,
                       mNormalOutputTexture,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_WRITE,
                       GL_RGBA8);

    glActiveTexture(GL_TEXTURE3);
    glBindImageTexture(3,
                       mColorOutputTexture,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_WRITE,
                       GL_RGBA8);
}

int FragmentList::getVoxelCount() const
{
    return mVoxelCount;
}

void FragmentList::setVoxelCount(int count)
{
    assert(count >=0);

    mVoxelCount = count;
}

void FragmentList::mapToCUDA()
{
    cudaErrorCheck(cudaGraphicsMapResources(1, &mFragmentListResource, 0));

    size_t size = mMaxListSize * sizeof(GLubyte) * 4;
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&mDevPointer,
                                                        &size,
                                                        mFragmentListResource));
}

void FragmentList::unmapFromCUDA()
{
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &mFragmentListResource, 0));
}

uchar4 *FragmentList::getColorBufferDevPointer()
{
    return mDevPointer;
}
