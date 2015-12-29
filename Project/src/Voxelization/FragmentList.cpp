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
    cudaGraphicsUnregisterResource(mPositionFragmentList);
    cudaGraphicsUnregisterResource(mColorFragmentList);
    cudaGraphicsUnregisterResource(mNormalFragmentList);

    cudaGLUnregisterBufferObject(mPositionOutputBuffer);
    cudaGLUnregisterBufferObject(mColorOutputBuffer);
    cudaGLUnregisterBufferObject(mNormalOutputBuffer);

    glDeleteTextures(1, &mPositionOutputTexture);
    glDeleteTextures(1, &mColorOutputTexture);
    glDeleteTextures(1, &mNormalOutputTexture);
    glDeleteBuffers(1, &mPositionOutputBuffer);
    glDeleteBuffers(1, &mColorOutputBuffer);
    glDeleteBuffers(1, &mNormalOutputBuffer);
}

void FragmentList::init(GLuint maxListSize)
{
    mMaxListSize = maxListSize;

    // ### OpenGL ###

    // Position buffer
    glGenBuffers(1, &mPositionOutputBuffer);
    glBindBuffer(GL_TEXTURE_BUFFER, mPositionOutputBuffer);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(GLuint) * mMaxListSize, 0, GL_DYNAMIC_DRAW);
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

    // Register the texture for cuda (just once)
    // ### POSITION
    cudaErrorCheck(cudaGraphicsGLRegisterBuffer(&mPositionFragmentList,mPositionOutputBuffer,cudaGraphicsMapFlagsReadOnly));
    cudaErrorCheck(cudaGraphicsMapResources(1, &mPositionFragmentList, 0));

    size_t sizePosition = maxListSize * sizeof(GLuint);
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&mPositionDevPointer,
                                                        &sizePosition, mPositionFragmentList));

    cudaGraphicsUnmapResources(1, &mPositionFragmentList, 0);

    // ### COLOR
    cudaErrorCheck(cudaGraphicsGLRegisterBuffer(&mColorFragmentList,mColorOutputBuffer,cudaGraphicsMapFlagsReadOnly));
    cudaErrorCheck(cudaGraphicsMapResources(1, &mColorFragmentList, 0));

    size_t sizeColor = maxListSize * sizeof(GLubyte) * 4;
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&mColorDevPointer,
                                         &sizeColor, mColorFragmentList));

    cudaGraphicsUnmapResources(1, &mColorFragmentList, 0);

    // ### NORMAL
    cudaErrorCheck(cudaGraphicsGLRegisterBuffer(&mNormalFragmentList,mNormalOutputBuffer,cudaGraphicsMapFlagsReadOnly));
    cudaErrorCheck(cudaGraphicsMapResources(1, &mNormalFragmentList, 0));

    size_t sizeNormal = maxListSize * sizeof(GLubyte) * 4;
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&mNormalDevPointer,
                                                        &sizeNormal, mNormalFragmentList));

    cudaGraphicsUnmapResources(1, &mNormalFragmentList, 0);



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
    // ### POSITION
    cudaErrorCheck(cudaGraphicsMapResources(1, &mPositionFragmentList, 0));

    size_t sizePosition = mMaxListSize * sizeof(GLuint);
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&mPositionDevPointer,
                                                        &sizePosition, mPositionFragmentList));

    // ### COLOR
    cudaErrorCheck(cudaGraphicsMapResources(1, &mColorFragmentList, 0));

    size_t sizeColor = mMaxListSize * sizeof(GLubyte) * 4;
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&mColorDevPointer,
                                                        &sizeColor, mColorFragmentList));

    // ### NORMAL
    cudaErrorCheck(cudaGraphicsMapResources(1, &mNormalFragmentList, 0));

    size_t sizeNormal = mMaxListSize * sizeof(GLubyte) * 4;
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&mNormalDevPointer,
                                                        &sizeNormal, mNormalFragmentList));

}

void FragmentList::unmapFromCUDA()
{
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &mPositionFragmentList, 0));
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &mColorFragmentList, 0));
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &mNormalFragmentList, 0));
}

uchar4 *FragmentList::getColorBufferDevPointer()
{
    return mColorDevPointer;
}

uint1 *FragmentList::getPositionDevPointer()
{
    return mPositionDevPointer;
}

uchar4 *FragmentList::getNormalDevPointer()
{
    return mNormalDevPointer;
}
