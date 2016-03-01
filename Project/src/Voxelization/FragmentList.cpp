#include "FragmentList.h"

#include "Utilities/errorUtils.h"

#include <assert.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

FragmentList::FragmentList(GLuint maxListSize)
{
    m_voxelCount = 0;
    m_maxListSize = maxListSize;

        // ### OpenGL ###

    // Position buffer
    glGenBuffers(1, &m_positionOutputBuffer);
    glBindBuffer(GL_TEXTURE_BUFFER, m_positionOutputBuffer);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(GLuint) * m_maxListSize, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);

    // Position texture
    glGenTextures(1, &m_positionOutputTexture);
    glBindTexture(GL_TEXTURE_BUFFER, m_positionOutputTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, m_positionOutputBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    // Normal buffer
    glGenBuffers(1, &m_normalOutputBuffer);
    glBindBuffer(GL_TEXTURE_BUFFER, m_normalOutputBuffer);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(GLubyte) * 4 * m_maxListSize, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);

    // Normal texture
    glGenTextures(1, &m_normalOutputTexture);
    glBindTexture(GL_TEXTURE_BUFFER, m_normalOutputTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA8, m_normalOutputBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    // Color buffer
    glGenBuffers(1, &m_colorOutputBuffer);
    glBindBuffer(GL_TEXTURE_BUFFER, m_colorOutputBuffer);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(GLubyte) * 4 * m_maxListSize, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);

    // Color texture
    glGenTextures(1, &m_colorOutputTexture);
    glBindTexture(GL_TEXTURE_BUFFER, m_colorOutputTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA8, m_colorOutputBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    // ### Cuda ###

    // Register the texture for cuda (just once)
    // ### POSITION
    cudaErrorCheck(cudaGraphicsGLRegisterBuffer(&m_positionFragmentList,m_positionOutputBuffer,cudaGraphicsMapFlagsReadOnly));
    cudaErrorCheck(cudaGraphicsMapResources(1, &m_positionFragmentList, 0));

    size_t sizePosition = m_maxListSize * sizeof(GLuint);
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&m_positionDevPointer,
                                                        &sizePosition, m_positionFragmentList));

    cudaGraphicsUnmapResources(1, &m_positionFragmentList, 0);

    // ### COLOR
    cudaErrorCheck(cudaGraphicsGLRegisterBuffer(&m_colorFragmentList,m_colorOutputBuffer,cudaGraphicsMapFlagsReadOnly));
    cudaErrorCheck(cudaGraphicsMapResources(1, &m_colorFragmentList, 0));

    size_t sizeColor = m_maxListSize * sizeof(GLubyte) * 4;
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&m_colorDevPointer,
                                         &sizeColor, m_colorFragmentList));

    cudaGraphicsUnmapResources(1, &m_colorFragmentList, 0);

    // ### NORMAL
    cudaErrorCheck(cudaGraphicsGLRegisterBuffer(&m_normalFragmentList,m_normalOutputBuffer,cudaGraphicsMapFlagsReadOnly));
    cudaErrorCheck(cudaGraphicsMapResources(1, &m_normalFragmentList, 0));

    size_t sizeNormal = m_maxListSize * sizeof(GLubyte) * 4;
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&m_normalDevPointer,
                                                        &sizeNormal, m_normalFragmentList));

    cudaGraphicsUnmapResources(1, &m_normalFragmentList, 0);

    glBindBuffer(GL_TEXTURE_BUFFER, 0);
}

FragmentList::~FragmentList()
{
    cudaGraphicsUnregisterResource(m_positionFragmentList);
    cudaGraphicsUnregisterResource(m_colorFragmentList);
    cudaGraphicsUnregisterResource(m_normalFragmentList);

    cudaGLUnregisterBufferObject(m_positionOutputBuffer);
    cudaGLUnregisterBufferObject(m_colorOutputBuffer);
    cudaGLUnregisterBufferObject(m_normalOutputBuffer);

    glDeleteTextures(1, &m_positionOutputTexture);
    glDeleteTextures(1, &m_colorOutputTexture);
    glDeleteTextures(1, &m_normalOutputTexture);
    glDeleteBuffers(1, &m_positionOutputBuffer);
    glDeleteBuffers(1, &m_colorOutputBuffer);
    glDeleteBuffers(1, &m_normalOutputBuffer);
}

void FragmentList::bind()
{
    glBindImageTexture(1,
                       m_positionOutputTexture,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_WRITE,
                       GL_R32UI);

    glBindImageTexture(2,
                       m_normalOutputTexture,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_WRITE,
                       GL_RGBA8);

    glBindImageTexture(3,
                       m_colorOutputTexture,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_WRITE,
                       GL_RGBA8);
}

void FragmentList::bindWriteonly()
{
    glBindImageTexture(1,
                       m_positionOutputTexture,
                       0,
                       GL_TRUE,
                       0,
                       GL_WRITE_ONLY,
                       GL_R32UI);

    glBindImageTexture(2,
                       m_normalOutputTexture,
                       0,
                       GL_TRUE,
                       0,
                       GL_WRITE_ONLY,
                       GL_RGBA8);

    glBindImageTexture(3,
                       m_colorOutputTexture,
                       0,
                       GL_TRUE,
                       0,
                       GL_WRITE_ONLY,
                       GL_RGBA8);
}

void FragmentList::bindReadonly()
{
    glBindImageTexture(1,
                       m_positionOutputTexture,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_ONLY,
                       GL_R32UI);

    glBindImageTexture(2,
                       m_normalOutputTexture,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_ONLY,
                       GL_RGBA8);

    glBindImageTexture(3,
                       m_colorOutputTexture,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_ONLY,
                       GL_RGBA8);
}

void FragmentList::bindPosition()
{
    glBindImageTexture(1,
                       m_positionOutputTexture,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_WRITE,
                       GL_R32UI);
}

int FragmentList::getVoxelCount() const
{
    return m_voxelCount;
}

void FragmentList::setVoxelCount(int count)
{
    assert(count >=0);

    m_voxelCount = count;
}

void FragmentList::mapToCUDA()
{
    // ### POSITION
    cudaErrorCheck(cudaGraphicsMapResources(1, &m_positionFragmentList, 0));

    size_t sizePosition = m_maxListSize * sizeof(GLuint);
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&m_positionDevPointer,
                                                        &sizePosition, m_positionFragmentList));

    // ### COLOR
    cudaErrorCheck(cudaGraphicsMapResources(1, &m_colorFragmentList, 0));

    size_t sizeColor = m_maxListSize * sizeof(GLubyte) * 4;
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&m_colorDevPointer,
                                                        &sizeColor, m_colorFragmentList));

    // ### NORMAL
    cudaErrorCheck(cudaGraphicsMapResources(1, &m_normalFragmentList, 0));

    size_t sizeNormal = m_maxListSize * sizeof(GLubyte) * 4;
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&m_normalDevPointer,
                                                        &sizeNormal, m_normalFragmentList));

}

void FragmentList::unmapFromCUDA()
{
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &m_positionFragmentList, 0));
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &m_colorFragmentList, 0));
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &m_normalFragmentList, 0));
}

uchar4 *FragmentList::getColorBufferDevPointer()
{
    return m_colorDevPointer;
}

uint1 *FragmentList::getPositionDevPointer()
{
    return m_positionDevPointer;
}

uchar4 *FragmentList::getNormalDevPointer()
{
    return m_normalDevPointer;
}
