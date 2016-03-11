#include "FragmentList.h"

#include "Utilities/errorUtils.h"

#include <assert.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

FragmentList::FragmentList(GLuint voxelizationResolution, GLuint maxListSize)
{
    mVolumeResolution = voxelizationResolution;
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

    // ### Cuda ###

    // ### POSITION
    cudaErrorCheck(cudaGraphicsGLRegisterBuffer(&m_positionFragmentList,m_positionOutputBuffer,cudaGraphicsMapFlagsReadOnly));
    cudaErrorCheck(cudaGraphicsMapResources(1, &m_positionFragmentList, 0));

    size_t sizePosition = m_maxListSize * sizeof(GLuint);
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&m_positionDevPointer,
                                                        &sizePosition, m_positionFragmentList));

    cudaGraphicsUnmapResources(1, &m_positionFragmentList, 0);

    glBindBuffer(GL_TEXTURE_BUFFER, 0);

    // Volumes
    createVolumes();

    // ### COLOR
    cudaErrorCheck(cudaGraphicsGLRegisterImage(&mColorVolumeResource, mColorVolume, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsReadOnly));
    cudaErrorCheck(cudaGraphicsMapResources(1, &mColorVolumeResource, 0));
    cudaErrorCheck(cudaGraphicsSubResourceGetMappedArray(&mColorVolumeArray, mColorVolumeResource, 0, 0));
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &mColorVolumeResource, 0));

    // ### NORMAL
    cudaErrorCheck(cudaGraphicsGLRegisterImage(&mNormalVolumeResource, mNormalVolume, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsReadOnly));
    cudaErrorCheck(cudaGraphicsMapResources(1, &mNormalVolumeResource, 0));
    cudaErrorCheck(cudaGraphicsSubResourceGetMappedArray(&mNormalVolumeArray, mNormalVolumeResource, 0, 0));
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &mNormalVolumeResource, 0));
}

FragmentList::~FragmentList()
{
    // Position
    cudaGraphicsUnregisterResource(m_positionFragmentList);
    cudaGLUnregisterBufferObject(m_positionOutputBuffer);

    glDeleteTextures(1, &m_positionOutputTexture);
    glDeleteBuffers(1, &m_positionOutputBuffer);

    // Color and normal
    cudaGraphicsUnregisterResource(mColorVolumeResource);
    cudaGraphicsUnregisterResource(mNormalVolumeResource);

    // Delete volume textures
    deleteVolumes();
}

void FragmentList::reset()
{
    GLuint clearInt = 0;
    glClearTexImage(mColorVolume, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &clearInt);
    glClearTexImage(mNormalVolume, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &clearInt);
}

void FragmentList::bind() const
{
    glBindImageTexture(1,
                       m_positionOutputTexture,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_WRITE,
                       GL_R32UI);

    glBindImageTexture(2,
                       mColorVolume,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_WRITE,
                       GL_R32UI);

    glBindImageTexture(3,
                       mNormalVolume,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_WRITE,
                       GL_R32UI);
}

void FragmentList::bindWriteonly() const
{
    glBindImageTexture(1,
                       m_positionOutputTexture,
                       0,
                       GL_TRUE,
                       0,
                       GL_WRITE_ONLY,
                       GL_R32UI);

    glBindImageTexture(2,
                       mColorVolume,
                       0,
                       GL_TRUE,
                       0,
                       GL_WRITE_ONLY,
                       GL_R32UI);

    glBindImageTexture(3,
                       mNormalVolume,
                       0,
                       GL_TRUE,
                       0,
                       GL_WRITE_ONLY,
                       GL_R32UI);
}

void FragmentList::bindReadonly() const
{
    glBindImageTexture(1,
                       m_positionOutputTexture,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_ONLY,
                       GL_R32UI);

    glBindImageTexture(2,
                       mColorVolume,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_ONLY,
                       GL_R32UI);

    glBindImageTexture(3,
                       mNormalVolume,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_ONLY,
                       GL_R32UI);
}

void FragmentList::bindPosition() const
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
    cudaErrorCheck(cudaGraphicsMapResources(1, &mColorVolumeResource, 0));
    cudaErrorCheck(cudaGraphicsSubResourceGetMappedArray(&mColorVolumeArray, mColorVolumeResource, 0, 0));

    // ### NORMAL
    cudaErrorCheck(cudaGraphicsMapResources(1, &mNormalVolumeResource, 0));
    cudaErrorCheck(cudaGraphicsSubResourceGetMappedArray(&mNormalVolumeArray, mNormalVolumeResource, 0, 0));

}

void FragmentList::unmapFromCUDA()
{
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &m_positionFragmentList, 0));
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &mColorVolumeResource, 0));
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &mNormalVolumeResource, 0));
}

uint1 *FragmentList::getPositionDevPointer()
{
    return m_positionDevPointer;
}

cudaArray* FragmentList::getColorVolumeArray()
{
    return mColorVolumeArray;
}

cudaArray* FragmentList::getNormalVolumeArray()
{
    return mNormalVolumeArray;
}

GLuint FragmentList::getVoxelizationResolution() const
{
    return mVolumeResolution;
}

void FragmentList::createVolumes()
{
    // Color volume
    glGenTextures(1, &mColorVolume);
    glBindTexture(GL_TEXTURE_3D, mColorVolume);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32UI, mVolumeResolution, mVolumeResolution, mVolumeResolution, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL);
    glBindTexture(GL_TEXTURE_3D,0);

    // Normal volume
    glGenTextures(1, &mNormalVolume);
    glBindTexture(GL_TEXTURE_3D, mNormalVolume);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32UI, mVolumeResolution, mVolumeResolution, mVolumeResolution, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL);
    glBindTexture(GL_TEXTURE_3D,0);
}

void FragmentList::deleteVolumes() const
{
    glDeleteTextures(1, &mColorVolume);
    glDeleteTextures(1, &mNormalVolume);
}
