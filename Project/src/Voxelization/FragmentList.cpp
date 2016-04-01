#include "FragmentList.h"

#include "Utilities/errorUtils.h"

#include <assert.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

FragmentList::FragmentList(GLuint voxelizationResolution, GLuint maxListSize)
{
    m_volumeResolution = voxelizationResolution;
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

    // POSITION
    cudaErrorCheck(cudaGraphicsGLRegisterBuffer(&m_positionFragmentList,m_positionOutputBuffer,cudaGraphicsMapFlagsReadOnly));
    cudaErrorCheck(cudaGraphicsMapResources(1, &m_positionFragmentList, 0));

    size_t sizePosition = m_maxListSize * sizeof(GLuint);
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&m_positionDevPointer,
                                                        &sizePosition, m_positionFragmentList));

    cudaGraphicsUnmapResources(1, &m_positionFragmentList, 0);

    glBindBuffer(GL_TEXTURE_BUFFER, 0);

    // Volumes for color and normal
    createVolumes();

    // COLOR
    cudaErrorCheck(cudaGraphicsGLRegisterImage(&m_colorVolumeResource, m_colorVolume, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsReadOnly));
    cudaErrorCheck(cudaGraphicsMapResources(1, &m_colorVolumeResource, 0));
    cudaErrorCheck(cudaGraphicsSubResourceGetMappedArray(&m_colorVolumeArray, m_colorVolumeResource, 0, 0));
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &m_colorVolumeResource, 0));

    // NORMAL
    cudaErrorCheck(cudaGraphicsGLRegisterImage(&m_normalVolumeResource, m_normalVolume, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsReadOnly));
    cudaErrorCheck(cudaGraphicsMapResources(1, &m_normalVolumeResource, 0));
    cudaErrorCheck(cudaGraphicsSubResourceGetMappedArray(&m_normalVolumeArray, m_normalVolumeResource, 0, 0));
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &m_normalVolumeResource, 0));
}

FragmentList::~FragmentList()
{
    // Position
    cudaGraphicsUnregisterResource(m_positionFragmentList);
    cudaGLUnregisterBufferObject(m_positionOutputBuffer);

    glDeleteTextures(1, &m_positionOutputTexture);
    glDeleteBuffers(1, &m_positionOutputBuffer);

    // Color and normal
    cudaGraphicsUnregisterResource(m_colorVolumeResource);
    cudaGraphicsUnregisterResource(m_normalVolumeResource);

    // Delete volume textures
    deleteVolumes();
}

void FragmentList::reset()
{
    GLuint clearInt = 0;
    glClearTexImage(m_colorVolume, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &clearInt);
    glClearTexImage(m_normalVolume, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &clearInt);
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
                       m_colorVolume,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_WRITE,
                       GL_R32UI);

    glBindImageTexture(3,
                       m_normalVolume,
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
                       m_colorVolume,
                       0,
                       GL_TRUE,
                       0,
                       GL_WRITE_ONLY,
                       GL_R32UI);

    glBindImageTexture(3,
                       m_normalVolume,
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
                       m_colorVolume,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_ONLY,
                       GL_R32UI);

    glBindImageTexture(3,
                       m_normalVolume,
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
    // POSITION
    cudaErrorCheck(cudaGraphicsMapResources(1, &m_positionFragmentList, 0));

    size_t sizePosition = m_maxListSize * sizeof(GLuint);
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&m_positionDevPointer,
                                                        &sizePosition, m_positionFragmentList));

    // COLOR
    cudaErrorCheck(cudaGraphicsMapResources(1, &m_colorVolumeResource, 0));
    cudaErrorCheck(cudaGraphicsSubResourceGetMappedArray(&m_colorVolumeArray, m_colorVolumeResource, 0, 0));

    // NORMAL
    cudaErrorCheck(cudaGraphicsMapResources(1, &m_normalVolumeResource, 0));
    cudaErrorCheck(cudaGraphicsSubResourceGetMappedArray(&m_normalVolumeArray, m_normalVolumeResource, 0, 0));

}

void FragmentList::unmapFromCUDA()
{
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &m_positionFragmentList, 0));
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &m_colorVolumeResource, 0));
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &m_normalVolumeResource, 0));
}

uint1 *FragmentList::getPositionDevPointer() const
{
    return m_positionDevPointer;
}

cudaArray* FragmentList::getColorVolumeArray() const
{
    return m_colorVolumeArray;
}

cudaArray* FragmentList::getNormalVolumeArray() const
{
    return m_normalVolumeArray;
}

GLuint FragmentList::getVoxelizationResolution() const
{
    return m_volumeResolution;
}

void FragmentList::createVolumes()
{
    // Color volume
    glGenTextures(1, &m_colorVolume);
    glBindTexture(GL_TEXTURE_3D, m_colorVolume);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32UI, m_volumeResolution, m_volumeResolution, m_volumeResolution, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL);
    glBindTexture(GL_TEXTURE_3D,0);

    // Normal volume
    glGenTextures(1, &m_normalVolume);
    glBindTexture(GL_TEXTURE_3D, m_normalVolume);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32UI, m_volumeResolution, m_volumeResolution, m_volumeResolution, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL);
    glBindTexture(GL_TEXTURE_3D,0);
}

void FragmentList::deleteVolumes() const
{
    glDeleteTextures(1, &m_colorVolume);
    glDeleteTextures(1, &m_normalVolume);
}
