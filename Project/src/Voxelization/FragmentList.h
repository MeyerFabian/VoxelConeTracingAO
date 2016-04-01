/*
 * The fragmentlist consits of 3 texturebuffers which contain the information needed by voxel cone tracing (color, normal, position).
 * furthermore the list is able to map its resources to cuda, allowing easy interoperability.
 * One might consider using multiple instances of this class to group geometry in the scene as static/dynamic
 * */

#ifndef FRAGMENT_LIST_H
#define FRAGMENT_LIST_H

#include "externals/gl3w/include/GL/gl3w.h"
#include "externals/GLFW/include/GLFW/glfw3.h"

#include <driver_types.h>
#include <vector_types.h>

class FragmentList
{
    friend class Voxelization;  // setVoxelCount should only be called by Voxelization
public:

    FragmentList(GLuint voxelizationResolution, GLuint maxListSize = 20000000);
    ~FragmentList();

    void reset();

    void bind() const;
    void bindWriteonly() const;
    void bindReadonly() const;
    void bindPosition() const;
    int getVoxelCount() const;

    void mapToCUDA();
    void unmapFromCUDA();

    uint1* getPositionDevPointer() const;
    cudaArray* getColorVolumeArray() const;
    cudaArray* getNormalVolumeArray() const;

    GLuint getVoxelizationResolution() const;

private:

    void setVoxelCount(int count);

    void createVolumes();
    void deleteVolumes() const;

    // 32bit, 10 per axis, 2 unused
    GLuint m_positionOutputBuffer;
    GLuint m_positionOutputTexture;

    // 32bit uint for RGB colors with 8 bit per channel (alpha used for blending)
    GLuint m_colorVolume;
    GLuint m_normalVolume;

    int m_voxelCount;
    size_t m_maxListSize;

    // Position for cuda
    cudaGraphicsResource_t m_positionFragmentList;
    uint1 *m_positionDevPointer;

    // Color for cuda
    cudaGraphicsResource_t m_colorVolumeResource;
    cudaArray *m_colorVolumeArray;

    // Normal for cuda
    cudaGraphicsResource_t m_normalVolumeResource;
    cudaArray *m_normalVolumeArray;

    GLuint m_volumeResolution;
};

#endif //FRAGMENTLIST_H
