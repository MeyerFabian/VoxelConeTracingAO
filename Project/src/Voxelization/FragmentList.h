/* List of positions of fragment voxels. 3D textures save averaged color and
 normal. */

#ifndef FRAGMENT_LIST_H
#define FRAGMENT_LIST_H

#include "externals/gl3w/include/GL/gl3w.h"
#include "externals/GLFW/include/GLFW/glfw3.h"

#include <driver_types.h>
#include <vector_types.h>

class FragmentList
{
    friend class Voxelization; // setVoxelCount should only be called by Voxelization
public:

    // Constructor
    FragmentList(GLuint voxelizationResolution, GLuint maxListSize = 20000000);

    // Destructor
    ~FragmentList();

    // Methods
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

    // Methods
    void setVoxelCount(int count);
    void createVolumes();
    void deleteVolumes() const;

    // 32bit, 10 per axis, 2 unused
    GLuint m_positionOutputBuffer;
    GLuint m_positionOutputTexture;

    // 32bit uint for RGB colors with 8 bit per channel (alpha used for blending)
    GLuint m_colorVolume;
    GLuint m_normalVolume;

    // Position for cuda
    cudaGraphicsResource_t m_positionFragmentList;
    uint1 *m_positionDevPointer;

    // Color for cuda
    cudaGraphicsResource_t m_colorVolumeResource;
    cudaArray *m_colorVolumeArray;

    // Normal for cuda
    cudaGraphicsResource_t m_normalVolumeResource;
    cudaArray *m_normalVolumeArray;

    // Members
    int m_voxelCount;
    size_t m_maxListSize;
    GLuint m_volumeResolution;
};

#endif //FRAGMENT_LIST_H
