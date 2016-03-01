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

    FragmentList(GLuint maxListSize = 20000000);
    ~FragmentList();

    void bind();
    void bindWriteonly();
    void bindReadonly();
    void bindPosition();
    int getVoxelCount() const;

    void mapToCUDA();
    void unmapFromCUDA();

    uint1* getPositionDevPointer();
    uchar4* getColorBufferDevPointer();
    uchar4* getNormalDevPointer();

private:

    void setVoxelCount(int count);

    // 32bit, 10 per axis, 2 unused
    GLuint m_positionOutputBuffer;
    GLuint m_positionOutputTexture;

    // RGBA 8
    GLuint m_normalOutputBuffer;
    GLuint m_normalOutputTexture;

    // RGBA8
    GLuint m_colorOutputBuffer;
    GLuint m_colorOutputTexture;

    int m_voxelCount;
    size_t m_maxListSize;

    cudaGraphicsResource_t  m_positionFragmentList;
    uint1 *m_positionDevPointer;

    cudaGraphicsResource_t  m_colorFragmentList;
    uchar4 *m_colorDevPointer;

    cudaGraphicsResource_t  m_normalFragmentList;
    uchar4 *m_normalDevPointer;
};

#endif //FRAGMENTLIST_H
