//
// Created by nils1990 on 28.12.15.
//


#ifndef FRAGMENTLIST_H
#define FRAGMENTLIST_H

#include <driver_types.h>
#include <vector_types.h>
#include "externals/gl3w/include/GL/gl3w.h"
#include "externals/GLFW/include/GLFW/glfw3.h"

/*
 * The fragmentlist consits of 3 texturebuffers which contain the information needed by voxel cone tracing (color, normal, position).
 * furthermore the list is able to map its resources to cuda, allowing easy interoperability.
 * One might consider using multiple instances of this class to group geometry in the scene as static/dynamic
 * */
class FragmentList
{
    friend class Voxelization;  // setVoxelCount should only be called by Voxelization
public:
    FragmentList(GLuint maxListSize = 9000000); //8000000);
    ~FragmentList();

    void init(GLuint maxListSize);

    void bind();
    int getVoxelCount() const;

    void mapToCUDA();
    void unmapFromCUDA();

    uint1* getPositionDevPointer();
    uchar4* getColorBufferDevPointer();
    uchar4* getNormalDevPointer();

private:

    // 32bit, 10 per axis, 2 unused
    GLuint mPositionOutputBuffer;
    GLuint mPositionOutputTexture;

    // RGBA 8
    GLuint mNormalOutputBuffer;
    GLuint mNormalOutputTexture;

    // RGBA8
    GLuint mColorOutputBuffer;
    GLuint mColorOutputTexture;

    int mVoxelCount;
    size_t  mMaxListSize;
    void setVoxelCount(int count);

    cudaGraphicsResource_t  mPositionFragmentList;
    uint1 *mPositionDevPointer;

    cudaGraphicsResource_t  mColorFragmentList;
    uchar4 *mColorDevPointer;

    cudaGraphicsResource_t  mNormalFragmentList;
    uchar4 *mNormalDevPointer;
};


#endif //FRAGMENTLIST_H
