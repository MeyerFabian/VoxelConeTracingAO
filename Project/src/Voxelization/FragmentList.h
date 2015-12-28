//
// Created by nils1990 on 28.12.15.
//


#ifndef FRAGMENTLIST_H
#define FRAGMENTLIST_H

#include <driver_types.h>
#include <vector_types.h>
#include "externals/gl3w/include/GL/gl3w.h"
#include "externals/GLFW/include/GLFW/glfw3.h"


class FragmentList
{
    friend class Voxelization;
public:
    FragmentList(GLuint maxListSize = 2750071);
    ~FragmentList();

    void init(GLuint maxListSize);

    void bind();
    int getVoxelCount() const;

private:
    GLuint mColorOutputBuffer;
    GLuint mColorOutputTexture;

    int mVoxelCount;
    void setVoxelCount(int count);

    cudaGraphicsResource_t  mFragmentListResource;
    cudaArray_t mFragmentListArray;
};


#endif //FRAGMENTLIST_H
