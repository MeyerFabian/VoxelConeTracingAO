//
// Created by nils1990 on 28.12.15.
//

#include "FragmentList.h"
#include <assert.h>

FragmentList::FragmentList(GLuint maxListSize) : mVoxelCount(0)
{
    init(maxListSize);
}

FragmentList::~FragmentList()
{

}

void FragmentList::init(GLuint maxListSize)
{
    // Color buffer
    glGenBuffers(1, &mColorOutputBuffer);
    glBindBuffer(GL_TEXTURE_BUFFER, mColorOutputBuffer);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(GLubyte) * 4 * maxListSize, 0, GL_DYNAMIC_DRAW);

    // Color texture
    glGenTextures(1, &mColorOutputTexture);

    glBindBuffer(GL_TEXTURE_BUFFER, 0);
}

void FragmentList::bind()
{
    glActiveTexture(GL_TEXTURE1); // 0 probably used for diffuse texture for texture mapping
    glBindTexture(GL_TEXTURE_1D, mColorOutputTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA8, mColorOutputBuffer);
    glBindImageTexture(1,
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
