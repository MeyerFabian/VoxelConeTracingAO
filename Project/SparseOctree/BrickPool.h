//
// Created by nils1990 on 03.12.15.
//

#ifndef BRICKPOOL_H
#define BRICKPOOL_H

#include <driver_types.h>
#include "externals/gl3w/include/GL/gl3w.h"
#include "externals/GLFW/include/GLFW/glfw3.h"

class BrickPool
{
public:
    BrickPool();
    ~BrickPool();

    void init(int width = 512, int height = 512, int depth = 512);
    void registerTextureForCUDAWriting();
    void registerTextureForCUDAReading();
    void unregisterTextureForCUDA();

    void voxelizeMaxDetail();
    void subdivideOctree();
    void fillBricks();

private:
    GLuint m_brickPoolID;
    cudaGraphicsResource_t  m_brickPoolRessource;
    cudaArray_t m_brickPoolArray;

    void mapRessourceToArray();
    void unmapRessource();
};


#endif //REALTIMERENDERING_BRICKPOOL_H
