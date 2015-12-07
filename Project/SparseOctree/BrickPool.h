//
// Created by nils1990 on 03.12.15.
//

#ifndef BRICKPOOL_H
#define BRICKPOOL_H

#include <driver_types.h>
#include <vector_types.h>
#include "externals/gl3w/include/GL/gl3w.h"
#include "externals/GLFW/include/GLFW/glfw3.h"
#include "NodePool.h"

class BrickPool
{
public:
    BrickPool();
    ~BrickPool();

	void init(int width = 384, int height = 384, int depth = 384);
    void registerTextureForCUDAWriting();
    void registerTextureForCUDAReading();
    void unregisterTextureForCUDA();

    cudaArray_t *fillBrickPool(const NodePool &nodepool);

private:
    GLuint m_brickPoolID;
    cudaGraphicsResource_t  m_brickPoolRessource;
    cudaArray_t m_brickPoolArray;

	dim3 m_poolSize;
};


#endif //REALTIMERENDERING_BRICKPOOL_H
