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

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

class BrickPool
{
public:
    BrickPool();
    ~BrickPool();

	void init(int width = 384, int height = 384, int depth = 384);
    void registerTextureForCUDAWriting();
    void registerTextureForCUDAReading();
    void unregisterTextureForCUDA();
	void mapToCUDA();
	void unmapFromCUDA();
	void bind();
	const dim3& getResolution();

	cudaArray *getBrickPoolArray();
private:
    GLuint m_brickPoolID;
    cudaGraphicsResource_t  m_brickPoolRessource;
    cudaArray *m_brickPoolArray;

	dim3 m_poolSize;
};


#endif //REALTIMERENDERING_BRICKPOOL_H
