//
// Created by nils1990 on 03.12.15.
//

#ifndef NODEPOOL_H
#define NODEPOOL_H

#include "externals/GLM/glm/glm.hpp"
#include <driver_types.h>
#include <vector_types.h>
#include "externals/gl3w/include/GL/gl3w.h"
#include "externals/GLFW/include/GLFW/glfw3.h"


struct node
{
    unsigned int nodeTilePointer;  // Points to the next node tile or marks as max. Furthermore dertermines the type of value
    // first bit: 1 => maximum touched  0 => not touched
    // second bit: 1 => interpret value as constant color   2: => interpret value as brick pointer
    unsigned int value;    // encodes the pointer to the brick or represents a constant color. encoding works as follows:
    // in case of constant value: RGBA8 color
    // in case of pointer: first two bits not used. last 30 bits are a X,Y,Z coordinate to the assigned brick (10 bit per channel)
};

class NodePool
{
public:
    NodePool(){}
    ~NodePool();
    void init(int nodeCount = 1000000);
                                         // copying global to const memory before traversal might improve the performance
    void updateConstMemory();
    void fillNodePool(uchar4* colorBufferDevPointer);
    void clearNodePool();

    void mapToCUDA();
    void unmapFromCUDA();
    void bind();

    int getPoolSize();
    node *getNodePoolDevicePointer();

private:
    int m_poolSize;
    node *m_hNodePool; // host representation of the node pool => initialised once at the beginning of the program
                       // i decided against thrust to make things easier with constant memory mapping

    node *m_dNodePool;

    cudaGraphicsResource_t  mNodePoolFragmentList;
    GLuint mNodePoolOutputBuffer;
    GLuint mNodePoolOutputTexture;
};


#endif //REALTIMERENDERING_NODEPOOL_H
