//
// Created by nils1990 on 03.12.15.
//

#ifndef NODEPOOL_H
#define NODEPOOL_H

#include "externals/GLM/glm/glm.hpp"

struct node // 8 byte per node
{
    int nodeTilePointer;  // Points to the next node tile or marks as max. Furthermore dertermines the type of value
    // first bit: 1 => maximum subdivision reached  0 => has children
    // second bit: 1 => interpret value as constant color   2: => interpret value as brick pointer
    int value;    // encodes the pointer to the brick or represents a constant color. encoding works as follows:
    // in case of constant value: RGBA8 color
    // in case of pointer: first two bits not used. last 30 bits are a X,Y,Z coordinate to the assigned brick (10 bit per channel)
};

class NodePool
{
public:
    NodePool(){}
    void init(int nodeCount = 8192); // i chose 8192 without any practical experience => maximum count that fits in constant memory
                                         // copying global to const memory before traversal might improve the performance

    void updateConstMemory();
private:
    node *m_hNodePool; // host representation of the node pool => initialised once at the beginning of the program
                       // i decided against thrust to make things easier with constant memory mapping
};


#endif //REALTIMERENDERING_NODEPOOL_H
