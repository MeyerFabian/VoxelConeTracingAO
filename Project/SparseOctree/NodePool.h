//
// Created by nils1990 on 03.12.15.
//

#ifndef NODEPOOL_H
#define NODEPOOL_H

#include <thrust/device_vector.h>
#include "externals/GLM/glm/glm.hpp"

struct node
{
    glm::vec3 const_value;
    glm::vec3 textureBrick;
    node*     child; // todo: read paper ..
};

class NodePool
{
public:
private:
    thrust::device_vector<node> m_nodes;
};


#endif //REALTIMERENDERING_NODEPOOL_H
