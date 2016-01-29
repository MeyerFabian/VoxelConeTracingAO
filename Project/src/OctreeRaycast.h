//
// Created by miland on 16.01.16.
//

#ifndef REALTIMERENDERING_MINECRAFT_H
#define REALTIMERENDERING_MINECRAFT_H

#include <memory>
#include "externals/gl3w/include/GL/gl3w.h"
#include <src/Rendering/ShaderProgram.h>
#include <src/Scene/Camera.h>
#include <src/SparseOctree/NodePool.h>
#include <src/Rendering/GBuffer.h>

class OctreeRaycast {
public:
    OctreeRaycast();
    void draw(glm::vec3 camPos, NodePool& nodePool, std::unique_ptr<GBuffer>& gbuffer, float stepSize) const;
private:
    std::unique_ptr<ShaderProgram> mupOctreeRaycastShader;
    GLuint vaoID;
};


#endif //REALTIMERENDERING_MINECRAFT_H
