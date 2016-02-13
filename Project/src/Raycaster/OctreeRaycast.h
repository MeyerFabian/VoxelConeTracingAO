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
#include <src/SparseOctree/BrickPool.h>
#include "Controllable.h"

class OctreeRaycast : public Controllable{
public:
    OctreeRaycast(App* pApp);
    void draw(
        glm::vec3 camPos,
        NodePool& nodePool,
        BrickPool& brickPool,
		std::unique_ptr<GBuffer>& gbuffer, 
		GLuint ScreenQuad,
        float volumeExtent) const;
        void fillGui();

private:
    std::unique_ptr<ShaderProgram> mupOctreeRaycastShader;
    GLuint vaoID;
    float stepSize;
    float directionBeginScale;
    int maxSteps;
};


#endif //REALTIMERENDERING_MINECRAFT_H
