/* Raycaster which visualizes the sparse voxel octree. */

#ifndef OCTREE_RAYCASTER_H_
#define OCTREE_RAYCASTER_H_

#include "src/Rendering/ShaderProgram.h"
#include "src/Scene/Camera.h"
#include "src/SparseOctree/NodePool.h"
#include "src/Rendering/GBuffer.h"
#include "src/SparseOctree/BrickPool.h"
#include "src/Controllable.h"
#include "externals/gl3w/include/GL/gl3w.h"

#include <memory>

class OctreeRaycaster : public Controllable
{
public:

    // Constructor
    OctreeRaycaster(App* pApp);

    // Methods
    void draw(glm::vec3 camPos,
        NodePool& nodePool,
        BrickPool& brickPool,
        std::unique_ptr<GBuffer>& gbuffer,
        GLuint screenQuad,
        float volumeExtent) const;
    void fillGui();

private:

    // Members
    std::unique_ptr<ShaderProgram> m_upOctreeRaycasterShader;
    GLuint m_VAO;
    float m_stepSize;
    float m_directionBeginScale;
    int m_maxSteps;
    int m_maxLevel;
};


#endif // OCTREE_RAYCASTER_H_
