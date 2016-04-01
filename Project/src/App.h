/* App contains initialization of all major objects and render loop. Initializes
also OpenGL */

#ifndef APP_H_
#define APP_H_

#include "Controllable.h"

#include "src/SparseOctree/SparseVoxelOctree.h"
#include "src/SparseOctree/BrickPool.h"
#include "src/SparseOctree/NodePool.h"
#include "src/Scene/Scene.h"
#include "src/Voxelization/Voxelization.h"
#include "src/OctreeRaycaster/OctreeRaycaster.h"
#include "src/Rendering/VoxelConeTracing.h"
#include "src/Rendering/LightViewMap.h"
#include "src/Rendering/FullScreenQuad.h"
#include "src/PointCloud/PointCloud.h"
#include "src/VoxelCubes/VoxelCubes.h"
#include "src/Utilities/enums.h"
#include "src/Rendering/SSR.h"
#include "externals/gl3w/include/GL/gl3w.h"
#include "externals/GLFW/include/GLFW/glfw3.h"
#include "externals/GLM/glm/glm.hpp"

#include <vector>
#include <memory>


// Available visualizations
enum Visualization { RAYCASTING, VOXEL_CUBES, POINT_CLOUD, GBUFFER, PHONG, AMBIENT_OCCLUSION, VOXEL_CONE_TRACING, SHADOW_MAP,VOXEL_GLOW };

class App: public Controllable
{
public:

    // Some constants
    const float DYNAMIC_OBJECT_SPEED = 10.0f;

    // Constructor / Destructor
    App();
    virtual ~App();

    // Methods
    void run();
    void registerControllable(Controllable* pControllable);
    void fillGui();

private:

    // Methods
    void handleCamera(GLfloat deltaTime);
    void voxelizeAndFillOctree();

    // Members
    GLFWwindow* m_pWindow;
    GLfloat m_prevTime;
    std::vector<Controllable*> m_controllables;
    std::unique_ptr<SparseVoxelOctree> m_upSVO;
    std::unique_ptr<Scene> m_upScene;
    std::unique_ptr<Voxelization> m_upVoxelization;
    std::unique_ptr<OctreeRaycaster> m_upOctreeRaycaster;
    std::unique_ptr<VoxelConeTracing> m_upVoxelConeTracing;
    std::unique_ptr<LightViewMap> m_upLightViewMap;
    std::unique_ptr<FullScreenQuad> m_upFullScreenQuad;
    std::unique_ptr<PointCloud> m_upPointCloud;
    std::unique_ptr<VoxelCubes> m_upVoxelCubes;
    std::unique_ptr<SSR> m_upSSR;
    bool m_voxelizeEachFrame;
    bool m_showGBuffer;
};

#endif // APP_H_
