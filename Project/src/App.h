#ifndef APP_H_
#define APP_H_

#include "Controllable.h"

#include "SparseOctree/SparseVoxelOctree.h"
#include "SparseOctree/BrickPool.h"
#include "SparseOctree/NodePool.h"
#include "Scene/Scene.h"
#include "Voxelization/Voxelization.h"
#include "OctreeRaycaster/OctreeRaycaster.h"
#include "Rendering/VoxelConeTracing.h"
#include "Rendering/LightViewMap.h"
#include "Rendering/FullScreenQuad.h"
#include "PointCloud/PointCloud.h"
#include "VoxelCubes/VoxelCubes.h"
#include "Utilities/enums.h"
#include "externals/gl3w/include/GL/gl3w.h"
#include "externals/GLFW/include/GLFW/glfw3.h"
#include "externals/GLM/glm/glm.hpp"

#include <vector>
#include <memory>
#include <src/Rendering/SSR.h>

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

    void handleCamera(GLfloat deltaTime);
    void voxelizeAndFillOctree();

    GLFWwindow* m_pWindow;
    GLfloat m_prevTime;

    std::vector<Controllable*> m_controllables;
    std::unique_ptr<SparseVoxelOctree> m_upSVO;
    std::unique_ptr<Scene> m_upScene;
    std::unique_ptr<Voxelization> m_upVoxelization;
    std::unique_ptr<FragmentList> m_upFragmentList;
    std::unique_ptr<OctreeRaycaster> m_upOctreeRaycaster;
    std::unique_ptr<VoxelConeTracing> m_upVoxelConeTracing;
    std::unique_ptr<LightViewMap> m_upLightViewMap;
    std::unique_ptr<FullScreenQuad> m_upFullScreenQuad;
    std::unique_ptr<PointCloud> m_upPointCloud;
    std::unique_ptr<VoxelCubes> m_upVoxelCubes;
    std::unique_ptr<SSR> m_upSSR;

    bool m_voxeliseEachFrame;
    bool m_showGBuffer;
};

#endif // APP_H_
