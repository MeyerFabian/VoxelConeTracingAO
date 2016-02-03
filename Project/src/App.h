#ifndef APP_H_
#define APP_H_

#include "Controllable.h"

#include "externals/gl3w/include/GL/gl3w.h"
#include "externals/GLFW/include/GLFW/glfw3.h"
#include "externals/GLM/glm/glm.hpp"

#include <vector>
#include <memory>

#include "SparseOctree/SparseVoxelOctree.h"
#include "SparseOctree/BrickPool.h"
#include "SparseOctree/NodePool.h"

#include "Scene/Scene.h"
#include "Voxelization/Voxelization.h"
#include "OctreeRaycast.h"
#include "Rendering/VoxelConeTracing.h"
#include "Rendering/LightViewMap.h"
#include "Rendering/FullScreenQuad.h"
#include "PointCloud/PointCloud.h"

class App
{
public:

    enum class Visualization {RAYCASTING, POINT_CLOUD};

    // Some constants
    const glm::vec3 VOLUME_CENTER = glm::vec3(0, 0, 0);
    const float VOLUME_EXTENT = 384.f;
    const unsigned int VOXELIZATION_RESOLUTION = 512;
    const Visualization VISUALIZATION = Visualization::RAYCASTING;

    App();
    virtual ~App(); // Virtual not necessary
    void run();
    void registerControllable(Controllable* pControllable);

private:

    GLFWwindow* mpWindow;
    GLfloat mPrevTime;

    std::vector<Controllable*> mControllables; // Could be weak pointers
    std::unique_ptr<SparseVoxelOctree> m_svo;
    std::unique_ptr<Scene> m_scene;
    std::unique_ptr<Voxelization> m_voxelization;
    std::unique_ptr<FragmentList> mFragmentList;
    std::unique_ptr<OctreeRaycast> mupOctreeRaycast;
    std::unique_ptr<VoxelConeTracing> m_VoxelConeTracing;
    std::unique_ptr<LightViewMap> m_LightViewMap;
    std::unique_ptr<FullScreenQuad> m_FullScreenQuad;
    std::unique_ptr<PointCloud> m_PointCloud;
};

#endif // APP_H_
