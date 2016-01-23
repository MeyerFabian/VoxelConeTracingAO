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

class App
{
public:
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

};

#endif // APP_H_
