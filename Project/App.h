#ifndef APP_H_
#define APP_H_

#include "Controllable.h"

#include "externals/gl3w/include/GL/gl3w.h"
#include "externals/GLFW/include/GLFW/glfw3.h"
#include "externals/GLM/glm/glm.hpp"

#include <vector>
#include <memory>

#include "SparseOctree/SparseVoxelOctree.h"
#include "Scene/Scene.h"

#include "SparseOctree/BrickPool.h"
#include "SparseOctree/NodePool.h"
#include "rendering/ShaderProgram.h"

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
    GLint mPrevWidth = 0;
    GLint mPrevHeight = 0;

    std::vector<Controllable*> mControllables; // Could be weak pointers
    std::unique_ptr<SparseVoxelOctree> m_svo;
    std::unique_ptr<Scene> m_scene;

    // Testing
    glm::mat4 uniformView;
    glm::mat4 uniformProjection;
    glm::mat4 uniformModel;
    GLuint vertexArrayID;
    ShaderProgram* pSimpleShader;
};

#endif // APP_H_
