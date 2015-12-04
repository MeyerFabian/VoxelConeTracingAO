#ifndef APP_H_
#define APP_H_

#include "Controllable.h"
#include "Voxelization.h"

#include "externals/gl3w/include/GL/gl3w.h"
#include "externals/GLFW/include/GLFW/glfw3.h"
#include "externals/GLM/glm/glm.hpp"

#include <vector>
#include <memory>

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
    std::unique_ptr<Voxelization> mupVoxelization;

    // Testing
    glm::mat4 uniformView;
    glm::mat4 uniformProjection;
    glm::mat4 uniformModel;
    GLuint vertexArrayID;
    ShaderProgram* pSimpleShader;


    BrickPool m_brickPool;
    NodePool m_nodePool;
};

#endif // APP_H_
