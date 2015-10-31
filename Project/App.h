#ifndef APP_H_
#define APP_H_

#include "externals/gl3w/include/GL/gl3w.h"
#include "externals/GLFW/include/GLFW/glfw3.h"
#include "externals/GLM/glm/glm.hpp"
#include "rendering/ShaderProgram.h"

class App
{
public:

    App();
    virtual ~App(); // Virtual not necessary
    void run();

private:

    GLFWwindow* mpWindow;
    GLfloat mPrevTime;
    GLint mPrevWidth = 0;
    GLint mPrevHeight = 0;

    // Testing
    glm::mat4 uniformView;
    glm::mat4 uniformProjection;
    glm::mat4 uniformModel;
    GLuint vertexArrayID;
    ShaderProgram* pSimpleShader;

};

#endif //APP_H_
