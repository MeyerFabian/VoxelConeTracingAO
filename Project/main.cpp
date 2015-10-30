extern "C" // this is not necessary imho, but gives a better idea on where the function comes from
{
    void test();    // test.cuh
}

#include <iostream>
#include <assimp/scene.h>
#include "External/OpenGLLoader/gl_core_4_3.h"
#include "External/GLFW/include/GLFW/glfw3.h"
#include "External/GLM/glm/glm.hpp"
#include "External/GLM/glm/gtc/matrix_transform.hpp"
#include "External/picoPNG/picopng.h"

// has to be included after opengl
#include <cuda.h>
#include <cuda_gl_interop.h>

// GLFW callback for errors
static void errorCallback(int error, const char* description)
{
    std::cout << error << " " << description << std::endl;
}

// Main
int main(void)
{
    std::cout << "hello from gcc" << std::endl;
    test();

    int width = 800;
    int height = 600;

    // Initialize GLFW and OpenGL
    GLFWwindow* pWindow;
    glfwSetErrorCallback(errorCallback);
    if (!glfwInit())
        exit(EXIT_FAILURE);

    pWindow = glfwCreateWindow(width, height, "VoxelConeTracing", NULL, NULL);
    if (!pWindow)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(pWindow);
    ogl_LoadFunctions();

    // OpenGL initialization
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1);
    glEnable(GL_TEXTURE_1D);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_TEXTURE_3D);
    glEnable(GL_CULL_FACE);

    // init cuda and enable opengl interop
    auto error = cudaGLSetGLDevice(0);

    // test if a cuda capable device is available
    if(error != static_cast<cudaError>(CUDA_SUCCESS))
    {
        std::cerr << "no cuda capable device" << std::endl;
    }

    // Variables for the loop
    GLfloat prevTime = (GLfloat)glfwGetTime();
    GLfloat deltaTime;
    GLint prevWidth = 0;
    GLint prevHeight = 0;

    glm::mat4 uniformView;
    glm::mat4 uniformProjection = glm::perspective(glm::radians(35.0f), ((GLfloat)width / (GLfloat)height), 0.1f, 100.f);

    // Loop
    while (!glfwWindowShouldClose(pWindow))
    {
        // Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Get window resolution
        GLint width, height;
        glfwGetWindowSize(pWindow, &width, &height);

        // Give OpenGL the window resolution
        if (width != prevWidth || height != prevHeight)
        {
            glViewport(0, 0, width, height);
            prevWidth = width;
            prevHeight = height;
        }

        // Calc time per frame
        GLfloat currentTime = (GLfloat)glfwGetTime();
        deltaTime = currentTime - prevTime;
        prevTime = currentTime;


        // Prepare next frame
        glfwSwapBuffers(pWindow);
        glfwPollEvents();

        //std::cout << "\r" << "FPS: " << (int)(1.0f / deltaTime);
    }

    // Termination
    glfwDestroyWindow(pWindow);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}
