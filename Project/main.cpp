extern "C" // this is not necessary imho, but gives a better idea on where the function comes from
{
    void test();    // test.cuh
}

#include <iostream>
#include <assimp/scene.h>
#include "externals/OpenGLLoader/gl_core_4_3.h"
#include "externals/GLFW/include/GLFW/glfw3.h"
#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/picoPNG/picopng.h"
#include "externals/ImGui/imgui.h"
#include "externals/ImGui/imgui_impl_glfw_gl3.h"

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

    // Init Cuda and enable OpenGL interop
    auto error = cudaGLSetGLDevice(0);

    // Test if a Cuda capable device is available
    if(error != static_cast<cudaError>(CUDA_SUCCESS))
    {
        std::cerr << "no cuda capable device" << std::endl;
    }

    // Init ImGui
    ImGui_ImplGlfwGL3_Init(pWindow, true);

    // Values for ImGui
    bool show_test_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImColor(114, 144, 154);

    // Load Fonts
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontDefault();
    io.Fonts->AddFontFromFileTTF("externals/ImGui/extra_fonts/Cousine-Regular.ttf", 15.0f);

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

        // ImGui new frame
        ImGui_ImplGlfwGL3_NewFrame();

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
