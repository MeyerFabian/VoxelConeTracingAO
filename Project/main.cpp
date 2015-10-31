extern "C" // this is not necessary imho, but gives a better idea on where the function comes from
{
    void test();    // test.cuh
}

#include <iostream>
#include <assimp/scene.h>
#include "externals/gl3w/include/GL/gl3w.h"
#include "externals/GLFW/include/GLFW/glfw3.h"
#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/picoPNG/picopng.h"
#include "externals/ImGui/imgui.h"
#include "externals/ImGui/imgui_impl_glfw_gl3.h"

// has to be included after opengl
#include <cuda.h>
#include <cuda_gl_interop.h>

#include "rendering/ShaderProgram.h"

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
    gl3wInit();

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

    // Variables for the loop
    GLfloat prevTime = (GLfloat)glfwGetTime();
    GLfloat deltaTime;
    GLint prevWidth = 0;
    GLint prevHeight = 0;

    glm::mat4 uniformView = glm::lookAt(glm::vec3(0, 0, 5),glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
    glm::mat4 uniformProjection = glm::perspective(glm::radians(35.0f), ((GLfloat)width / (GLfloat)height), 0.1f, 100.f);
    glm::mat4 uniformModel = glm::mat4(1.f);

    // Shader demo TODO: add model loading.. not this stupid triangle

    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    static const GLfloat g_vertex_buffer_data[] = {
            -1.0f, -1.0f, 0.0f,
            1.0f, -1.0f, 0.0f,
            0.0f,  1.0f, 0.0f,
    };

    GLuint vertexbuffer;
    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

    ShaderProgram* simpleShader = new ShaderProgram("/vertex_shaders/modelViewProjection.vert","/fragment_shaders/simpleColor.frag");
    simpleShader->updateUniform("color", glm::vec4(1.0f,0.0f,0.0f,1.0f));

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

        // 1. Show a simple window
        // Tip: if we don't call ImGui::Begin()/ImGui::End() the widgets appears in a window automatically called "Debug"
        {
            static float f = 0.0f;
            ImGui::Text("Hello, world!");
            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
            ImGui::ColorEdit3("clear color", (float*)&clear_color);
            if (ImGui::Button("Test Window")) show_test_window ^= 1;
            if (ImGui::Button("Another Window")) show_another_window ^= 1;
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        }

        // Render ImGui
        ImGui::Render();

        // use our shader
        simpleShader->use();
        simpleShader->updateUniform("color", glm::vec4(clear_color.x, clear_color.y, clear_color.z, clear_color.w));
        simpleShader->updateUniform("projection", uniformProjection);
        simpleShader->updateUniform("view", uniformView);
        simpleShader->updateUniform("model", uniformModel);

        //TODO: load models or draw something fancy
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glVertexAttribPointer(
                0,
                3,
                GL_FLOAT,
                GL_FALSE,
                0,
                (void*)0
        );
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glDisableVertexAttribArray(0);
        // - END TODO


        // Calc time per frame
        GLfloat currentTime = (GLfloat)glfwGetTime();
        deltaTime = currentTime - prevTime;
        prevTime = currentTime;

        // Prepare next frame
        glfwSwapBuffers(pWindow);
        glfwPollEvents();
    }

    // Termination
    delete simpleShader;

    glfwDestroyWindow(pWindow);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}
