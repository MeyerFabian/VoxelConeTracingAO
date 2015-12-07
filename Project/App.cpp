#include "App.h"

#include "externals/ImGui/imgui.h"
#include "externals/ImGui/imgui_impl_glfw_gl3.h"

#include <iostream>


// GLFW callback for errors
static void errorCallback(int error, const char* description)
{
    std::cout << error << " " << description << std::endl;
}

App::App()
{
    int width = 800;
    int height = 600;

    // Initialize GLFW and OpenGL
    glfwSetErrorCallback(errorCallback);
    if (!glfwInit())
        exit(EXIT_FAILURE);

    mpWindow = glfwCreateWindow(width, height, "VoxelConeTracing", NULL, NULL);
    if (!mpWindow)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(mpWindow);
    gl3wInit();

    // OpenGL initialization
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1);
    glEnable(GL_TEXTURE_1D);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_TEXTURE_3D);
    glEnable(GL_CULL_FACE);

    // Init ImGui
    ImGui_ImplGlfwGL3_Init(mpWindow, true);

    // Load fonts
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontDefault();

    // Variables for the loop
    mPrevTime = (GLfloat)glfwGetTime();
    mPrevWidth = 0;
    mPrevHeight = 0;

    // Scene
    //m_scene = std::unique_ptr<Scene>(new Scene(this, std::string(MESHES_PATH) + "/sponza.obj"));

    // Voxelization
    m_svo = std::unique_ptr<SparseVoxelOctree>(new SparseVoxelOctree(this));

    m_svo->init();
}

App::~App()
{
    glfwDestroyWindow(mpWindow);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}

void App::run()
{
    // Loop
    while (!glfwWindowShouldClose(mpWindow))
    {
        // Calc time per frame
        GLfloat currentTime = (GLfloat)glfwGetTime();
        GLfloat deltaTime = currentTime - mPrevTime;
        mPrevTime = currentTime;

        // Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // ImGui new frame
        ImGui_ImplGlfwGL3_NewFrame();

        // Get window resolution
        GLint width, height;
        glfwGetWindowSize(mpWindow, &width, &height);

        // Give OpenGL the window resolution
        if (width != mPrevWidth || height != mPrevHeight)
        {
            glViewport(0, 0, width, height);
            mPrevWidth = width;
            mPrevHeight = height;
        }

        m_svo->updateOctree();

        // Draw scene
        //m_scene->draw();

        // Update all controllables
        for(Controllable* pControllable : mControllables)
        {
            pControllable->updateGui();
        }

        // Global gui
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

        // Render ImGui (that what is defined by controllables)
        ImGui::Render();

        // Prepare next frame
        glfwSwapBuffers(mpWindow);
        glfwPollEvents();
    }
}

void App::registerControllable(Controllable* pControllable)
{
    mControllables.push_back(pControllable);
}
