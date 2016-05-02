#include "App.h"

#include <src/Utilities/errorUtils.h>
#include "externals/ImGui/imgui_impl_glfw_gl3.h"

#include <iostream>

// Easier creation of unique pointers
#ifdef __unix__
template<typename T, typename... Args>
std::unique_ptr<T> std::make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
#endif

// Global variables for GLFW callbacks
int visualization = Visualization::RAYCASTING;
float volumeExtent = 384.f;
int width = 1280; // initial width of window
int height = 720; // initial height of window
int mouseX, mouseY = 0;
int deltaCameraYaw = 0;
int deltaCameraPitch = 0;
bool camTurbo = false;
bool moveForwards = false;
bool moveBackwards = false;
bool strafeLeft = false;
bool strafeRight = false;
bool moveUpwards = false;
bool moveDownwards = false;
bool rotateCamera = false;
bool rotateLight = false;
glm::vec3 dynamicObjectDelta = glm::vec3(0,0,0);

// GLFW callback for errors
static void errorCallback(int error, const char* description)
{
    std::cout << error << " " << description << std::endl;
}

// GLFW callback for keys
static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    // Check whether ImGui is handling this
    ImGuiIO& io = ImGui::GetIO();
    if(io.WantCaptureKeyboard)
    {
        camTurbo = false;
        moveForwards = false;
        moveBackwards = false;
        strafeLeft = false;
        strafeRight = false;
        moveUpwards = false;
        moveDownwards = false;
        rotateCamera = false;
        return;
    }

    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, true);
    }

    // ### Camera Handling ###

    // Cam turbo
    if(key == GLFW_KEY_LEFT_SHIFT && action == GLFW_PRESS)
    {
        camTurbo = true;
    }
    if(key == GLFW_KEY_LEFT_SHIFT && action == GLFW_RELEASE)
    {
        camTurbo = false;
    }

    // Move foreward
    if(key == GLFW_KEY_W && action == GLFW_PRESS)
    {
        moveForwards = true;
    }
    if(key == GLFW_KEY_W && action == GLFW_RELEASE)
    {
        moveForwards = false;
    }

    // Strafe Left
    if(key == GLFW_KEY_A&& action == GLFW_PRESS)
    {
        strafeLeft = true;
    }
    if(key == GLFW_KEY_A&& action == GLFW_RELEASE)
    {
        strafeLeft = false;
    }

    // Move Backwards
    if(key == GLFW_KEY_S&& action == GLFW_PRESS)
    {
        moveBackwards = true;
    }
    if(key == GLFW_KEY_S&& action == GLFW_RELEASE)
    {
        moveBackwards = false;
    }

    // Strafe Right
    if(key == GLFW_KEY_D&& action == GLFW_PRESS)
    {
        strafeRight = true;
    }
    if(key == GLFW_KEY_D&& action == GLFW_RELEASE)
    {
        strafeRight = false;
    }

    // Move Up
    if(key == GLFW_KEY_E&& action == GLFW_PRESS)
    {
        moveUpwards = true;
    }
    if(key == GLFW_KEY_E&& action == GLFW_RELEASE)
    {
        moveUpwards = false;
    }

    // Move Down
    if(key == GLFW_KEY_Q&& action == GLFW_PRESS)
    {
        moveDownwards = true;
    }
    if(key == GLFW_KEY_Q&& action == GLFW_RELEASE)
    {
        moveDownwards = false;
    }

    // ### Visualization handling ###

    // Voxel cone tracing
    if(key == GLFW_KEY_1& action == GLFW_PRESS)
    {
        visualization = Visualization::VOXEL_CONE_TRACING;
    }
    //  Raycasting
    if(key == GLFW_KEY_2& action == GLFW_PRESS)
    {
        visualization = Visualization::RAYCASTING;
    }
    //  Voxel cubes
    if(key == GLFW_KEY_3& action == GLFW_PRESS)
    {
        visualization = Visualization::VOXEL_CUBES;
    }
    //  Point cloud
    if(key == GLFW_KEY_4& action == GLFW_PRESS)
    {
        visualization = Visualization::POINT_CLOUD;
    }
    //  Gbuffer
    if(key == GLFW_KEY_5& action == GLFW_PRESS)
    {
        visualization = Visualization::GBUFFER;
    }
    //  Phong
    if(key == GLFW_KEY_6& action == GLFW_PRESS)
    {
        visualization = Visualization::PHONG;
    }
    //  Ambient occlusion
    if(key == GLFW_KEY_7& action == GLFW_PRESS)
    {
        visualization = Visualization::AMBIENT_OCCLUSION;
    }
    //  Shadow map
    if (key == GLFW_KEY_8& action == GLFW_PRESS)
    {
        visualization = Visualization::SHADOW_MAP;
    }
    // Voxel glow
    if (key == GLFW_KEY_9& action == GLFW_PRESS)
    {
        visualization = Visualization::VOXEL_GLOW;
    }

    // ### Dynamic object control ###
    if (key == GLFW_KEY_I & action == GLFW_PRESS)
    {
        dynamicObjectDelta.x = 1;
    }
    if (key == GLFW_KEY_K & action == GLFW_PRESS)
    {
        dynamicObjectDelta.x = -1;
    }
    if (key == GLFW_KEY_J & action == GLFW_PRESS)
    {
        dynamicObjectDelta.z = 1;
    }
    if (key == GLFW_KEY_L & action == GLFW_PRESS)
    {
        dynamicObjectDelta.z = -1;
    }
    if (key == GLFW_KEY_O & action == GLFW_PRESS)
    {
        dynamicObjectDelta.y = 1;
    }
    if (key == GLFW_KEY_U & action == GLFW_PRESS)
    {
        dynamicObjectDelta.y = -1;
    }
    if (key == GLFW_KEY_I & action == GLFW_RELEASE)
    {
        dynamicObjectDelta.x = 0;
    }
    if (key == GLFW_KEY_K & action == GLFW_RELEASE)
    {
        dynamicObjectDelta.x = 0;
    }
    if (key == GLFW_KEY_J & action == GLFW_RELEASE)
    {
        dynamicObjectDelta.z = 0;
    }
    if (key == GLFW_KEY_L & action == GLFW_RELEASE)
    {
        dynamicObjectDelta.z = 0;
    }
    if (key == GLFW_KEY_O & action == GLFW_RELEASE)
    {
        dynamicObjectDelta.y = 0;
    }
    if (key == GLFW_KEY_U & action == GLFW_RELEASE)
    {
        dynamicObjectDelta.y = 0;
    }
}

// GLFW callback for cursor position
static void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    // Check whether ImGui is handling this
    ImGuiIO& io = ImGui::GetIO();
    if(io.WantCaptureMouse)
    {
        deltaCameraYaw = 0;
        deltaCameraPitch = 0;
        return;
    }
    else
    {
        // Update camera
        deltaCameraYaw = 5 * (width/2 - xpos);
        deltaCameraPitch = 5 * (height/2 - ypos);
    }
}

// Ugly workaround for broken CURSOR_HIDDEN
GLFWcursor* BlankCursor()
{
    const int w=1;//16;
    const int h=1;//16;
    unsigned char pixels[w * h * 4];
    memset(pixels, 0x00, sizeof(pixels));
    GLFWimage image;
    image.width = w;
    image.height = h;
    image.pixels = pixels;
    return glfwCreateCursor(&image, 0, 0);
}

// GLFW callback for mouse buttons
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    // Check whether ImGui is handling this
    ImGuiIO& io = ImGui::GetIO();
    if(io.WantCaptureMouse)
    {
        return;
    }
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        deltaCameraPitch = 0;
        deltaCameraYaw = 0;
        glfwSetCursor(window, BlankCursor());
        rotateCamera = true;
    }
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
    {
        glfwSetCursor(window, NULL);
        rotateCamera = false;
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
    {
        deltaCameraPitch = 0;
        deltaCameraYaw = 0;
        glfwSetCursor(window, BlankCursor());
        rotateLight = true;
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
    {
        glfwSetCursor(window, NULL);
        rotateLight = false;
    }
}

App::App() : Controllable("App")
{
    // Initialize members
    m_showGBuffer = false;
    m_voxelizeEachFrame = false;

    // Initialize GLFW and OpenGL
    glfwSetErrorCallback(errorCallback);
    if (!glfwInit())
        exit(EXIT_FAILURE);

    // Create window
    m_pWindow = glfwCreateWindow(width, height, "VoxelConeTracing", NULL, NULL);
    if (!m_pWindow)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Initialize OpenGL
    glfwMakeContextCurrent(m_pWindow);
    gl3wInit();

    // OpenGL setup
    glClearColor(0.0f, 0.0f, 0.0f, 1);

    // Init ImGui
    ImGui_ImplGlfwGL3_Init(m_pWindow, true);
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontDefault();

    // Set GLFW callbacks after ImGui
    glfwSetKeyCallback(m_pWindow, keyCallback);
    glfwSetCursorPosCallback(m_pWindow, cursorPositionCallback);
    glfwSetMouseButtonCallback(m_pWindow, mouseButtonCallback);

    // Variables for the loop
    m_prevTime = (GLfloat)glfwGetTime();

    // Register app as controllable
    this->registerControllable(this);

    // Scene
    m_upScene = std::unique_ptr<Scene>(new Scene(this, "sponza"));

    // Voxelization class (takes polygons and fills voxel fragment list)
    m_upVoxelization = std::unique_ptr<Voxelization>(new Voxelization(this));

    // Visualization of voxel fragments as point cloud
    m_upPointCloud = std::make_unique<PointCloud>(&(m_upScene->getCamera()));

    // Sparse voxel octree takes voxel fragments and builds up octree
    m_upSVO = std::unique_ptr<SparseVoxelOctree>(new SparseVoxelOctree(this));
    m_upSVO->init();

    // Raycaster for visualization of sparse voxel octree
    m_upOctreeRaycaster = std::unique_ptr<OctreeRaycaster>(new OctreeRaycaster(this));

    // Visualization of sparse voxel octree with cubes
    m_upVoxelCubes = std::make_unique<VoxelCubes>(&(m_upScene->getCamera()));

    // Light view map
    m_upLightViewMap = std::make_unique<LightViewMap>(this);
    m_upLightViewMap->init();

    // Voxel cone tracing (does ambient occlusion and global illumination at the moment)
    m_upVoxelConeTracing = std::make_unique<VoxelConeTracing>(this);
    m_upVoxelConeTracing->init(width, height);

    // Some screen spaced reflection testing
    m_upSSR = std::make_unique<SSR>();

    // Many classes use the same fullscreen rendering quad
    m_upFullScreenQuad = std::make_unique<FullScreenQuad>();

    // Voxelize scene and fill octree at least once
    voxelizeAndFillOctree();
}

App::~App()
{
    glfwDestroyWindow(m_pWindow);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}

void App::run()
{
    // Loop
    while (!glfwWindowShouldClose(m_pWindow))
    {
        // Calc time per frame
        GLfloat currentTime = (GLfloat)glfwGetTime();
        GLfloat deltaTime = currentTime - m_prevTime;
        m_prevTime = currentTime;

        // Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // ImGui new frame
        ImGui_ImplGlfwGL3_NewFrame();

        // Get window resolution
        glfwGetWindowSize(m_pWindow, &width, &height);

        // Update camera
        handleCamera(deltaTime);

        // Update light
        if (rotateLight)
        {
            glfwSetCursorPos(m_pWindow, width/2, height/2);
            m_upScene->updateLight(0.01f * deltaCameraYaw * deltaTime, 0.01f * deltaCameraPitch * deltaTime);
        }
        else
        {
            m_upScene->updateLight(0, 0);
        }

        // Update dynamic object
        m_upScene->updateDynamicObject(dynamicObjectDelta * deltaTime * DYNAMIC_OBJECT_SPEED);

        // Voxelization of scene
        if(m_voxelizeEachFrame)
        {
            voxelizeAndFillOctree();
        }

        // Set viewport for scene rendering
        glViewport(0, 0, width, height);

        // Create shadow map
        m_upLightViewMap->shadowMapPass(m_upScene);

        // Fill GBuffes with position and normals etc.
        m_upVoxelConeTracing->geometryPass(width, height, m_upScene);

        // Choose visualization
        switch(visualization)
        {
        case Visualization::RAYCASTING:
            m_upOctreeRaycaster->draw(
                    m_upScene->getCamPos(),
                    m_upSVO->getNodePool(),
                    m_upSVO->getBrickPool(),
                    m_upVoxelConeTracing->getGBuffer(),
                    m_upFullScreenQuad->getvaoID(),
                    volumeExtent);
            break;
        case Visualization::VOXEL_CUBES:
            m_upVoxelCubes->draw(width,height, volumeExtent, m_upSVO->getNodePool(), m_upSVO->getBrickPool());
            break;
        case Visualization::POINT_CLOUD:
            m_upPointCloud->draw(width, height, volumeExtent, m_upVoxelization->getFragmentList());
            break;
        case Visualization::GBUFFER:
            m_showGBuffer = false;
            m_upVoxelConeTracing->drawGBuffer(width, height);
            break;
        case Visualization::PHONG:
            m_upVoxelConeTracing->drawSimplePhong(width, height, m_upLightViewMap->getCurrentShadowMapRes(), m_upFullScreenQuad->getvaoID(), m_upLightViewMap->getDepthTextureID(), m_upScene);
            // mSSR->draw(m_VoxelConeTracing->getGBuffer().get(),m_VoxelConeTracing.get(),m_scene->getCamera().getViewMatrix(),width, height);
            break;
        case Visualization::AMBIENT_OCCLUSION:
            m_upVoxelConeTracing->drawAmbientOcclusion(width, height, m_upFullScreenQuad->getvaoID(), m_upScene, m_upSVO->getNodePool(), m_upSVO->getBrickPool(), volumeExtent);
            break;
        case Visualization::VOXEL_CONE_TRACING:
            m_upVoxelConeTracing->drawVoxelConeTracing(width, height, m_upLightViewMap->getCurrentShadowMapRes(), m_upFullScreenQuad->getvaoID(), m_upLightViewMap->getDepthTextureID(), m_upScene, m_upSVO->getNodePool(), m_upSVO->getBrickPool(), 5, volumeExtent);
            break;
        case Visualization::SHADOW_MAP:
            m_showGBuffer = false;
            m_upVoxelConeTracing->drawVoxelConeTracing(width, height, m_upLightViewMap->getCurrentShadowMapRes(), m_upFullScreenQuad->getvaoID(), m_upLightViewMap->getDepthTextureID(), m_upScene, m_upSVO->getNodePool(), m_upSVO->getBrickPool(), 5, volumeExtent);
            m_upLightViewMap->shadowMapRender(width*0.25, height*0.25, width, height, m_upFullScreenQuad->getvaoID());
            break;
        case Visualization::VOXEL_GLOW:
            m_upVoxelConeTracing->drawVoxelGlow(width, height, m_upFullScreenQuad->getvaoID(), m_upScene, m_upSVO->getNodePool(), m_upSVO->getBrickPool(), volumeExtent);
            break;
        }

        // One can display the gbuffer in addition
        if (m_showGBuffer){
            m_upLightViewMap->shadowMapRender(150, 150, width, height, m_upFullScreenQuad->getvaoID());
            m_upVoxelConeTracing->drawGBufferPanels(width, height);
        }

        // Update all controllables
        bool opened = true;
        ImGui::Begin("Properties", &opened, ImVec2(100, 200));
        for(Controllable* pControllable : m_controllables)
        {
            pControllable->updateGui();
        }
        ImGui::End();

        // Render ImGui (that what is defined by controllables)
        ImGui::Render();

        // Prepare next frame
        glfwSwapBuffers(m_pWindow);
        glfwPollEvents();

    }
}

void App::registerControllable(Controllable* pControllable)
{
    m_controllables.push_back(pControllable);
}

void App::fillGui()
{
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::SliderFloat("VolumeExtent", &volumeExtent, 300.f, 1024.f, "%0.5f");
    ImGui::Checkbox("Voxelize each frame",&m_voxelizeEachFrame);
    ImGui::Combo("Visualization", &visualization, "Raycasting\0VoxelCubes\0PointCloud\0GBuffer\0Phong\0AmbientOcclusion\0VoxelConeTracing\0LightViewMap\0VoxelGlow\0");
    ImGui::Checkbox("Show GBuffer", &m_showGBuffer);
    ImGui::Text("Controls:\n1: Voxel Cone Tracing \n2: Raycasting \n3: Voxel Cubes \n4: Point Cloud \n5: Gbuffer \n6: Phong \n7: Ambient Occlusion \n8: Shadow Map \n9: Voxel Glow");
}

void App::handleCamera(GLfloat deltaTime)
{
    if(camTurbo)
    {
        m_upScene->setCameraSpeed(50.f*deltaTime);
    }
    else
    {
        m_upScene->setCameraSpeed(25.f*deltaTime);
    }
    if(moveForwards)
    {
        m_upScene->updateCamera(FORWARDS, 0, 0);
    }
    if(moveBackwards)
    {
        m_upScene->updateCamera(BACKWARDS, 0, 0);
    }
    if(strafeLeft)
    {
        m_upScene->updateCamera(LEFT, 0, 0);
    }
    if(strafeRight)
    {
        m_upScene->updateCamera(RIGHT, 0, 0);
    }
    if(moveUpwards)
    {
        m_upScene->updateCamera(UP, 0, 0);
    }
    if(moveDownwards)
    {
        m_upScene->updateCamera(DOWN, 0, 0);
    }
    if(rotateCamera)
    {
        glfwSetCursorPos(m_pWindow, width/2, height/2);
        m_upScene->updateCamera(NONE, 0.1f * deltaCameraYaw * deltaTime, 0.1f * deltaCameraPitch * deltaTime);
        deltaCameraPitch = 0;
        deltaCameraYaw = 0;
    }
}

void App::voxelizeAndFillOctree()
{
    m_upVoxelization->voxelize(volumeExtent, m_upScene.get());
    m_upSVO->clearOctree();
    m_upVoxelization->mapFragmentListToCUDA();
    m_upSVO->buildOctree(
                        m_upVoxelization->getFragmentList()->getPositionDevPointer(),
                        m_upVoxelization->getFragmentList()->getColorVolumeArray(),
                        m_upVoxelization->getFragmentList()->getNormalVolumeArray(),
                        m_upVoxelization->getFragmentList()->getVoxelCount(),
                        m_upVoxelization->getFragmentList()->getVoxelizationResolution());
    m_upVoxelization->unmapFragmentListFromCUDA();
}
