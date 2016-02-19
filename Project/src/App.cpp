#include "App.h"

#include "externals/ImGui/imgui_impl_glfw_gl3.h"

#include <iostream>

using namespace std;

#ifdef __unix__
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
#endif

// Ugly static variables
GLint width, height;
int VISUALIZATION = Visualization::RAYCASTING;
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
glm::vec3 dynamicObjectDelta;

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
    // Camera Handling
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

    // Visualitation Handling
    // Voxel cone tracing
    if(key == GLFW_KEY_1& action == GLFW_PRESS)
    {
        VISUALIZATION = Visualization::VOXEL_CONE_TRACING;
    }
    //  Raycasting
    if(key == GLFW_KEY_2& action == GLFW_PRESS)
    {
        VISUALIZATION = Visualization::RAYCASTING;
    }
    //  Voxel Cubes
    if(key == GLFW_KEY_3& action == GLFW_PRESS)
    {
        VISUALIZATION = Visualization::VOXEL_CUBES;
    }
    //  Point Cloud
    if(key == GLFW_KEY_4& action == GLFW_PRESS)
    {
        VISUALIZATION = Visualization::POINT_CLOUD;
    }
    //  Gbuffer
    if(key == GLFW_KEY_5& action == GLFW_PRESS)
    {
        VISUALIZATION = Visualization::GBUFFER;
    }
    //  Phong
    if(key == GLFW_KEY_6& action == GLFW_PRESS)
    {
        VISUALIZATION = Visualization::PHONG;
    }
    //  Ambient occlusion
    if(key == GLFW_KEY_7& action == GLFW_PRESS)
    {
        VISUALIZATION = Visualization::AMBIENT_OCCLUSION;
    }
    //  Shadow Map
    if (key == GLFW_KEY_8& action == GLFW_PRESS)
    {
        VISUALIZATION = Visualization::SHADOW_MAP;
    }
    if (key == GLFW_KEY_9& action == GLFW_PRESS)
    {
        VISUALIZATION = Visualization::VOXEL_GLOW;
    }

    // Dynamic object control
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
        deltaCameraYaw = 5 * (width/2 - xpos);
        deltaCameraPitch = 5 * (height/2 - ypos);
    }
}

// ugly workaround for broken CURSOR_HIDDEN
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
    width = 1024;
    height = 1024;

    mShowGBuffer = false;
    mVoxeliseEachFrame = false;

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

    // Initialize
    glfwMakeContextCurrent(mpWindow);
    gl3wInit();

    // OpenGL initialization
    glClearColor(0.0f, 0.0f, 0.0f, 1);
    glEnable(GL_TEXTURE_1D);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_TEXTURE_3D);

    // Init ImGui
    ImGui_ImplGlfwGL3_Init(mpWindow, true);

    // Set GLFW callbacks after ImGui
    glfwSetKeyCallback(mpWindow, keyCallback);
    glfwSetCursorPosCallback(mpWindow, cursorPositionCallback);
    glfwSetMouseButtonCallback(mpWindow, mouseButtonCallback);

    // Load fonts
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontDefault();

    // Variables for the loop
    mPrevTime = (GLfloat)glfwGetTime();

    // Register app as controllable
    this->registerControllable(this);

    // Scene (load polygon scene)
    m_scene = std::unique_ptr<Scene>(new Scene(this, "sponza"));

    // Voxelization
    m_voxelization = std::unique_ptr<Voxelization>(
        new Voxelization(this));

    mFragmentList = std::unique_ptr<FragmentList>(
            new FragmentList());

    // Sparse voxel octree (use fragment voxels and create octree for later use)

    m_svo = std::unique_ptr<SparseVoxelOctree>(new SparseVoxelOctree(this));

    m_svo->init();

    mupOctreeRaycast = std::unique_ptr<OctreeRaycast>(new OctreeRaycast(this));

    m_LightViewMap = make_unique<LightViewMap>(this);
    m_LightViewMap->init();
    m_VoxelConeTracing = make_unique<VoxelConeTracing>(this);

    m_VoxelConeTracing->init(width, height);
    mSSR = make_unique<SSR>();

    m_FullScreenQuad = make_unique<FullScreenQuad>();

    m_PointCloud = make_unique<PointCloud>(mFragmentList.get(), &(m_scene->getCamera()));

    m_VoxelCubes = make_unique<VoxelCubes>(&(m_scene->getCamera()));

    // create octree from static geometrie
    // Voxelization (create fragment voxels)
    m_voxelization->voxelize(VOLUME_EXTENT, m_scene.get(), mFragmentList.get());

    // Testing fragment list
    //
    m_svo->clearOctree();
    mFragmentList->mapToCUDA();


    //m_svo->updateOctree(mFragmentList->getColorBufferDevPointer());
    m_svo->buildOctree(mFragmentList->getPositionDevPointer(),
                       mFragmentList->getColorBufferDevPointer(),
                       mFragmentList->getNormalDevPointer(),
                       mFragmentList->getVoxelCount());

    mFragmentList->unmapFromCUDA();
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

        //Get window resolution
        glfwGetWindowSize(mpWindow, &width, &height);

        // Update camera
        handleCamera(deltaTime);

        // Update light
        if (rotateLight)
        {
            glfwSetCursorPos(mpWindow, width/2, height/2);
            m_scene->updateLight(0.01f * deltaCameraYaw * deltaTime, 0.01f * deltaCameraPitch * deltaTime);
        }
        else
        {
            m_scene->updateLight(0, 0);
        }

        // Update dynamic object
        m_scene->updateDynamicObject(dynamicObjectDelta * deltaTime * DYNAMIC_OBJECT_SPEED);

        // Voxelization of scene
        if(mVoxeliseEachFrame)
        {
            // Voxelization (create fragment voxels)
            m_voxelization->voxelize(VOLUME_EXTENT, m_scene.get(), mFragmentList.get());


            // Testing fragment list
            //
            m_svo->clearOctree();
            mFragmentList->mapToCUDA();


            //m_svo->updateOctree(mFragmentList->getColorBufferDevPointer());
            m_svo->buildOctree(mFragmentList->getPositionDevPointer(),
                               mFragmentList->getColorBufferDevPointer(),
                               mFragmentList->getNormalDevPointer(),
                               mFragmentList->getVoxelCount());

            mFragmentList->unmapFromCUDA();
        }

        // Set viewport for scene rendering
        glViewport(0, 0, width, height);

        m_LightViewMap->shadowMapPass(m_scene);

        m_VoxelConeTracing->geometryPass(width,height,m_scene);

        // Choose visualization TODO: make this available to user interface
        switch(VISUALIZATION)
        {
        case Visualization::RAYCASTING:
            mupOctreeRaycast->draw(
                    m_scene->getCamPos(),
                    m_svo->getNodePool(),
                    m_svo->getBrickPool(),
                    m_VoxelConeTracing->getGBuffer(),
                    m_FullScreenQuad->getvaoID(),
                    VOLUME_EXTENT);
            break;
        case Visualization::VOXEL_CUBES:
            m_VoxelCubes->draw(width,height, VOLUME_EXTENT, m_svo->getNodePool(), m_svo->getBrickPool());
            break;
        case Visualization::POINT_CLOUD:
            m_PointCloud->draw(width,height, VOLUME_EXTENT);
            break;
        case Visualization::GBUFFER:
            mShowGBuffer = false;
            m_VoxelConeTracing->drawGBuffer(width, height);
            break;
        case Visualization::PHONG:
            m_VoxelConeTracing->drawSimplePhong(width, height, m_LightViewMap->getCurrentShadowMapRes(), m_FullScreenQuad->getvaoID(), m_LightViewMap->getDepthTextureID(), m_scene);
           // mSSR->draw(m_VoxelConeTracing->getGBuffer().get(),m_VoxelConeTracing.get(),m_scene->getCamera().getViewMatrix(),width, height);
                break;
        case Visualization::AMBIENT_OCCLUSION:
            m_VoxelConeTracing->drawAmbientOcclusion(width, height, m_FullScreenQuad->getvaoID(), m_scene, m_svo->getNodePool(), m_svo->getBrickPool(), VOLUME_EXTENT);
            break;
        case Visualization::VOXEL_CONE_TRACING:
            m_VoxelConeTracing->drawVoxelConeTracing(width, height, m_LightViewMap->getCurrentShadowMapRes(), m_FullScreenQuad->getvaoID(), m_LightViewMap->getDepthTextureID(), m_scene, m_svo->getNodePool(), m_svo->getBrickPool(), 5, VOLUME_EXTENT);
            break;
        case Visualization::SHADOW_MAP:
            mShowGBuffer = false;
            m_VoxelConeTracing->drawVoxelConeTracing(width, height, m_LightViewMap->getCurrentShadowMapRes(), m_FullScreenQuad->getvaoID(), m_LightViewMap->getDepthTextureID(), m_scene, m_svo->getNodePool(), m_svo->getBrickPool(), 5, VOLUME_EXTENT);
            m_LightViewMap->shadowMapRender(width*0.25, height*0.25, width, height, m_FullScreenQuad->getvaoID());
            break;
        case Visualization::VOXEL_GLOW:
            m_VoxelConeTracing->drawVoxelGlow(width, height, m_FullScreenQuad->getvaoID(), m_scene, m_svo->getNodePool(), m_svo->getBrickPool(), VOLUME_EXTENT);
            break;
        }

        if (mShowGBuffer){
            m_LightViewMap->shadowMapRender(150, 150, width, height, m_FullScreenQuad->getvaoID());
            m_VoxelConeTracing->drawGBufferPanels(width, height);
        }
        // FUTURE STUFF


        // Update all controllables
        bool opened = true;
        ImGui::Begin("Properties", &opened, ImVec2(100, 200));
        for(Controllable* pControllable : mControllables)
        {
            pControllable->updateGui();
        }
        ImGui::End();

        // Render ImGui (that what is defined by controllables)
        ImGui::Render();

        // Prepare next frame
        glfwSwapBuffers(mpWindow);
        glfwPollEvents();

    }
}

void App::handleCamera(GLfloat deltaTime)
{
    if(camTurbo)
    {
        m_scene->setCameraSpeed(50.f*deltaTime);
    }
    else
    {
        m_scene->setCameraSpeed(25.f*deltaTime);
    }
    if(moveForwards)
    {
        m_scene->updateCamera(FORWARDS, 0, 0);
    }
    if(moveBackwards)
    {
        m_scene->updateCamera(BACKWARDS, 0, 0);
    }
    if(strafeLeft)
    {
        m_scene->updateCamera(LEFT, 0, 0);
    }
    if(strafeRight)
    {
        m_scene->updateCamera(RIGHT, 0, 0);
    }
    if(moveUpwards)
    {
        m_scene->updateCamera(UP, 0, 0);
    }
    if(moveDownwards)
    {
        m_scene->updateCamera(DOWN, 0, 0);
    }
    if(rotateCamera)
    {
        glfwSetCursorPos(mpWindow, width/2, height/2);
        m_scene->updateCamera(NONE, 0.1f * deltaCameraYaw * deltaTime, 0.1f * deltaCameraPitch * deltaTime);
        deltaCameraPitch = 0;
        deltaCameraYaw = 0;
    }
}

void App::registerControllable(Controllable* pControllable)
{
    mControllables.push_back(pControllable);
}

void App::fillGui()
{
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::SliderFloat("VolumeExtent", &VOLUME_EXTENT, 300.f, 1024.f, "%0.5f");
    ImGui::Checkbox("Voxelize each frame",&mVoxeliseEachFrame);
    ImGui::Combo("Visualisation", &VISUALIZATION, "RayCasting\0VoxelCubes\0PointCloud\0GBuffer\0Phong\0AmbientOcclusion\0VoxelConeTracing\0LightViewMap\0VoxelGlow\0");
    ImGui::Checkbox("Show GBuffer", &mShowGBuffer);
    ImGui::Text("Controls:\n1: Voxel Cone Tracing \n2: Raycasting \n3: Voxel Cubes \n4: Point Cloud \n5: Gbuffer \n6: Phong \n7: Ambient Occlusion \n8: Shadow Map \n9: Voxel Glow");
   // ImGui::Combo("Visualisation",&VISUALIZATION, "RayCasting\0PointCloud\0LightViewMap\0GBuffer\0VoxelConeTracing\0\0");
}
