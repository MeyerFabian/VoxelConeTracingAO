#include "Voxelization.h"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"

#include <iostream>

// Easier creation of unique pointers
#ifdef __unix__
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
#endif

Voxelization::Voxelization(App *pApp ) :Controllable(pApp, "Voxelization")
{
    m_resolution = determineVoxelizeResolution(m_voxelizationResolution);

    // ### Shader program ###
    m_upVoxelizationShader = std::unique_ptr<ShaderProgram>(
            new ShaderProgram("/vertex_shaders/voxelization.vert","/fragment_shaders/voxelization.frag", "/geometry_shaders/voxelization.geom"));

    // ### Fragment list ###
    m_upFragmentList = make_unique<FragmentList>(m_resolution);

    // ### Atomic counter ###
    glGenBuffers(1, &m_atomicBuffer);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_atomicBuffer);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);
    resetAtomicCounter();
}

Voxelization::~Voxelization()
{
    glDeleteBuffers(1, &m_atomicBuffer);
}


void Voxelization::voxelize(float extent, Scene const * pScene)
{
    // Resolution
    int resolution = determineVoxelizeResolution(m_voxelizationResolution);

    if(resolution != m_resolution)
    {
        m_resolution = resolution;
        m_upFragmentList = make_unique<FragmentList>(resolution);
    }

    // Setup OpenGL for voxelization
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glViewport(0, 0, m_resolution, m_resolution);

    // Use voxelization shader
    m_upVoxelizationShader->use();

    // Orthographic projections
    float halfExtent = extent / 2.f;
    glm::mat4 orthographicProjection = glm::ortho(-halfExtent, halfExtent, -halfExtent, halfExtent, -halfExtent, halfExtent);
    m_upVoxelizationShader->updateUniform("orthographicProjection", orthographicProjection);

    // Reset the atomic counter
    resetAtomicCounter();

    // Give shader the pixel size for conservative rasterization (not used at the moment?)
    glUniform1f(glGetUniformLocation(static_cast<GLuint>(m_upVoxelizationShader->getShaderProgramHandle()), "pixelSize"), 2.f / m_resolution);

    // Bind fragment list with output textures / buffers
    m_upFragmentList->reset();
    m_upFragmentList->bindWriteonly();

    // Draw it with custom shader
    pScene->draw(m_upVoxelizationShader.get(), "model");

    // Wait until finished
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);

    // Remember count
    m_upFragmentList->setVoxelCount(readAtomicCounter());

    // Disable shader
    m_upVoxelizationShader->disable();
}

void Voxelization::fillGui()
{
    ImGui::Combo("Resolution", &m_voxelizationResolution ," 256x256x256\0 384*384*384\0 512*512*512\0 1024*1024*1024\0");
}

FragmentList const * Voxelization::getFragmentList() const
{
    return m_upFragmentList.get();
}

void Voxelization::mapFragmentListToCUDA()
{
    m_upFragmentList->mapToCUDA();
}

void Voxelization::unmapFragmentListFromCUDA()
{
    m_upFragmentList->unmapFromCUDA();
}

GLuint Voxelization::readAtomicCounter() const
{
    // Read atomic counter
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_atomicBuffer);

    GLuint *mapping = (GLuint*)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER,
                                                0,
                                                sizeof(GLuint),
                                                GL_MAP_READ_BIT);

    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

    return mapping[0];
}

void Voxelization::resetAtomicCounter() const
{
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_atomicBuffer);

    // Map the buffer
    GLuint* mapping = (GLuint*)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER,
                                                0 ,
                                                sizeof(GLuint),
                                                GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    // Set memory to new value
    memset(mapping, 0, sizeof(GLuint));

    // Unmap the buffer
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, m_atomicBuffer);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
}

unsigned int Voxelization::determineVoxelizeResolution(int res) const
{
    switch (res)
    {
        case VoxelizeResolutions::RES_256 :
            return 256;
        case VoxelizeResolutions::RES_384 :
            return 384;
        case VoxelizeResolutions::RES_512 :
            return 512;
        case VoxelizeResolutions::RES_1024 :
            return 1024;
        default:
            return 256;
    }
}
