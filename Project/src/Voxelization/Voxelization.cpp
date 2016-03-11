#include "Voxelization.h"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"

#include <iostream>

Voxelization::Voxelization(App *pApp ) :Controllable(pApp, "Voxelization")
{
    // ### Shader program ###
    m_voxelizationShader = std::unique_ptr<ShaderProgram>(
            new ShaderProgram("/vertex_shaders/voxelization.vert","/fragment_shaders/voxelization.frag", "/geometry_shaders/voxelization.geom"));

    // ### Atomic counter #####
    // Generate atomic buffer
    glGenBuffers(1, &m_atomicBuffer);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_atomicBuffer);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);

    resetAtomicCounter();
}

Voxelization::~Voxelization()
{
    // TODO: Delete all the OpenGL stuff
    glDeleteBuffers(1, &m_atomicBuffer);
}


void Voxelization::voxelize(float extent, Scene const * pScene, FragmentList* pFragmentList)
{
    // Resolution
    int resolution = determineVoxeliseResolution(m_voxelizationResolution);

    // Setup OpenGL for voxelization
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glViewport(0, 0, resolution, resolution);

    // Use voxelization shader
    m_voxelizationShader->use();

    // Orthographic projections
    float halfExtent = extent / 2.f;
    glm::mat4 orthographicProjection = glm::ortho(-halfExtent, halfExtent, -halfExtent, halfExtent, -halfExtent, halfExtent);
    m_voxelizationShader->updateUniform("orthographicProjection", orthographicProjection);

    // Reset the atomic counter
    resetAtomicCounter();

    // Give shader the pixel size for conservative rasterization
    glUniform1f(glGetUniformLocation(static_cast<GLuint>(m_voxelizationShader->getShaderProgramHandle()), "pixelSize"), 2.f / resolution);

    // Bind fragment list with output textures / buffers
    pFragmentList->reset();
    pFragmentList->bindWriteonly();

    // Draw it with custom shader
    pScene->draw(m_voxelizationShader.get(), "model");

    // Wait until finished
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);

    // Remember count
    pFragmentList->setVoxelCount(readAtomicCounter());

    // Disable shade
    m_voxelizationShader->disable();
}

void Voxelization::fillGui()
{
    ImGui::Combo("Resolution", &m_voxelizationResolution ," 256x256x256\0 384*384*384\0 512*512*512\0 1024*1024*1024\0");
}

int Voxelization::getResolution() const
{
    return determineVoxeliseResolution(m_voxelizationResolution);
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

unsigned int Voxelization::determineVoxeliseResolution(int res) const
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
