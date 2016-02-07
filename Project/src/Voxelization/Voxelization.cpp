#include "Voxelization.h"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"

#include <iostream>

Voxelization::Voxelization(App *pApp ) :Controllable(pApp, "Voxelisation")
{
    // ### Shader program ###
    mVoxelizationShader = std::unique_ptr<ShaderProgram>(
            new ShaderProgram("/vertex_shaders/voxelization.vert","/fragment_shaders/voxelization.frag", "/geometry_shaders/voxelization.geom"));

    // ### Atomic counter #####
    // Generate atomic buffer
    glGenBuffers(1, &mAtomicBuffer);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, mAtomicBuffer);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);

    resetAtomicCounter();
}

Voxelization::~Voxelization()
{
    // TODO: Delete all the OpenGL stuff
    glDeleteBuffers(1, &mAtomicBuffer);
}


void Voxelization::voxelize(float extent, Scene const * pScene, FragmentList *fragmentList)
{
    // Setup OpenGL for voxelization
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glViewport(0, 0, determineVoxeliseResolution(VOXELIZE_RESOLUTION), determineVoxeliseResolution(VOXELIZE_RESOLUTION));

    mVoxelizationShader->use();

    // ### Transformation ###

    // Orthographic projections
    float halfExtent = extent / 2.f;
    glm::mat4 orthographicProjection = glm::ortho(-halfExtent, halfExtent, -halfExtent, halfExtent, -halfExtent, halfExtent);
    mVoxelizationShader->updateUniform("orthographicProjection", orthographicProjection);

    resetAtomicCounter();

    // Bind correct texture slots (TODO: should be done else where and cleaner)
    GLint positionOutputUniformPosition = glGetUniformLocation(static_cast<GLuint>(mVoxelizationShader->getShaderProgramHandle()), "positionOutputImage");
    glUniform1i(positionOutputUniformPosition, 1);
    GLint normalOutputUniformPosition = glGetUniformLocation(static_cast<GLuint>(mVoxelizationShader->getShaderProgramHandle()), "normalOutputImage");
    glUniform1i(normalOutputUniformPosition, 2);
    GLint colorOutputUniformPosition = glGetUniformLocation(static_cast<GLuint>(mVoxelizationShader->getShaderProgramHandle()), "colorOutputImage");
    glUniform1i(colorOutputUniformPosition, 3);

    // Give shader the pixel size for conservative rasterization
    glUniform1f(glGetUniformLocation(static_cast<GLuint>(mVoxelizationShader->getShaderProgramHandle()), "pixelSize"), 2.f / determineVoxeliseResolution(VOXELIZE_RESOLUTION));

    // Bind fragment list with output textures / buffers
    fragmentList->bindWriteonly();

    // Draw scene with voxelization shader
    for (auto& bucket : pScene->getRenderBuckets())
    {
        // Bind texture of mesh material (pointer to shader is needed for location)
        bucket.first->bind(mVoxelizationShader.get());

        // Draw all meshes in that bucket
        for (Mesh const * pMesh : bucket.second)
        {
            pMesh->draw();
        }
    }

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);

    fragmentList->setVoxelCount(readAtomicCounter());

    mVoxelizationShader->disable();
}

GLuint Voxelization::readAtomicCounter() const {// Read atomic counter
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, mAtomicBuffer);

    GLuint *mapping = (GLuint*)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER,
                                                0,
                                                sizeof(GLuint),
                                                GL_MAP_READ_BIT
    );

    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

    return mapping[0];
}

void Voxelization::resetAtomicCounter() const {
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, mAtomicBuffer);

    // Map the buffer
    GLuint* mapping = (GLuint*)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER,
                                                0 ,
                                                sizeof(GLuint),
                                                GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT
    );
    // Set memory to new value
    memset(mapping, 0, sizeof(GLuint));

    // Unmap the buffer
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, mAtomicBuffer);

    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
}

unsigned int Voxelization::determineVoxeliseResolution(int res) {
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

void Voxelization::fillGui()
{
    ImGui::Combo("Resolution", &VOXELIZE_RESOLUTION ," 256x256x256\0 384*384*384\0 512*512*512\0 1024*1024*1024\0");
}
