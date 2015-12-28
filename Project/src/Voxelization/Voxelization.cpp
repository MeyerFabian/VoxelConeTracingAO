#include "Voxelization.h"

#include "externals/GLM/glm/gtc/matrix_transform.hpp"

#include <iostream>

Voxelization::Voxelization()
{
    // TODO: Normals and position of voxel fragments
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
    glDeleteBuffers(1, &mAtomicBuffer);
}


void Voxelization::voxelize(Scene const * pScene,
                            float volumeLeft,
                            float volumeRight,
                            float volumeBottom,
                            float volumeTop,
                            float volumeNear,
                            float volumeFar,
                            FragmentList *fragmentList)
{
    // Setup OpenGL for voxelization
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glViewport(0, 0, 384, 384);

    mVoxelizationShader->use();

    // ### Transformation ###

    // Orthographic projection
    glm::mat4 projection = glm::ortho(
            volumeLeft,
            volumeRight,
            volumeBottom,
            volumeTop,
            volumeNear,
            volumeFar);
    mVoxelizationShader->updateUniform("model", glm::mat4(1.0));
    mVoxelizationShader->updateUniform("modelNormal", glm::mat4(1.0)); // same since identity
    mVoxelizationShader->updateUniform("projectionView", projection);

    resetAtomicCounter();

    // Color
    GLuint colorOutputUniformPosition = glGetUniformLocation(mVoxelizationShader->getShaderProgramHandle(), "colorOutputImage");
    glUniform1i(colorOutputUniformPosition, 1); // TODO: getter for texture unit? or hard coded?

    fragmentList->bind();

    // Draw scene with voxelization shader
    pScene->drawWithCustomShader(); // uses texture slot 0 for diffuse texture mapping

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

