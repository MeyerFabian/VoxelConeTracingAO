#include "Voxelization.h"

#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/gl3w/include/GL/gl3w.h"

#include <iostream>

Voxelization::Voxelization(
        Scene const * pScene,
        float volumeLeft,
        float volumeRight,
        float volumeBottom,
        float volumeTop,
        float volumeNear,
        float volumeFar)
{
    // Saved in members, at the moment not necessary
    mpScene = pScene;

    // Setup OpenGL for voxelization
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glViewport(0, 0, 384, 384);

    // ### Shader program ###
    ShaderProgram shader("/vertex_shaders/voxelization.vert","/fragment_shaders/voxelization.frag", "/geometry_shaders/voxelization.geom");
    shader.use();

    // ### Transformation ###

    // Orthographic projection
    glm::mat4 projection = glm::ortho(
        volumeLeft,
        volumeRight,
        volumeBottom,
        volumeTop,
        volumeNear,
        volumeFar);
    shader.updateUniform("model", glm::mat4(1.0));
    shader.updateUniform("modelNormal", glm::mat4(1.0)); // same since identity
    shader.updateUniform("projectionView", projection);

    // ### Atomic counter #####

    // Generate atomic buffer
    GLuint atomicBuffer;
    glGenBuffers(1, &atomicBuffer);

    // Bind buffer and define capacity
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicBuffer);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);

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

    // Bind it to shader
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, atomicBuffer);

    // Unbind the buffer for the moment
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

    // ### Buffer texture ###

    // TODO (world pos, normal, color)

    // ### Execution ###

    // Draw scene with voxelization shader
    mpScene->drawWithCustomShader();

    glMemoryBarrier(GL_ATOMIC_COUNTER_BARRIER_BIT);

    // ### Finish ###

    // Read atomic counter
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicBuffer);

    mapping = (GLuint*)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER,
                                             0,
                                             sizeof(GLuint),
                                             GL_MAP_READ_BIT
                                            );
    std::cout << "Voxel fragments count: " << mapping[0] << std::endl;
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);

    // Cleaning up
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
    glDeleteBuffers(1, &atomicBuffer);
}
