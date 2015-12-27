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
    /*GLuint atomicsBuffer;
    glGenBuffers(1, &atomicsBuffer);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicsBuffer);
    GLuint a = 0;
    glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0 , sizeof(GLuint), &a);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, atomicsBuffer); */

    // ### Buffer texture ###

    // TODO (world pos, normal, color)

    // ### Execution ###

    // Draw scene with voxelization shader
    mpScene->drawWithCustomShader();

    // ### Finish ###

    // Read atomic counter
    /*GLuint* userCounters;
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicsBuffer);
    glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), userCounters);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
    std::cout << "Fragment count = " << *userCounters << std::endl;*/

    // Cleaning up
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
}
