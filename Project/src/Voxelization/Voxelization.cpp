#include "Voxelization.h"

#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/gl3w/include/GL/gl3w.h"

#include <iostream>

// Defines
const GLuint maxVoxelFragmentCount = 2750071; // Needed for buffer creation...

Voxelization::Voxelization(
        Scene const * pScene,
        float volumeLeft,
        float volumeRight,
        float volumeBottom,
        float volumeTop,
        float volumeNear,
        float volumeFar)
{
    // TODO: Normals and position of voxel fragments

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

    // Color
    GLuint colorOutputUniformPosition = glGetUniformLocation(shader.getShaderProgramHandle(), "colorOutputImage");

    // Color buffer
    glGenBuffers(1, &mColorOutputBuffer);
    glBindBuffer(GL_TEXTURE_BUFFER, mColorOutputBuffer);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(GLubyte) * 4 * maxVoxelFragmentCount, 0, GL_DYNAMIC_DRAW);

    // Color texture
    glGenTextures(1, &mColorOutputTexture);

    glBindBuffer(GL_TEXTURE_BUFFER, 0);

    // ### Execution ###

    // Setup write texture stuff
    glActiveTexture(GL_TEXTURE1); // 0 probably used for diffuse texture for texture mapping
    glBindTexture(GL_TEXTURE_1D, mColorOutputTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA8, mColorOutputBuffer);
    glBindImageTexture(1,
                       mColorOutputTexture,
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_WRITE,
                       GL_RGBA8);
    glUniform1i(colorOutputUniformPosition, 1);

    // Draw scene with voxelization shader
    mpScene->drawWithCustomShader(); // uses texture slot 0 for diffuse texture mapping

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);

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

Voxelization::~Voxelization()
{
    glDeleteTextures(1, &mColorOutputTexture);
    glDeleteBuffers(1, &mColorOutputBuffer);
}

GLuint Voxelization::getColorOutputTexture() const
{
    return mColorOutputTexture;
}
