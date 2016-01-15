//
// Created by miland on 13.01.16.
//

#include "VoxelRepresentation.h"

void VoxelRepresentation::VoxelRepresentation()
{

}

void VoxelRepresentation::draw(float windowWidth, float windowHeight) const
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    // Use the one and only shader
    mupShader->use();

    // Create uniforms used by shader
    glm::mat4 uniformProjection = glm::perspective(glm::radians(35.0f), windowWidth / windowHeight, 0.1f, 300.f);
    glm::mat4 uniformModel = glm::mat4(1.f);

    // Fill uniforms to shader
    mupShader->updateUniform("projection", uniformProjection);
    mupShader->updateUniform("view", mCamera.getViewMatrix());
    mupShader->updateUniform("model", uniformModel); // all meshes have center at 0,0,0

    float *testVertices  = new float[9];
    testVertices = {
            +0.0f, +0.0f, -1.0f,
            +0.5f, +0.0f, -1.0f,
            -0.5f, +0.0f, -1.0f
    };

    glGenBuffers(1, &mTestPointBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, mTestPointBuffer);
    glBufferData(GL_ARRAY_BUFFER, 3 * 2 * sizeof(GLfloat), testVertices, GL_STATIC_DRAW);

    delete testVertices;
}