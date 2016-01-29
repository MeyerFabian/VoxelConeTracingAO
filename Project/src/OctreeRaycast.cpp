//
// Created by miland on 16.01.16.
//

#include "OctreeRaycast.h"

OctreeRaycast::OctreeRaycast()
{
    glGenVertexArrays(1, &vaoID);
    glBindVertexArray(vaoID);

    mupOctreeRaycastShader = std::unique_ptr<ShaderProgram>(new ShaderProgram("/vertex_shaders/octreeRaycast.vert",
                                                                              "/fragment_shaders/octreeRaycast.frag"));
    mupOctreeRaycastShader->use();

    //*/
    const GLfloat plane_vert_data[] = {
            -1.0f, -1.0f,
            +1.0f, -1.0f,
            -1.0f, +1.0f,
            +1.0f, +1.0f,
    };
    /*/
    const GLfloat plane_vert_data[] = {
            -0.5f, -0.5f,
            +0.5f, -0.5f,
            -0.5f, +0.5f,
            +0.5f, +0.5f,
    };
    //*/
    GLuint mBufferID;
    glGenBuffers(1, &mBufferID);
    glBindBuffer(GL_ARRAY_BUFFER, mBufferID);
    glBufferData(GL_ARRAY_BUFFER, sizeof(plane_vert_data), plane_vert_data, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);

    glBindVertexArray(0);
}

void OctreeRaycast::draw(glm::vec3 camPos, NodePool& nodePool, std::unique_ptr<GBuffer>& gbuffer, float stepSize) const
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    GLint octreeUniform = glGetUniformLocation(static_cast<GLuint>(mupOctreeRaycastShader->getShaderProgramHandle()), "octree");
    glUniform1i(octreeUniform, 0);
    // bind octree texture
    nodePool.bind();

    // update uniforms
    mupOctreeRaycastShader->updateUniform("stepSize", stepSize);
    mupOctreeRaycastShader->updateUniform("camPos", camPos);

    //mupOctreeRaycastShader->addTexture("positionTex", gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_POSITION));

    // Position texture as image
    glActiveTexture(GL_TEXTURE1);
    glBindImageTexture(1,
                       gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_POSITION),
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_ONLY,
                       GL_RGBA32F);
    GLint worldPosUniform = glGetUniformLocation(static_cast<GLuint>(mupOctreeRaycastShader->getShaderProgramHandle()), "worldPos");
    glUniform1i(worldPosUniform, 1);

    // use shader AFTER texture is added
    mupOctreeRaycastShader->use();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // draw voxel
    glBindVertexArray(vaoID);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    mupOctreeRaycastShader->disable();
}
