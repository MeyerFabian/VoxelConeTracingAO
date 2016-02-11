//
// Created by miland on 16.01.16.
//

#include "OctreeRaycast.h"

OctreeRaycast::OctreeRaycast(App* pApp) : Controllable(pApp, "Raycasting")
{
    stepSize = 0.05f;
    directionBeginScale=0.5f;
    maxSteps=100;
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

void OctreeRaycast::draw(glm::vec3 camPos,
        NodePool& nodePool,
        BrickPool& brickPool,
        std::unique_ptr<GBuffer>& gbuffer,
        float volumeExtent,
        int maxLevel) const
{
    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);

    GLint octreeUniform = glGetUniformLocation(static_cast<GLuint>(mupOctreeRaycastShader->getShaderProgramHandle()), "octree");
    glUniform1i(octreeUniform, 0);
    // bind octree texture
    nodePool.bind();

    // update uniforms
    mupOctreeRaycastShader->updateUniform("stepSize", stepSize);
    mupOctreeRaycastShader->updateUniform("directionBeginScale", directionBeginScale);
    mupOctreeRaycastShader->updateUniform("maxSteps", maxSteps);
    mupOctreeRaycastShader->updateUniform("camPos", camPos);
    mupOctreeRaycastShader->updateUniform("volumeExtent", volumeExtent);
    mupOctreeRaycastShader->updateUniform("volumeRes", static_cast<float>(brickPool.getResolution().x-1));
    mupOctreeRaycastShader->updateUniform("maxLevel", maxLevel);

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

    GLint brickPoolUniform = glGetUniformLocation(static_cast<GLuint>(mupOctreeRaycastShader->getShaderProgramHandle()), "brickPool");
    glUniform1i(brickPoolUniform, 2);
    glActiveTexture(GL_TEXTURE2);
    brickPool.bind();

    // use shader AFTER texture is added
    mupOctreeRaycastShader->use();

    // draw voxel
    glBindVertexArray(vaoID);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    mupOctreeRaycastShader->disable();
}

void OctreeRaycast::fillGui(){
    ImGui::SliderFloat("step size", &stepSize, 0.001f, 1.0f, "%.3f");
    ImGui::SliderInt("max steps", &maxSteps, 50, 2000,"%.0f");
    ImGui::SliderFloat("ray begin", &directionBeginScale, 0.0f, 5.0f, "%.1f");
}
