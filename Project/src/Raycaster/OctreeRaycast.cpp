//
// Created by miland on 16.01.16.
//

#include "OctreeRaycast.h"

OctreeRaycast::OctreeRaycast(App* pApp) : Controllable(pApp, "Raycasting")
{
    stepSize = 0.05f;
    directionBeginScale = 0.5f;
    maxSteps = 100;
    maxLevel = 8;

    mupOctreeRaycastShader = std::unique_ptr<ShaderProgram>(new ShaderProgram("/vertex_shaders/octreeRaycast.vert",
                                                                              "/fragment_shaders/octreeRaycast.frag"));
}

void OctreeRaycast::draw(glm::vec3 camPos,
        NodePool& nodePool,
        BrickPool& brickPool,
        std::unique_ptr<GBuffer>& gbuffer,
        GLuint ScreenQuad,
        float volumeExtent) const
{
    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);

    // Bind octree image
    nodePool.bind();

    // Update uniforms
    mupOctreeRaycastShader->updateUniform("stepSize", stepSize);
    mupOctreeRaycastShader->updateUniform("directionBeginScale", directionBeginScale);
    mupOctreeRaycastShader->updateUniform("maxSteps", maxSteps);
    mupOctreeRaycastShader->updateUniform("camPos", camPos);
    mupOctreeRaycastShader->updateUniform("volumeExtent", volumeExtent);
    mupOctreeRaycastShader->updateUniform("volumeRes", static_cast<float>(brickPool.getResolution().x));
    mupOctreeRaycastShader->updateUniform("maxLevel", maxLevel);

    // Position texture as image
    glBindImageTexture(1,
                       gbuffer->getTextureID(GBuffer::GBUFFER_TEXTURE_TYPE_POSITION),
                       0,
                       GL_TRUE,
                       0,
                       GL_READ_ONLY,
                       GL_RGBA32F);


    // Brick pool binding as sampler texture
    glActiveTexture(GL_TEXTURE0);
    GLint brickPoolUniform = glGetUniformLocation(static_cast<GLuint>(mupOctreeRaycastShader->getShaderProgramHandle()), "brickPool");
    glUniform1i(brickPoolUniform, 0);
    brickPool.bind();

    // Use shader
    mupOctreeRaycastShader->use();

    // Draw voxel
    glBindVertexArray(ScreenQuad);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    mupOctreeRaycastShader->disable();
}

void OctreeRaycast::fillGui(){
    ImGui::SliderFloat("step size", &stepSize, 0.001f, 1.0f, "%.3f");
    ImGui::SliderInt("max steps", &maxSteps, 50, 2000,"%.0f");
    ImGui::SliderFloat("ray begin", &directionBeginScale, 0.0f, 30.0f, "%.1f");
    ImGui::SliderInt("max level", &maxLevel, 1, 8, "%.0f");

}
