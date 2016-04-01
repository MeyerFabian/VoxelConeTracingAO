#include "OctreeRaycaster.h"

OctreeRaycaster::OctreeRaycaster(App* pApp) : Controllable(pApp, "Raycasting")
{
    // Initialize members
    m_stepSize = 0.05f;
    m_directionBeginScale = 0.5f;
    m_maxSteps = 100;
    m_maxLevel = 8;
    m_upOctreeRaycasterShader = std::unique_ptr<ShaderProgram>(new ShaderProgram("/vertex_shaders/octreeRaycast.vert",
                                                                           "/fragment_shaders/octreeRaycast.frag"));
}

void OctreeRaycaster::draw(glm::vec3 camPos,
        NodePool& nodePool,
        BrickPool& brickPool,
        std::unique_ptr<GBuffer>& gbuffer,
        GLuint screenQuad,
        float volumeExtent) const
{
    // Prepare OpenGL
    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);

    // Bind octree image to binding 0
    nodePool.bind();

    // Update uniforms
    m_upOctreeRaycasterShader->updateUniform("stepSize", m_stepSize);
    m_upOctreeRaycasterShader->updateUniform("directionBeginScale", m_directionBeginScale);
    m_upOctreeRaycasterShader->updateUniform("maxSteps", m_maxSteps);
    m_upOctreeRaycasterShader->updateUniform("camPos", camPos);
    m_upOctreeRaycasterShader->updateUniform("volumeExtent", volumeExtent);
    m_upOctreeRaycasterShader->updateUniform("volumeRes", static_cast<float>(brickPool.getResolution().x));
    m_upOctreeRaycasterShader->updateUniform("maxLevel", m_maxLevel);

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
    GLint brickPoolUniform = glGetUniformLocation(static_cast<GLuint>(m_upOctreeRaycasterShader->getShaderProgramHandle()), "brickPool");
    glUniform1i(brickPoolUniform, 0);
    brickPool.bind();

    // Use shader
    m_upOctreeRaycasterShader->use();

    // Draw voxel
    glBindVertexArray(screenQuad);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    // Disable shader
    m_upOctreeRaycasterShader->disable();
}

void OctreeRaycaster::fillGui(){
    ImGui::SliderFloat("step size", &m_stepSize, 0.001f, 1.0f, "%.3f");
    ImGui::SliderInt("max steps", &m_maxSteps, 50, 2000,"%.0f");
    ImGui::SliderFloat("ray begin", &m_directionBeginScale, 0.0f, 30.0f, "%.1f");
    ImGui::SliderInt("max level", &m_maxLevel, 1, 8, "%.0f");
}
