#include "VoxelCubes.h"

#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/GLM/glm/gtx/string_cast.hpp"

VoxelCubes::VoxelCubes(Camera const * pCamera)
{
    m_pCamera = pCamera;
    m_upShaderProgram = std::unique_ptr<ShaderProgram>(new ShaderProgram("/vertex_shaders/voxelcubes.vert", "/fragment_shaders/voxelcubes.frag", "/geometry_shaders/voxelcubes.geom"));
}

void VoxelCubes::draw(
        float width,
        float height,
        float volumeExtent,
        NodePool& nodePool,
        BrickPool& brickPool) const
{
    // Initialize some values
    int maxLevel = 8;
    int resolution = (double)pow(2.0, (double)maxLevel);

    // Prepare OpenGL
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    // Use shader
    m_upShaderProgram->use();

    // Bind octree image to binding 0
    nodePool.bind();

    // Brick pool binding as sampler texture
    glActiveTexture(GL_TEXTURE0);
    GLint brickPoolUniform = glGetUniformLocation(static_cast<GLuint>(m_upShaderProgram->getShaderProgramHandle()), "brickPool");
    glUniform1i(brickPoolUniform, 0);
    brickPool.bind();

    // Update uniforms
    m_upShaderProgram->updateUniform("cameraView", m_pCamera->getViewMatrix());
    m_upShaderProgram->updateUniform("projection", m_pCamera->getProjection(width, height));
    m_upShaderProgram->updateUniform("volumeExtent", volumeExtent);
    m_upShaderProgram->updateUniform("resolution", resolution);
    m_upShaderProgram->updateUniform("maxLevel", maxLevel);

    // Draw for each voxel one point
    glBindVertexArray(0);
    glDrawArrays(GL_POINTS, 0, (GLuint)pow((double)resolution, 3.0));

    // Disable shader
    m_upShaderProgram->disable();

}
