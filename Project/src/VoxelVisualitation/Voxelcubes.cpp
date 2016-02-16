#include "Voxelcubes.h"

#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/GLM/glm/gtx/string_cast.hpp"

VoxelCubes::VoxelCubes(Camera const * pCamera)
{
    mpCamera = pCamera;
    mupShaderProgram = std::unique_ptr<ShaderProgram>(new ShaderProgram("/vertex_shaders/voxelcubes.vert", "/fragment_shaders/voxelcubes.frag", "/geometry_shaders/voxelcubes.geom"));
}

void VoxelCubes::draw(float width, float height, float volumeExtent) const
{
    int level = 8;
    int resolution = (double)pow(2.0, (double)level);

    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    // Use shader
    mupShaderProgram->use();

    /*
    // Bind octree image to binding 0
    nodePool.bind();

    // Brick pool binding as sampler texture
    glActiveTexture(GL_TEXTURE0);
    GLint brickPoolUniform = glGetUniformLocation(static_cast<GLuint>(mupShaderProgram->getShaderProgramHandle()), "brickPool");
    glUniform1i(brickPoolUniform, 0);
    brickPool.bind();
    */

    // Update uniforms
    mupShaderProgram->updateUniform("cameraView", mpCamera->getViewMatrix());
    mupShaderProgram->updateUniform("projection", glm::perspective(glm::radians(35.0f), width / height, 0.1f, 400.f));
    mupShaderProgram->updateUniform("volumeExtent", volumeExtent);
    mupShaderProgram->updateUniform("resolution", resolution);

    // Draw for each voxel one point
    glBindVertexArray(0);
    glDrawArrays(GL_POINTS, 0, (GLuint)pow((double)resolution, 3.0));

    // Disable shader
    mupShaderProgram->disable();

}
