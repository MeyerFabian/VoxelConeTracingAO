#include "PointCloud.h"

#include "externals/GLM/glm/gtx/string_cast.hpp"

PointCloud::PointCloud(Camera const * pCamera)
{
    m_pCamera = pCamera;
    m_upShaderProgram = std::unique_ptr<ShaderProgram>(new ShaderProgram("/vertex_shaders/point.vert", "/fragment_shaders/point.frag"));
    glPointSize(15.f);
}


PointCloud::~PointCloud()
{
    // Nothing to do
}

void PointCloud::draw(float width, float height, float volumeExtent, FragmentList const * pFragmentList)
{
    // Prepare OpenGL
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    // Bind shader and no VAO
    glBindVertexArray(0);
    m_upShaderProgram->use();

    // Set view and projection matrix
    m_upShaderProgram->updateUniform("cameraView", m_pCamera->getViewMatrix());
    m_upShaderProgram->updateUniform("projection", m_pCamera->getProjection(width, height));

    // Volume center and extent for scaling
    m_upShaderProgram->updateUniform("volumeExtent", volumeExtent);

    // Bind fragment lists
    pFragmentList->bindReadonly();

    // Draw points
    glDrawArrays(GL_POINTS, 0, pFragmentList->getVoxelCount());

    // Unbind shader
    m_upShaderProgram->disable();
}
