#include "PointCloud.h"

#include "externals/GLM/glm/gtx/string_cast.hpp"

PointCloud::PointCloud(FragmentList* pFragmentList, Camera const * pCamera)
{
    mpFragmentList = pFragmentList;
    mpCamera = pCamera;
    mupShaderProgram = std::unique_ptr<ShaderProgram>(new ShaderProgram("/vertex_shaders/point.vert", "/fragment_shaders/point.frag"));
    glPointSize(15.f);
}


PointCloud::~PointCloud()
{
    // Nothing to do
}

void PointCloud::draw(float width, float height, float volumeExtent)
{
    // Prepare OpenGL
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    // Bind shader and no VAO
    glBindVertexArray(0);
    mupShaderProgram->use();

    // Set view and projection matrix
    mupShaderProgram->updateUniform("cameraView", mpCamera->getViewMatrix());
    mupShaderProgram->updateUniform("projection", mpCamera->getProjection(width, height));

    // Volume center and extent for scaling
    mupShaderProgram->updateUniform("volumeExtent", volumeExtent);

    // Bind fragment lists
    mpFragmentList->bindReadonly();

    // Draw points
    glDrawArrays(GL_POINTS, 0, mpFragmentList->getVoxelCount());

    // Unbind shader
    mupShaderProgram->disable();
}
