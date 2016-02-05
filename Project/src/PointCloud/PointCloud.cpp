#include "PointCloud.h"

#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/GLM/glm/gtx/string_cast.hpp"

PointCloud::PointCloud(FragmentList* pFragmentList, Camera const * pCamera, GLint pointCount)
{

    mpFragmentList = pFragmentList;
    mpCamera = pCamera;
    mPointCount = pointCount;

    mupShaderProgram = std::unique_ptr<ShaderProgram>(new ShaderProgram("/vertex_shaders/point.vert", "/fragment_shaders/point.frag"));

    glGenVertexArrays(1, &mVAO);

    glPointSize(1.f);
}


PointCloud::~PointCloud()
{
    glDeleteVertexArrays(1, &mVAO);
}

void PointCloud::draw(float width, float height, float volumeExtent)
{
    // Bind VAO and shader
    glBindVertexArray(mVAO);
    mupShaderProgram->use();

    // Set view and projection matrix
    mupShaderProgram->updateUniform("cameraView", mpCamera->getViewMatrix());
    mupShaderProgram->updateUniform("projection", glm::perspective(glm::radians(35.0f), width / height, 0.1f, 400.f)); // TODO: should be in camera class

    // Volume center and extent for scaling
    mupShaderProgram->updateUniform("volumeExtent", volumeExtent);

    // Fragment voxel count
    mupShaderProgram->updateUniform("voxelCount", (float)mpFragmentList->getVoxelCount());

    // Uniforms for fragment lists
    mupShaderProgram->updateUniform("positionImage", 1);
    mupShaderProgram->updateUniform("normalImage", 2);
    mupShaderProgram->updateUniform("colorImage", 3);

    // Bind fragment lists
    mpFragmentList->bind();

    // Draw points
    glDrawArrays(GL_POINTS, 0, mPointCount);

    // Unbind VAO and shader
    glBindVertexArray(0);
    mupShaderProgram->disable();
}
