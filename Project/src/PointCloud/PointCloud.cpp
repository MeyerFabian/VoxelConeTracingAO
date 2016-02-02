#include "PointCloud.h"

PointCloud::PointCloud(FragmentList* pFragmentList, Camera const * pCamera)
{
    // VertexAttribArray necessary?
    mpFragmentList = pFragmentList;
    mpCamera = pCamera;

    mupShaderProgram = std::unique_ptr<ShaderProgram>(new ShaderProgram("/vertex_shaders/point.vert", "/fragment_shaders/point.frag"));

    glGenBuffers(1, &mVBO);
    glBindBuffer(GL_ARRAY_BUFFER, mVBO);
    glBufferData(GL_ARRAY_BUFFER, POINT_COUNT * sizeof(GL_FLOAT), 0, GL_STATIC_DRAW);

    glGenVertexArrays(1, &mVAO);
    glBindVertexArray(mVAO);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, NULL);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


PointCloud::~PointCloud()
{
    glDeleteVertexArrays(1, &mVAO);
    glDeleteBuffers(1, &mVBO);

}

void PointCloud::draw()
{
    glBindVertexArray(mVAO);

    // Fragment voxel count
    glUniform1f(glGetUniformLocation(static_cast<GLuint>(mupShaderProgram->getShaderProgramHandle()), "voxelCount"), mpFragmentList->getVoxelCount());

    // Uniforms for fragment lists
    GLint positionUniformPosition = glGetUniformLocation(static_cast<GLuint>(mupShaderProgram->getShaderProgramHandle()), "positionImage");
    glUniform1i(positionUniformPosition, 1);
   /*GLint normalUniformPosition = glGetUniformLocation(static_cast<GLuint>(mupShaderProgram->getShaderProgramHandle()), "normalImage");
    glUniform1i(normalUniformPosition, 2);
    GLint colorUniformPosition = glGetUniformLocation(static_cast<GLuint>(mupShaderProgram->getShaderProgramHandle()), "colorImage");
    glUniform1i(colorUniformPosition, 3);*/

    // Bind fragment lists
    mpFragmentList->bindPosition();

    glDrawArrays(GL_POINTS, 0, POINT_COUNT);
    glBindVertexArray(0);
}
