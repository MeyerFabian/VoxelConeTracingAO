//
// Created by miland on 16.01.16.
//

#include <src/Scene/Camera.h>
#include <src/SparseOctree/NodePool.h>
#include "OctreeRaycast.h"

OctreeRaycast::OctreeRaycast()
{
    glGenVertexArrays(1, &vaoID);
    glBindVertexArray(vaoID);

    mupOctreeRaycastShader = std::unique_ptr<ShaderProgram>(new ShaderProgram("/vertex_shaders/octreeRaycast.vert",
                                                                              "/fragment_shaders/octreeRaycast.frag"));
    mupOctreeRaycastShader->use();

    /*/
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

void OctreeRaycast::draw(glm::vec3 camPos, NodePool& nodePool, float stepSize) const
{
    GLint octreeUniform = glGetUniformLocation(static_cast<GLuint>(mupOctreeRaycastShader->getShaderProgramHandle()), "octree");
    glUniform1i(octreeUniform, 1);
    // bind octree texture
    nodePool.bind();

    // update uniforms
    mupOctreeRaycastShader->updateUniform("stepSize", stepSize);
    mupOctreeRaycastShader->updateUniform("camPos", camPos);
    //mupOctreeRaycastShader->addTexture("octree", nodePool.getNodePoolTextureID());

    // use shader AFTER texture is added
    mupOctreeRaycastShader->use();

    // draw voxel
    glBindVertexArray(vaoID);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}
