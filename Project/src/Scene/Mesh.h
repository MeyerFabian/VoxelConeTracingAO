#ifndef MESH_H_
#define MESH_H_

#include "externals/gl3w/include/GL/gl3w.h"

#include <assimp/scene.h>
#include <string>
#include <vector>

const float MESH_SCALE = 0.1f;

class Mesh
{
public:

    Mesh(aiMesh const * mesh);
    virtual ~Mesh();

    void draw() const;

private:

    // Members
    GLuint m_vertexBuffer;
    GLuint m_normalBuffer;
    GLuint m_indexBuffer;
    GLuint m_tangentBuffer;
    GLuint m_UVBuffer;
    GLuint m_vertexArrayObject;
    int m_elementCount;
};

#endif // MESH_H_
