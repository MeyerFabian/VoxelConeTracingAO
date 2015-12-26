#ifndef MESH_H_
#define MESH_H_

#include <GL/gl3w.h>
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
    GLuint mVertexBuffer;
    GLuint mNormalBuffer;
    GLuint mIndexBuffer;
    GLuint mUVBuffer;
    GLuint mVertexArrayObject;
    int mElementCount;
};

#endif // MESH_H_
