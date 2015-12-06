#ifndef MESH_H_
#define MESH_H_

#include <assimp/scene.h>
#include <string>
#include <vector>

class Mesh
{
public:

    Mesh(aiMesh const * pAssimpMesh);
    virtual ~Mesh();

private:

    // Members
    int mVertexBuffer;
    int mNormalBuffer;
    int mIndexBuffer;
    int mUVBuffer;
};

#endif // MESH_H_
