#include "Mesh.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "Utilities/errorUtils.h"

Mesh::Mesh(std::string filepath)
{
    // Create instance of assimp
    Assimp::Importer importer;

    // Import
    const aiScene* scene = importer.ReadFile(filepath,
        aiProcess_CalcTangentSpace       |
        aiProcess_Triangulate            |
        aiProcess_JoinIdenticalVertices  |
        aiProcess_SortByPType);

    // Check whether import was successful
    if(!scene)
    {
        ErrorHandler::log(importer.GetErrorString());
    }

    // Perpare OpenGL buffers


    // Iterate over meshes in scene
    for(int i = 0; i < scene->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[i];

        // Collect positions
        float *vertices = new float[mesh->mNumVertices * 3];
        for(int i = 0; i < mesh->mNumVertices; i++) {
            vertices[i * 3] = mesh->mVertices[i].x;
            vertices[i * 3 + 1] = mesh->mVertices[i].y;
            vertices[i * 3 + 2] = mesh->mVertices[i].z;
        }

        /*glGenBuffers(1, &mVBOs[VERTEX_BUFFER]);
        glBindBuffer(GL_ARRAY_BUFFER, mVBOs[VERTEX_BUFFER]);
        glBufferData(GL_ARRAY_BUFFER, 3 * mesh->mNumVertices * sizeof(GLfloat), vertices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray (0);*/

        delete[] vertices;

        // Collect normals

        // Collect indices

        // Get material

    }

    // Scene deleted by importer destructor
}

Mesh::~Mesh()
{

}
