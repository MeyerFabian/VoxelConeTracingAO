#include "Mesh.h"
#include <GL/gl3w.h>

Mesh::Mesh(aiMesh const * pAssimpMesh)
{
    // Vertices
    float *vertices = new float[mesh->mNumVertices * 3];
    for(int i = 0; i < mesh->mNumVertices; i++)
    {
        vertices[i * 3] = mesh->mVertices[i].x;
        vertices[i * 3 + 1] = mesh->mVertices[i].y;
        vertices[i * 3 + 2] = mesh->mVertices[i].z;
    }

    glGenBuffers(1, &mVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, mesh->mNumVertices * 3 * sizeof(GLfloat), vertices, GL_STATIC_DRAW);

    delete[] vertices;

    // Normals
    float *normals = new float[mesh->mNumVertices * 3];
    for(int i = 0; i < mesh->mNumVertices; ++i)
    {
        normals[i * 3] = mesh->mNormals[i].x;
        normals[i * 3 + 1] = mesh->mNormals[i].y;
        normals[i * 3 + 2] = mesh->mNormals[i].z;
    }

    glGenBuffers(1, &mNormalBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, mNormalBuffer);
    glBufferData(GL_ARRAY_BUFFER, mesh->mNumVertices * 3 * sizeof(GLfloat), normals, GL_STATIC_DRAW);

    delete[] normals;

    // Indices
    unsigned int *indices = new unsigned int[mesh->mNumFaces * 3];
    for(int i = 0; i < mesh->mNumFaces; ++i)
    {
        indices[i * 3] = mesh->mFaces[i].mIndices[0];
        indices[i * 3 + 1] = mesh->mFaces[i].mIndices[1];
        indices[i * 3 + 2] = mesh->mFaces[i].mIndices[2];
    }

    glGenBuffers(1, &mIndexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh->mNumFaces * 3 * sizeof(GLuint), indices, GL_STATIC_DRAW);

    delete[] indices;

    // Texture coordinates
    float *texCoords = new float[mesh->mNumVertices * 2];
    for(int i = 0; i < mesh->mNumVertices; ++i)
    {
        texCoords[i * 2] = mesh->mTextureCoords[0][i].x;
        texCoords[i * 2 + 1] = mesh->mTextureCoords[0][i].y;
    }

    glGenBuffers(1, &mUVBuffers[i]);
    glBindBuffer(GL_ARRAY_BUFFER, mUVBuffers[i]);
    glBufferData(GL_ARRAY_BUFFER, mesh->mNumVertices * 2 * sizeof(GLfloat), texCoords, GL_STATIC_DRAW);

    delete[] texCoords;
}

Mesh::~Mesh()
{
    glDeleteBuffers(1, &mVertexBuffer);
    glDeleteBuffers(1, &mNormalBuffer);
    glDeleteBuffers(1, &mIndexBuffer);
    glDeleteBuffers(1, &mUVBuffer);
}
