#include "Mesh.h"

Mesh::Mesh(aiMesh const * mesh)
{
    // Element count
    mElementCount = mesh->mNumFaces * 3;

    // Vertex Array Buffer
    mVertexArrayObject = 0;
    glGenVertexArrays(1, &mVertexArrayObject);
    glBindVertexArray(mVertexArrayObject);

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

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    delete[] vertices;

    // Normals
    float *normals = new float[mesh->mNumVertices * 3];
    for(int i = 0; i < mesh->mNumVertices; i++)
    {
        normals[i * 3] = mesh->mNormals[i].x;
        normals[i * 3 + 1] = mesh->mNormals[i].y;
        normals[i * 3 + 2] = mesh->mNormals[i].z;
    }

    glGenBuffers(1, &mNormalBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, mNormalBuffer);
    glBufferData(GL_ARRAY_BUFFER, mesh->mNumVertices * 3 * sizeof(GLfloat), normals, GL_STATIC_DRAW);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    delete[] normals;

    // Indices
    unsigned int *indices = new unsigned int[mesh->mNumFaces * 3];
    for(int i = 0; i < mesh->mNumFaces; i++)
    {
        indices[i * 3] = mesh->mFaces[i].mIndices[0];
        indices[i * 3 + 1] = mesh->mFaces[i].mIndices[1];
        indices[i * 3 + 2] = mesh->mFaces[i].mIndices[2];
    }

    glGenBuffers(1, &mIndexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh->mNumFaces * 3 * sizeof(GLuint), indices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    delete[] indices;

    // Texture coordinates
    float *texCoords = new float[mesh->mNumVertices * 2];
    for(int i = 0; i < mesh->mNumVertices; i++)
    {
        texCoords[i * 2] = mesh->mTextureCoords[0][i].x;
        texCoords[i * 2 + 1] = mesh->mTextureCoords[0][i].y;
    }

    glGenBuffers(1, &mUVBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, mUVBuffer);
    glBufferData(GL_ARRAY_BUFFER, mesh->mNumVertices * 2 * sizeof(GLfloat), texCoords, GL_STATIC_DRAW);

    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 0, NULL);

    delete[] texCoords;

    // Unbind vertex array object
    glBindVertexArray(0);
}

Mesh::~Mesh()
{
    glDeleteVertexArrays(1, &mVertexArrayObject);
    glDeleteBuffers(1, &mVertexBuffer);
    glDeleteBuffers(1, &mNormalBuffer);
    glDeleteBuffers(1, &mIndexBuffer);
    glDeleteBuffers(1, &mUVBuffer);
}

void Mesh::draw() const
{
    glBindVertexArray(mVertexArrayObject);
    glDrawElements(GL_TRIANGLES, mElementCount, GL_UNSIGNED_INT, NULL);
    glBindVertexArray(0);
}
