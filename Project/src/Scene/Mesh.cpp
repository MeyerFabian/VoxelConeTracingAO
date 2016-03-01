#include "Mesh.h"

Mesh::Mesh(aiMesh const * mesh)
{
    // Element count
    m_elementCount = mesh->mNumFaces * 3;

    // Vertex Array Buffer
    m_vertexArrayObject = 0;
    glGenVertexArrays(1, &m_vertexArrayObject);
    glBindVertexArray(m_vertexArrayObject);

    // Vertices
    float *vertices = new float[mesh->mNumVertices * 3];
    for(int i = 0; i < mesh->mNumVertices; i++)
    {
        vertices[i * 3] = mesh->mVertices[i].x * MESH_SCALE;
        vertices[i * 3 + 1] = mesh->mVertices[i].y * MESH_SCALE;
        vertices[i * 3 + 2] = mesh->mVertices[i].z * MESH_SCALE;
    }

    glGenBuffers(1, &m_vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, mesh->mNumVertices * 3 * sizeof(GLfloat), vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    delete[] vertices;

    // Texture coordinates
    float *texCoords = new float[mesh->mNumVertices * 2];
    for(int i = 0; i < mesh->mNumVertices; i++)
    {
        texCoords[i * 2] = mesh->mTextureCoords[0][i].x;
        texCoords[i * 2 + 1] = mesh->mTextureCoords[0][i].y;
    }

    glGenBuffers(1, &m_UVBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, m_UVBuffer);
    glBufferData(GL_ARRAY_BUFFER, mesh->mNumVertices * 2 * sizeof(GLfloat), texCoords, GL_STATIC_DRAW);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

    delete[] texCoords;

    // Normals
    float *normals = new float[mesh->mNumVertices * 3];
    for(int i = 0; i < mesh->mNumVertices; i++)
    {
        normals[i * 3] = mesh->mNormals[i].x;
        normals[i * 3 + 1] = mesh->mNormals[i].y;
        normals[i * 3 + 2] = mesh->mNormals[i].z;
    }

    glGenBuffers(1, &m_normalBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, m_normalBuffer);
    glBufferData(GL_ARRAY_BUFFER, mesh->mNumVertices * 3 * sizeof(GLfloat), normals, GL_STATIC_DRAW);

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    delete[] normals;

    // Tangents
    float *tangents = new float[mesh->mNumVertices * 3];
    for (int i = 0; i < mesh->mNumVertices; i++)
    {
        tangents[i * 3] = mesh->mTangents[i].x;
        tangents[i * 3 + 1] = mesh->mTangents[i].y;
        tangents[i * 3 + 2] = mesh->mTangents[i].z;
    }

    glGenBuffers(1, &m_tangentBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, m_tangentBuffer);
    glBufferData(GL_ARRAY_BUFFER, mesh->mNumVertices * 3 * sizeof(GLfloat), tangents, GL_STATIC_DRAW);

    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    delete[] tangents;

    // Indices
    unsigned int *indices = new unsigned int[mesh->mNumFaces * 3];
    for(int i = 0; i < mesh->mNumFaces; i++)
    {
        indices[i * 3] = mesh->mFaces[i].mIndices[0];
        indices[i * 3 + 1] = mesh->mFaces[i].mIndices[1];
        indices[i * 3 + 2] = mesh->mFaces[i].mIndices[2];
    }

    glGenBuffers(1, &m_indexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh->mNumFaces * 3 * sizeof(GLuint), indices, GL_STATIC_DRAW);

    delete[] indices;

    // Unbind vertex array object
    glBindVertexArray(0);
}

Mesh::~Mesh()
{
    glDeleteVertexArrays(1, &m_vertexArrayObject);
    glDeleteBuffers(1, &m_vertexBuffer);
    glDeleteBuffers(1, &m_normalBuffer);
    glDeleteBuffers(1, &m_indexBuffer);
    glDeleteBuffers(1, &m_UVBuffer);
}

void Mesh::draw() const
{
    glBindVertexArray(m_vertexArrayObject);
    glDrawElements(GL_TRIANGLES, m_elementCount, GL_UNSIGNED_INT, NULL);
    glBindVertexArray(0);
}
