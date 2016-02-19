#include "Cube.h"
#include <vector>

Cube::Cube(float size)
{
	m_mode = GL_TRIANGLES;

    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);

    GLuint vertexBufferHandles[3];
    glGenBuffers(3, vertexBufferHandles);

	m_positions.m_vboHandle = vertexBufferHandles[0];
	m_uvs.m_vboHandle = vertexBufferHandles[1];
	m_normals.m_vboHandle = vertexBufferHandles[2];

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferHandles[0]);
    float positions[] = {
		        -size,-size,size, size,-size,size, size,size,size,
		        size,size,size, -size,size,size, -size,-size,size,
		        // Right face
		        size,-size,size, size,-size,-size, size,size,-size,
		        size,size,-size, size,size,size, size,-size,size,
		        // Back face
		        -size,-size,-size, size,-size,-size, size,size,-size,
		        size,size,-size, -size,size,-size, -size,-size,-size,
		        // Left face
		        -size,-size,size, -size,-size,-size, -size,size,-size,
		        -size,size,-size, -size,size,size, -size,-size,size,
		        // Bottom face
		        -size,-size,size, size,-size,size, size,-size,-size,
		        size,-size,-size, -size,-size,-size, -size,-size,size,
		        // Top Face
		        -size,size,size, size,size,size, size,size,-size,
		        size,size,-size, -size,size,-size, -size,size,size,
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    GLfloat uvCoordinates[] = {
        // Front face
        0,0, 1,0, 1,1,
        1,1, 0,1, 0,0,
        // Right face
        0,0, 1,0, 1,1,
        1,1, 0,1, 0,0,
        // Back face
        0,0, 1,0, 1,1,
        1,1, 0,1, 0,0,
        // Left face
        0,0, 1,0, 1,1,
        1,1, 0,1, 0,0,
        // Bottom face
        0,0, 1,0, 1,1,
        1,1, 0,1, 0,0,
        // Top face
        0,0, 1,0, 1,1,
        1,1, 0,1, 0,0,
    };
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferHandles[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(uvCoordinates), uvCoordinates, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

	 GLfloat normals[] = {
        // Front face
		0.0f, 0.0f, 1.0f, 
		0.0f, 0.0f, 1.0f, 
		0.0f, 0.0f, 1.0f, 
		
		0.0f, 0.0f, 1.0f, 
		0.0f, 0.0f, 1.0f, 
		0.0f, 0.0f, 1.0f, 
        // Right face
		1.0f, 0.0f, 0.0f, 
        1.0f, 0.0f, 0.0f, 
		1.0f, 0.0f, 0.0f, 

		1.0f, 0.0f, 0.0f, 
        1.0f, 0.0f, 0.0f, 
		1.0f, 0.0f, 0.0f, 

		// Back face
		0.0f, 0.0f, -1.0f,
        0.0f, 0.0f, -1.0f, 
		0.0f, 0.0f, -1.0f, 

		0.0f, 0.0f, -1.0f,
        0.0f, 0.0f, -1.0f, 
		0.0f, 0.0f, -1.0f, 
		// Left face
		-1.0f, 0.0f, 0.0f, 
        -1.0f, 0.0f, 0.0f, 
		-1.0f, 0.0f, 0.0f, 

		-1.0f, 0.0f, 0.0f, 
        -1.0f, 0.0f, 0.0f, 
		-1.0f, 0.0f, 0.0f, 
        // Bottom face
		0.0f, -1.0f, 0.0f, 
		0.0f, -1.0f, 0.0f, 
		0.0f, -1.0f, 0.0f, 
		
		0.0f, -1.0f, 0.0f, 
		0.0f, -1.0f, 0.0f, 
		0.0f, -1.0f, 0.0f, 
		// Top face
		0.0f, 1.0f, 0.0f, 
		0.0f, 1.0f, 0.0f, 
		0.0f, 1.0f, 0.0f, 
		
		0.0f, 1.0f, 0.0f, 
		0.0f, 1.0f, 0.0f, 
		0.0f, 1.0f, 0.0f, 

    };
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferHandles[2]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(normals), normals, GL_STATIC_DRAW);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(2);
}

Cube::~Cube()
{
	std::vector<GLuint> buffers;
	buffers.push_back(m_positions.m_vboHandle);
	buffers.push_back(m_uvs.m_vboHandle);
	buffers.push_back(m_normals.m_vboHandle);

	glDeleteBuffers(3, &buffers[0]);
}

void Cube::render()
{
    glBindVertexArray(m_vao);
    glDrawArrays(m_mode, 0, 36);
}
