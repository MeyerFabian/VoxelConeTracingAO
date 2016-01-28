#include "FullScreenQuad.h"


FullScreenQuad::FullScreenQuad()
{

	glGenVertexArrays(1, &m_vaoID);
	glBindVertexArray(m_vaoID);


	const GLfloat plane_vert_data[] = {
		-1.0f, -1.0f,
		+1.0f, -1.0f,
		-1.0f, +1.0f,
		+1.0f, +1.0f,
	};
	GLuint mBufferID;
	glGenBuffers(1, &mBufferID);
	glBindBuffer(GL_ARRAY_BUFFER, mBufferID);
	glBufferData(GL_ARRAY_BUFFER, sizeof(plane_vert_data), plane_vert_data, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);

	glBindVertexArray(0);
}
FullScreenQuad::~FullScreenQuad()
{
	
}
GLuint FullScreenQuad::getvaoID() const
{
	return m_vaoID;
}