#include "GBuffer.h"
#include <stdio.h>
#include <iostream>

GBuffer::GBuffer()
{
	m_fbo = 0;
	m_depthTexture = 0;
}


GBuffer::~GBuffer()
{
	if (m_fbo != 0) {
		glDeleteFramebuffers(1, &m_fbo);
	}

	if (m_textures[0] != 0) {
		glDeleteTextures(ARRAY_SIZE_IN_ELEMENTS(m_textures), m_textures);
	}

	if (m_depthTexture != 0) {
		glDeleteTextures(1, &m_depthTexture);
	}
}

void GBuffer::init(int width, int height)
{
	
	//GL_COLOR_ATTACHMENT0 pos
	//GL_COLOR_ATTACHMENT1 dif
	//GL_COLOR_ATTACHMENT2 nor
	//GL_COLOR_ATTACHMENT3 tex
	//GL_DEPTH_ATTACHMENT depth

	// First create FBO 
	glGenFramebuffers(1, &m_fbo);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_fbo);

	// Create GBuffer to store our geometry information textures
	glGenTextures(ARRAY_SIZE_IN_ELEMENTS(m_textures), m_textures);
	glGenTextures(1, &m_depthTexture);
	for (unsigned int i = 0; i < ARRAY_SIZE_IN_ELEMENTS(m_textures); i++)
	{
		glBindTexture(GL_TEXTURE_2D, m_textures[i]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, m_textures[i], 0);

	}
	glBindTexture(GL_TEXTURE_2D, m_depthTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_depthTexture, 0);

	GLenum DrawBuffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 };
	glDrawBuffers(ARRAY_SIZE_IN_ELEMENTS(DrawBuffers), DrawBuffers);

	GLenum Status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (Status != GL_FRAMEBUFFER_COMPLETE)
	{
		printf("FB error: 0x%x\n", Status);
	}
	else
	{
		printf("GBuffer successfully initialized.\n");
	}
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	

}
void GBuffer::bindForReading()
{
	glBindFramebuffer(GL_READ_FRAMEBUFFER, m_fbo);
}

void GBuffer::bindForWriting()
{
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_fbo);
}

void GBuffer::setReadBuffer(GBUFFER_TEXTURE_TYPE tt)
{
	glReadBuffer(GL_COLOR_ATTACHMENT0 + tt);
}

GLuint GBuffer::getTextureID(GBUFFER_TEXTURE_TYPE tt){
	return m_textures[tt];
}
GLuint GBuffer::getDepthTextureID(){
	return m_depthTexture;
}