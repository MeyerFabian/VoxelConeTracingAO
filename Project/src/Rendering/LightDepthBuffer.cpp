#include "LightDepthBuffer.h"
#include <stdio.h>
#include <iostream>

LightDepthBuffer::LightDepthBuffer()
{
	m_fbo = 0;
	m_depthTexture = 0;
}


LightDepthBuffer::~LightDepthBuffer()
{
	if (m_fbo != 0) {
		glDeleteFramebuffers(1, &m_fbo);
	}


	if (m_depthTexture != 0) {
		glDeleteTextures(1, &m_depthTexture);
	}
}

void LightDepthBuffer::init(int width, int height)
{
	

	// First create FBO 
	glGenFramebuffers(1, &m_fbo);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_fbo);

	// Create GBuffer to store our geometry information textures
	glGenTextures(1, &m_depthTexture);

	glBindTexture(GL_TEXTURE_2D, m_depthTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_depthTexture, 0);

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
void LightDepthBuffer::bindForReading()
{
	glBindFramebuffer(GL_READ_FRAMEBUFFER, m_fbo);
}

void LightDepthBuffer::bindForWriting()
{
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_fbo);
}


GLuint LightDepthBuffer::getDepthTextureID(){
	return m_depthTexture;
}