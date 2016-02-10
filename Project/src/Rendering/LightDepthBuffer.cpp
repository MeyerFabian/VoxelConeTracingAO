#include "LightDepthBuffer.h"
#include <stdio.h>
#include <iostream>

LightDepthBuffer::LightDepthBuffer()
{
	m_width = 0;
	m_height = 0;
	m_fbo = 0;
	m_depthTexture = 0;
	glGenFramebuffers(1, &m_fbo);
}


LightDepthBuffer::~LightDepthBuffer()
{
	clear();
	if (m_fbo != 0) {
		glDeleteFramebuffers(1, &m_fbo);
	}


}
void LightDepthBuffer::clear(){
	if (m_depthTexture != 0) {
		glDeleteTextures(1, &m_depthTexture);
	}
}
void LightDepthBuffer::onResize(int width, int height){
	if (m_width != width || m_height != height){
		clear();
		init(width, height);
	}
}

void LightDepthBuffer::init(int width, int height)
{

	m_width = width;
	m_height = height;

	// First create FBO 
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_fbo);

	// Create GBuffer to store our geometry information textures
	glGenTextures(1, &m_depthTexture);

	glBindTexture(GL_TEXTURE_2D, m_depthTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_depthTexture, 0);

	GLenum Status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (Status != GL_FRAMEBUFFER_COMPLETE)
	{
		printf("FB error: 0x%x\n", Status);
	}
	else
	{
		printf("LightDepthBuffer successfully initialized.\n");
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