#ifndef LIGHTDEPTHBUFFER_H
#define LIGHTDEPTHBUFFER_H
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>


class LightDepthBuffer
{
public:

	LightDepthBuffer();
	~LightDepthBuffer();

	void init(int width, int height);
	void bindForWriting();
	void bindForReading();
	GLuint getDepthTextureID();
	GLuint m_fbo;
	GLuint m_depthTexture;
};


#endif //LIGHTDEPTHBUFFER_H