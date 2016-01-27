#ifndef GBUFFER_H
#define GBUFFER_H
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>

// define that return the size of an array
#define ARRAY_SIZE_IN_ELEMENTS(a) (sizeof(a)/sizeof(a[0]))

class GBuffer
{
public:
	enum GBUFFER_TEXTURE_TYPE
	{
		GBUFFER_TEXTURE_TYPE_POSITION,
		GBUFFER_TEXTURE_TYPE_DIFFUSE,
		GBUFFER_TEXTURE_TYPE_NORMAL,
		GBUFFER_TEXTURE_TYPE_TEXCOORD,
		GBUFFER_NUM_TEXTURES
	};
	GBuffer();
	~GBuffer();

	void init(int width, int height);
	void bindForWriting();
	void bindForReading();
	void setReadBuffer(GBUFFER_TEXTURE_TYPE tt);
	void setDepthReadBuffer();
	GLuint getTextureID(GBUFFER_TEXTURE_TYPE tt); 
	GLuint getDepthTextureID();
	GLuint m_fbo;
	GLuint m_textures[GBUFFER_NUM_TEXTURES];
	GLuint m_depthTexture;
};


#endif //GBUFFER_H