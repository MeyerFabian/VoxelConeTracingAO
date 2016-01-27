#ifndef LIGHTVIEWMAP_H
#define LIGHTVIEWMAP_H
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>
#include <memory>
#include "LightDepthBuffer.h"
#include "Scene/Scene.h"
#include "ShaderProgram.h"
class LightViewMap
{
public:
	
	LightViewMap();
	~LightViewMap();

	void init(float width, float height);

	void shadowMapPass(const std::unique_ptr<Scene>& scene) const;
	float m_width;
	float m_height;
	GLuint getDepthTextureID(){
		return m_depthbuffer->getDepthTextureID();
	}
	std::unique_ptr<LightDepthBuffer> m_depthbuffer;

	std::unique_ptr<ShaderProgram> m_shadowMapPass;
};


#endif //LIGHTVIEWMAP_H