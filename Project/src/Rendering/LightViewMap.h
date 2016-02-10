#ifndef LIGHTVIEWMAP_H
#define LIGHTVIEWMAP_H
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>
#include <memory>
#include "LightDepthBuffer.h"
#include "Scene/Scene.h"
#include "ShaderProgram.h"


class LightViewMap : public Controllable
{
public:
	
	LightViewMap(App *pApp);
	~LightViewMap();

	enum ShadowMapResolutions { RES_1024, RES_2048, RES_4096 };
	int SHADOW_MAP_RESOLUTION = RES_2048;
	void init();

	void shadowMapPass(const std::unique_ptr<Scene>& scene) const;
	void shadowMapRender(GLuint RenderWidth, GLuint  RenderHeight, float windowWidth, float windowHeight, GLuint ScreenQuad) const;
	int determineShadowMapResolution(int res) const;
	void fillGui();
	int getCurrentShadowMapRes();
	GLuint getDepthTextureID(){
		return m_depthbuffer->getDepthTextureID();
	}
	std::unique_ptr<LightDepthBuffer> m_depthbuffer;
	std::unique_ptr<ShaderProgram> m_shadowMapPass;
	std::unique_ptr<ShaderProgram> m_shadowMapRender;
};


#endif //LIGHTVIEWMAP_H