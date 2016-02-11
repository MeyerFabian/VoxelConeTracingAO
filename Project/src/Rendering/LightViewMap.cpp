#include "LightViewMap.h"
#include <stdio.h>


#ifdef __unix__
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
	return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
#endif

using namespace std;

LightViewMap::LightViewMap(App *pApp) : Controllable(pApp, "Light-View-Map")
{
	m_depthbuffer = make_unique<LightDepthBuffer>();

}


LightViewMap::~LightViewMap()
{
}

void LightViewMap::init()
{
	m_shadowMapPass = make_unique<ShaderProgram>("/vertex_shaders/lightViewMap_pass.vert", "/fragment_shaders/lightViewMap_pass.frag");
	m_shadowMapRender= make_unique<ShaderProgram>("/vertex_shaders/shadowMapRender.vert", "/fragment_shaders/shadowMapRender.frag");

	m_depthbuffer->init(
		determineShadowMapResolution(SHADOW_MAP_RESOLUTION),
		determineShadowMapResolution(SHADOW_MAP_RESOLUTION));

}

void LightViewMap::shadowMapPass(const std::unique_ptr<Scene>& scene) const{

	int res = determineShadowMapResolution(SHADOW_MAP_RESOLUTION);
	m_depthbuffer->onResize(res, res);

	glDepthMask(true);

	//Bind the GBuffer before enabling (and texture stuff) else it will fail
	m_depthbuffer->bindForWriting();

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDisable(GL_BLEND);


	// Use the one and only shader
	m_shadowMapPass->use();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//Set the texture size of the light view map
	scene->getLight().setProjectionMatrix(res, res);

	// Create uniforms used by shader
	// Fill uniforms to shader
	m_shadowMapPass->updateUniform("LightProjection", scene->getLight().getProjectionMatrix());
	m_shadowMapPass->updateUniform("LightView", scene->getLight().getViewMatrix());



	// Render all the buckets' content
	for (auto& bucket : scene->getRenderBuckets())
	{
		// Bind material of bucket (which binds its uniforms and textures)
		bucket.first->bind(m_shadowMapPass.get());

		// Draw all meshes in that bucket
		for (Mesh const * pMesh : bucket.second)
		{
			pMesh->draw();
		}
	}
	m_shadowMapPass->disable();

}

void LightViewMap::shadowMapRender(GLuint RenderWidth, GLuint  RenderHeight, float windowWidth, float windowHeight, GLuint ScreenQuad) const{

	int res = determineShadowMapResolution(SHADOW_MAP_RESOLUTION);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glEnable(GL_BLEND);
	//glBlendEquation(GL_FUNC_ADD);
	//glBlendFunc(GL_ONE, GL_ONE); //BLEND_FUNCTION BY OPENGL MAY USE (GL_SRC_ALPHA/GL_ONE_MINUS_SRC_ALPHA) for transparency




	glm::mat4 WVP = glm::mat4(1.f);

	m_shadowMapRender->use();
	m_shadowMapRender->updateUniform("identity", WVP);
	m_shadowMapRender->updateUniform("screenSize", glm::vec2(RenderWidth, RenderHeight));
	m_shadowMapRender->updateUniform("shadowToWindowRatio", glm::vec2(windowWidth / (float)res, windowHeight / (float)res));
	m_shadowMapRender->addTexture("LightViewMapTex", m_depthbuffer->getDepthTextureID());
	glViewport(0, 0, RenderWidth, RenderHeight);
	glBindVertexArray(ScreenQuad);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);

	m_shadowMapRender->disable();

}
int LightViewMap::determineShadowMapResolution(int res) const{
	switch (res)
	{
		case ShadowMapResolutions::RES_1024:
			return 1024;
		case ShadowMapResolutions::RES_2048:
			return 2048;
		case ShadowMapResolutions::RES_4096:
			return 4096;
		default:
			return 2048;
	}
}
int LightViewMap::getCurrentShadowMapRes(){
	return determineShadowMapResolution(SHADOW_MAP_RESOLUTION);
}
void LightViewMap::fillGui()
{
	ImGui::Combo("ShadowMap Resolution", &SHADOW_MAP_RESOLUTION, " 1024x1024\0 2048x2048\0 4096x4096 \0");
}
