#include "LightViewMap.h"
#include <stdio.h>


#ifdef __unix__
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
	return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
#endif

using namespace std;

LightViewMap::LightViewMap()
{
	m_width = 0.0f;
	m_height = 0.0f;
	m_depthbuffer = make_unique<LightDepthBuffer>();

}


LightViewMap::~LightViewMap()
{
}

void LightViewMap::init(float width, float height)
{
	
	m_shadowMapPass = make_unique<ShaderProgram>("/vertex_shaders/lightViewMap_pass.vert", "/fragment_shaders/lightViewMap_pass.frag");
	m_shadowMapRender= make_unique<ShaderProgram>("/vertex_shaders/shadowMapRender.vert", "/fragment_shaders/shadowMapRender.frag");
	m_width = width;
	m_height = height;

	m_depthbuffer->init(m_width, m_height);

}

void LightViewMap::shadowMapPass(const std::unique_ptr<Scene>& scene) const{

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
	scene->getLight().setProjectionMatrix(m_width, m_height);

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

	glDepthMask(GL_FALSE);
	glDisable(GL_DEPTH_TEST);
}

void LightViewMap::shadowMapRender(GLuint ScreenQuad) const{

	GLuint RenderWidth = 150, RenderHeight = 150;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//glEnable(GL_BLEND);
	//glBlendEquation(GL_FUNC_ADD);
	//glBlendFunc(GL_ONE, GL_ONE); //BLEND_FUNCTION BY OPENGL MAY USE (GL_SRC_ALPHA/GL_ONE_MINUS_SRC_ALPHA) for transparency




	glm::mat4 WVP = glm::mat4(1.f);

	m_shadowMapRender->use();
	m_shadowMapRender->updateUniform("identity", WVP);
	m_shadowMapRender->updateUniform("screenSize", glm::vec2(RenderWidth, RenderHeight));
	m_shadowMapRender->addTexture("LightViewMapTex", m_depthbuffer->getDepthTextureID());
	glViewport(0, 0, RenderWidth, RenderHeight);
	glBindVertexArray(ScreenQuad);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);

	m_shadowMapRender->disable();

}
