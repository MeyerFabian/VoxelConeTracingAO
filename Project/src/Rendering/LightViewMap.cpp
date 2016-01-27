#include "LightViewMap.h"
#include <stdio.h>

#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/GLM/glm/gtx/string_cast.hpp"
LightViewMap::LightViewMap()
{
	m_width = 0.0f;
	m_height = 0.0f;
	m_depthbuffer = std::make_unique<LightDepthBuffer>();
}


LightViewMap::~LightViewMap()
{
}

void LightViewMap::init(float width, float height)
{
	
	m_shadowMapPass = std::make_unique<ShaderProgram>("/vertex_shaders/lightViewMap_pass.vert", "/fragment_shaders/lightViewMap_pass.frag");
	m_width = width;
	m_height = height;

	m_depthbuffer->init(m_width, m_height);

}

void LightViewMap::shadowMapPass(const std::unique_ptr<Scene>& scene) const{

	glDepthMask(true);
	//Bind the GBuffer before enabling (and texture stuff) else it will fail
	m_depthbuffer->bindForWriting();


	glm::mat4 uniformProjection = glm::perspective(glm::radians(35.0f), m_width / m_height, 0.1f, 300.f);
	glm::mat4 uniformModel = glm::mat4(1.f);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDisable(GL_BLEND);
	// Use the one and only shader
	m_shadowMapPass->use();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Create uniforms used by shader
	// Fill uniforms to shader
	m_shadowMapPass->updateUniform("model", uniformModel); // all meshes have center at 0,0,0
	m_shadowMapPass->updateUniform("projection", uniformProjection);
	m_shadowMapPass->updateUniform("view", scene->getCamera().getViewMatrix());



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