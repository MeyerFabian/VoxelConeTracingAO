#include "VoxelConeTracing.h"

#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"

#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"
#include "externals/GLM/glm/gtx/string_cast.hpp"

VoxelConeTracing::VoxelConeTracing()
{
	m_gbuffer = std::make_unique<GBuffer>();
}


VoxelConeTracing::~VoxelConeTracing()
{
}
void VoxelConeTracing::init(float width, float height) {
	// Prepare the one and only shader
	m_geomPass = std::make_unique<ShaderProgram>("/vertex_shaders/geom_pass.vert", "/fragment_shaders/geom_pass.frag");
	m_width = width;
	m_height = height;

	m_gbuffer->init(m_width, m_height);
}

void VoxelConeTracing::geometryPass(const std::unique_ptr<Scene>& scene, const float stepSize) const{

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	m_gbuffer->bindForWriting();
	// Use the one and only shader
	m_geomPass->use();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Create uniforms used by shader
	glm::mat4 uniformProjection = glm::perspective(glm::radians(35.0f), m_width / m_height, 0.1f, 300.f);
	glm::mat4 uniformModel = glm::mat4(1.f);

	// Fill uniforms to shader
	m_geomPass->updateUniform("projection", uniformProjection);
	m_geomPass->updateUniform("view", scene->getCamera().getViewMatrix());
	m_geomPass->updateUniform("model", uniformModel); // all meshes have center at 0,0,0



	// Render all the buckets' content
	for (auto& bucket : scene->getRenderBuckets())
	{
		// Bind material of bucket (which binds its uniforms and textures)
		bucket.first->bind(m_geomPass.get());

		// Draw all meshes in that bucket
		for (Mesh const * pMesh : bucket.second)
		{
			pMesh->draw();
		}
	}
	m_geomPass->disable();
}
void VoxelConeTracing::deferredShadingPass(const NodePool& nodePool) const{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(1.0, 1.0, 1.0, 1.0f);
	m_gbuffer->bindForReading();

	m_gbuffer->setReadBuffer(GBuffer::GBUFFER_TEXTURE_TYPE_POSITION);
	glBlitFramebuffer(0, 0, (GLint)m_width, (GLint)m_height, 0, 0, 150, 150, GL_COLOR_BUFFER_BIT, GL_LINEAR);


	m_gbuffer->setReadBuffer(GBuffer::GBUFFER_TEXTURE_TYPE_DIFFUSE);
	glBlitFramebuffer(0, 0, (GLint)m_width, (GLint)m_height, 150, 0, 300, 150, GL_COLOR_BUFFER_BIT, GL_LINEAR);

	m_gbuffer->setReadBuffer(GBuffer::GBUFFER_TEXTURE_TYPE_NORMAL);
	glBlitFramebuffer(0, 0, (GLint)m_width, (GLint)m_height, 300, 0, 450, 150, GL_COLOR_BUFFER_BIT, GL_LINEAR);

	m_gbuffer->setReadBuffer(GBuffer::GBUFFER_TEXTURE_TYPE_TEXCOORD);
	glBlitFramebuffer(0, 0, (GLint)m_width, (GLint)m_height, 450, 0, 600, 150, GL_COLOR_BUFFER_BIT, GL_LINEAR);
}