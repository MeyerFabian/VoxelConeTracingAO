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
	m_geomPass = std::make_unique<ShaderProgram>("/vertex_shaders/sponza.vert", "/fragment_shaders/sponza.frag");
	m_width = width;
	m_height = height;
	m_gbuffer->init(width, height);
}

void VoxelConeTracing::geometryPass(const std::unique_ptr<Scene>& scene, const float stepSize) const{
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	// Use the one and only shader
	m_geomPass->use();

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

	
}