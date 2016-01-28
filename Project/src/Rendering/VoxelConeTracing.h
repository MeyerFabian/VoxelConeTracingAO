#ifndef VOXELCONETRACING_H
#define VOXELCONETRACING_H
#include <memory>
#include "GBuffer.h"
#include <glm/vec3.hpp>
#include "SparseOctree/NodePool.h"
#include "ShaderProgram.h"
#include "Scene/Scene.h"
class VoxelConeTracing
{
public:
	VoxelConeTracing();
	~VoxelConeTracing();
	void init(float width,float height);
	void geometryPass(const std::unique_ptr<Scene>& scene) const;
	void draw(GLuint ScreenQuad, const GLuint lightViewMapTexture, const std::unique_ptr<Scene>& scene, const NodePool& nodePool, const float stepSize) const;

private:
	void supplyFullScreenQuad();
	std::unique_ptr<ShaderProgram> m_geomPass;
	std::unique_ptr<ShaderProgram> m_voxelConeTracing;
	std::unique_ptr<GBuffer> m_gbuffer;
	float m_width;
	float m_height;
	GLuint vaoID;
};
#endif //VOXELCONETRACING_H