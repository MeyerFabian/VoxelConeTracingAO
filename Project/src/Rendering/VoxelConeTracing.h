#ifndef VOXELCONETRACING_H
#define VOXELCONETRACING_H
#include <memory>
#include "GBuffer.h"
#include <glm/vec3.hpp>
#include "SparseOctree/NodePool.h"
#include "SparseOctree/BrickPool.h"
#include "ShaderProgram.h"
#include "Scene/Scene.h"
class VoxelConeTracing : public Controllable
{
public:
    VoxelConeTracing(App* pApp);
    ~VoxelConeTracing();
    void init(float width,float height);
	void geometryPass(float width, float height, const std::unique_ptr<Scene>& scene);
	void drawSimplePhong(float width, float height, int shadowMapResolution,
		GLuint ScreenQuad, const GLuint lightViewMapTexture,
		const std::unique_ptr<Scene>& scene) const;

	void drawVoxelConeTracing(	float width, float height, int shadowMapResolution, 
				GLuint ScreenQuad, const GLuint lightViewMapTexture, 
				const std::unique_ptr<Scene>& scene, const NodePool& nodePool, 
				BrickPool& brickPool, const float stepSize, const float volumeExtent) const;
	
	void drawAmbientOcclusion(	float width, float height,
							GLuint ScreenQuad, const std::unique_ptr<Scene>& scene, 
							const NodePool& nodePool, const BrickPool& brickPool,
							const float volumeExtent);
	void drawVoxelGlow(float width, float height, GLuint ScreenQuad, const std::unique_ptr<Scene>& scene, const NodePool& nodePool, const BrickPool& brickPool, const float volumeExtent);
	void drawGBuffer(float width, float height);
	void drawGBufferPanels(float width, float height);
	std::unique_ptr<GBuffer>& getGBuffer() { return m_gbuffer; }
    glm::mat4 getProjectionMatrix() {return m_uniformProjection;}

private:
    std::unique_ptr<ShaderProgram> m_geomPass;
    std::unique_ptr<ShaderProgram> m_voxelConeTracing;
	std::unique_ptr<ShaderProgram> m_ambientOcclusion; 
	std::unique_ptr<ShaderProgram> m_phongShading;
	std::unique_ptr<ShaderProgram> m_voxelGlow;
    std::unique_ptr<GBuffer> m_gbuffer;
    glm::mat4 m_uniformProjection;
	GLuint vaoID;
	float directionBeginScale;
	float beginningVoxelSize;
	float ambientOcclusionScale;
	float colorBleeding;
	float maxDistance;
	float lambda;
	void fillGui();
};
#endif //VOXELCONETRACING_H
