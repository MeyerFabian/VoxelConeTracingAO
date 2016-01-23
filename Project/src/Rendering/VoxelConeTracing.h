#ifndef VOXELCONETRACING_H
#define VOXELCONETRACING_H
#include <memory>
#include "GBuffer.h"
#include <glm\vec3.hpp>
#include "SparseOctree\NodePool.h"
class VoxelConeTracing
{
public:
	VoxelConeTracing();
	~VoxelConeTracing();
	void geometryPass() const;
	void deferredShadingPass(const glm::vec3 camPos, const NodePool& nodePool, const float stepSize) const;

private:
	std::unique_ptr<GBuffer> gbuffer;
};

#endif //VOXELCONETRACING_H