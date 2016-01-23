#ifndef VOXELCONETRACING_H
#define VOXELCONETRACING_H
#include <memory>
#include "GBuffer.h"
class VoxelConeTracing
{
public:
	VoxelConeTracing();
	~VoxelConeTracing();
private:
	std::unique_ptr<GBuffer> gbuffer;
};

#endif //VOXELCONETRACING_H