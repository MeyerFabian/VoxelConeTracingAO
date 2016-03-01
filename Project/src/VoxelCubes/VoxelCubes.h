#ifndef VOXEL_CUBES_H
#define VOXEL_CUBES_H

#include "Rendering/ShaderProgram.h"
#include "Scene/Camera.h"
#include "SparseOctree/BrickPool.h"
#include "SparseOctree/NodePool.h"

#include <memory>

class VoxelCubes
{
public:

    VoxelCubes(Camera const * pCamera);
    void draw(
        float width,
        float height,
        float volumeExtent,
        NodePool& nodePool,
        BrickPool& brickPool) const;

private:

    Camera const * m_pCamera;
    std::unique_ptr<ShaderProgram> m_upShaderProgram;
};

#endif // VOXEL_CUBES_H
