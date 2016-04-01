/* Visualization of sparse voxel octree using cubes. Cubes are generated in
 geometry shader. */

#ifndef VOXEL_CUBES_H
#define VOXEL_CUBES_H

#include "src/Rendering/ShaderProgram.h"
#include "src/Scene/Camera.h"
#include "src/SparseOctree/BrickPool.h"
#include "src/SparseOctree/NodePool.h"

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
