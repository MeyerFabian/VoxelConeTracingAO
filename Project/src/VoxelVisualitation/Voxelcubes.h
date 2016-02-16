#ifndef VOXELCUBES_H
#define VOXELCUBES_H

#include <memory>
#include "Voxelization/FragmentList.h"
#include "Rendering/ShaderProgram.h"

class VoxelCubes
{
public:
    VoxelCubes();
    void drawVoxel(FragmentList &fragmentList);

    std::unique_ptr<ShaderProgram> mupVoxelCubeShader;
};

#endif // VOXELCUBES_H
