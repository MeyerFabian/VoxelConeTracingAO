#ifndef VOXELCUBES_H
#define VOXELCUBES_H

#include <memory>
#include "Rendering/ShaderProgram.h"
#include "Scene/Camera.h"

class VoxelCubes
{
public:
    VoxelCubes(Camera const * pCamera);
    void draw(float width, float height, float volumeExtent) const;

private:

    Camera const * mpCamera;
    std::unique_ptr<ShaderProgram> mupShaderProgram;
};

#endif // VOXELCUBES_H
