#include "Voxelization.h"

#include "externals/GLM/glm/glm.hpp"
#include "externals/GLM/glm/gtc/matrix_transform.hpp"

Voxelization::Voxelization(Scene const * pScene, float volumeExtent)
{
    mpScene = pScene;
    mVolumeExtent = volumeExtent;

    // TODO
    // - wahrscheinlich depth test aus
    // - rendering auf double sided

    // ### Shader program ###
    ShaderProgram shader("/vertex_shaders/voxelization.vert","/fragment_shaders/voxelization.frag", "/geometry_shaders/voxelization.geom");

    // ### Transformation ###

    // Orthographic projection
    float halfVolumeExtent = volumeExtent / 2.0f;
    glm::mat4 projection = glm::ortho(
        -halfVolumeExtent,
        halfVolumeExtent,
        -halfVolumeExtent,
        halfVolumeExtent,
        -halfVolumeExtent,
        halfVolumeExtent);

    // ### Atomic counter ###

    // TODO: atomic counter to handle image store access on position, normal and color buffer texture
    // Only needed to determine index

    // ### Buffer texture ###

    // TODO (world pos, normal, color)

    // ### Execution ###

    // Draw scene with voxelization shader

    // TODO: one has to extend scene class to draw scene with custom shader and projection

}
