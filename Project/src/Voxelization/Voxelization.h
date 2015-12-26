/* Voxelization of scene. Creates list of fragments containing world position, normal and color.
Scene is taken and projected orthogonal. Then, for every triangle the rotation which maximizes
the projection area is found and applied in the geometry shader. In addition, the triangle is
moved to the center of projection to take advantage of the available rasterization area. Results
are save to buffer textures, collection world position, normal and color of each fragment. */

#ifndef VOXELIZATION_H_
#define VOXELIZATION_H_

#include "Scene/Scene.h"
class Voxelization
{
public:

    Voxelization(Scene const * pScene, float volumeExtent);

private:

    // Members
    Scene const * mpScene;
    float mVolumeExtent; // Extent in all three dimensions
};

#endif // VOXELIZATION_H_
