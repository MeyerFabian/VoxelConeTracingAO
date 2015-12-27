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

    // TODO: one needs the size of the color ouput texture for sure...

    Voxelization(
        Scene const * pScene,
        float volumeLeft,
        float volumeRight,
        float volumeBottom,
        float volumeTop,
        float volumeNear,
        float volumeFar);

    ~Voxelization();

    GLuint getColorOutputTexture() const;

private:

    // Members
    Scene const * mpScene;
    GLuint mColorOutputBuffer;
    GLuint mColorOutputTexture;

};

#endif // VOXELIZATION_H_