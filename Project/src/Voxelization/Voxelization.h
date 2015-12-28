/* Voxelization of scene. Creates list of fragments containing world position, normal and color.
Scene is taken and projected orthogonal. Then, for every triangle the rotation which maximizes
the projection area is found and applied in the geometry shader. In addition, the triangle is
moved to the center of projection to take advantage of the available rasterization area. Results
are save to buffer textures, collection world position, normal and color of each fragment. */

#ifndef VOXELIZATION_H_
#define VOXELIZATION_H_

#include "Scene/Scene.h"
#include "FragmentList.h"

class Voxelization
{
public:

    // TODO: one needs the size of the color ouput texture for sure...

    Voxelization();
    ~Voxelization();

    GLuint getColorOutputTexture() const;

    void voxelize(Scene const * pScene,
                  float volumeLeft,
                  float volumeRight,
                  float volumeBottom,
                  float volumeTop,
                  float volumeNear,
                  float volumeFar);

    const FragmentList* getFragmentList() const;

private:

    // Members
    Scene const * mpScene;
    std::unique_ptr<ShaderProgram> mVoxelizationShader;
    GLuint mColorOutputBuffer;
    GLuint mColorOutputTexture;
    GLuint mAtomicBuffer;

    FragmentList mFragmentList;

    void resetAtomicCounter() const;

    GLuint readAtomicCounter() const;
};

#endif // VOXELIZATION_H_
