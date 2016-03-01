/* Voxelization of scene. Creates list of fragments containing world position, normal and color.
Scene is taken and projected orthogonal. Then, for every triangle the rotation which maximizes
the projection area is found and applied in the geometry shader. In addition, the triangle is
moved to the center of projection to take advantage of the available rasterization area. Results
are save to buffer textures, collection world position, normal and color of each fragment. */

#ifndef VOXELIZATION_H_
#define VOXELIZATION_H_

#include "Scene/Scene.h"
#include "FragmentList.h"

// TODO:
// -Create own framebuffer to be independend from screen resoluation

class Voxelization : public Controllable
{
public:

    enum VoxelizeResolutions {RES_256, RES_384, RES_512, RES_1024};
    int voxelizationResolution = RES_384;

    Voxelization(App *pApp);
    ~Voxelization();

    void voxelize(float extent, Scene const * pScene, FragmentList* pFragmentList);

    virtual void fillGui();

private:

    void resetAtomicCounter() const;
    GLuint readAtomicCounter() const;
    unsigned int determineVoxeliseResolution(int res) const;

    // Members
    std::unique_ptr<ShaderProgram> m_voxelizationShader;
    GLuint m_atomicBuffer;
};

#endif // VOXELIZATION_H_
