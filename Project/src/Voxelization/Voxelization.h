/* Voxelization of scene. Renders each triangle with maximum area and saves
positions as entry in buffer. Color and normal is averaged and saved in 3D
texture. */

#ifndef VOXELIZATION_H_
#define VOXELIZATION_H_

#include "src/Scene/Scene.h"
#include "src/Voxelization/FragmentList.h"

// TODO:
// -Create own framebuffer to be independend from screen resolution (otherwise,
//  voxelization may not be higher resolution then window

class Voxelization : public Controllable
{
public:

    // Enums
    enum VoxelizeResolutions {RES_256, RES_384, RES_512, RES_1024};

    // Constructor
    Voxelization(App *pApp);

    // Destructor
    ~Voxelization();

    // Methods
    void voxelize(float extent, Scene const * pScene);
    virtual void fillGui();
    FragmentList const * getFragmentList() const;
    void mapFragmentListToCUDA();
    void unmapFragmentListFromCUDA();

    // Members
    int m_voxelizationResolution = RES_512;

private:

    // Methods
    void resetAtomicCounter() const;
    GLuint readAtomicCounter() const;
    unsigned int determineVoxelizeResolution(int res) const;

    // Members
    std::unique_ptr<ShaderProgram> m_upVoxelizationShader;
    std::unique_ptr<FragmentList> m_upFragmentList;
    GLuint m_atomicBuffer;
    GLint m_resolution;
};

#endif // VOXELIZATION_H_
